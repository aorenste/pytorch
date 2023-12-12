# Owner(s): ["oncall: distributed"]

from enum import auto, Enum

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
    get_state_dict,
)
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin


# Simple and boring model
class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(8, 16)
        self.net2 = nn.Linear(16, 32)
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Linear(64, 8)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        return x

    def get_input(self):
        return torch.rand(8, 8, device="cuda")


class TestStatefulObj:
    def __init__(self):
        self.data = torch.rand(10, 10, device="cuda")

    def state_dict(self):
        return {"data": self.data}

    def load_state_dict(self, state_dict):
        self.data = state_dict["data"]

    def __eq__(self, other):
        return torch.equal(self.data, other.data)


class ModelType(Enum):
    FSDP = auto()
    HSDP = auto()
    FSDP_TP = auto()
    NONE = auto()  # no parallelization


def _train(model, optim, train_steps=1):
    torch.manual_seed(0)
    loss = None
    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss


class TestE2ELoadAndSave(DTensorTestBase, VerifyStateDictMixin):
    def _create_model(self, compile, model_type, options):
        dummy_model = TestDummyModel().cuda()

        assert model_type in ModelType, f"{model_type} is not supported."
        if model_type == ModelType.FSDP:
            device_mesh = init_device_mesh(self.device_type, (self.world_size,))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
            )
        elif model_type == ModelType.HSDP:
            device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )
        elif model_type == ModelType.FSDP_TP:
            mesh_2d = init_device_mesh(
                self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
            )
            tp_mesh = mesh_2d["tp"]
            dp_mesh = mesh_2d["dp"]
            parallelize_plan = {
                "net1": ColwiseParallel(),
                "net2": RowwiseParallel(),
            }
            model = parallelize_module(dummy_model, tp_mesh, parallelize_plan)
            model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
        else:
            model = dummy_model

        if compile:
            # TODO: enable dynamic=True when dynamic shape support is enabled.
            # model = torch.compile(model)
            model = torch.compile(model, dynamic=False)

        optim = self._optim(model)
        if model_type is not ModelType.NONE:
            _patch_model_state_dict(model, options=options)
            _patch_optimizer_state_dict(model, optimizers=optim, options=options)

        return model, optim

    def _optim(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    @parametrize("compile", [True, False])
    # TODO: Previously PariwiseParallel does not shard properly, passing ModelType.FSDP_TP test where it
    # should have failed. Disabling the failed test temporarily to unblock the deprecation of PairwiseParallel.
    # @parametrize("model_type", [ModelType.FSDP, ModelType.HSDP, ModelType.FSDP_TP])
    @parametrize("model_type", [ModelType.FSDP, ModelType.HSDP])
    def test_e2e(self, compile, model_type):
        model, optim = self._create_model(compile, ModelType.NONE)
        _train(model, optim, train_steps=2)

        dist_model, dist_optim = self._create_model(compile, model_type)
        _train(dist_model, dist_optim, train_steps=2)

        original_stateful_obj = TestStatefulObj()  # tests arbitrary saving/loading

        checkpointer = DCP.FileSystemCheckpointer(self.temp_dir)
        checkpointer.save(
            state_dict={
                "model": dist_model,
                "optimizer": dist_optim,
                "s": original_stateful_obj,
            }
        )

        loaded_stateful_obj = TestStatefulObj()
        dist_model, dist_optim = self._create_model(compile, model_type)

        checkpointer.load(
            state_dict={
                "model": dist_model,
                "optimizer": dist_optim,
                "s": loaded_stateful_obj,
            }
        )

        self.assertEqual(original_stateful_obj, loaded_stateful_obj)

        # train one more step on both models
        loss = _train(model, optim, train_steps=1)
        dist_loss = _train(dist_model, dist_optim, train_steps=1)
        self.assertEqual(loss, dist_loss)

        dist_msd, dist_osd = get_state_dict(dist_model, optimizers=dist_optim)
        model_sd, optim_sd = get_state_dict(model, optimizers=optim)

        self._verify_msd(model_sd, dist_msd)
        self._verify_osd_by_load(model, optim, self._optim(model), dist_osd)

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_different_ordered_state_dict_keys(self):
        """Tests that the order of keys in the state dict does not matter when loading
        If order was not accounted for, the following test would cause a deadlock.
        """

        world_size = self.world_size

        class Foo:
            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                tl = [
                    torch.ones(2, dtype=torch.int64, device="cuda")
                    for _ in range(world_size)
                ]
                t = (
                    torch.arange(2, dtype=torch.int64, device="cuda")
                    + 1
                    + 2 * dist.get_rank()
                )
                dist.all_gather(tl, t, async_op=False)

        class Bar:
            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                tensor = (
                    torch.arange(2, dtype=torch.int64, device="cuda")
                    + 1
                    + 2 * dist.get_rank()
                )
                dist.all_reduce(tensor, op=ReduceOp.SUM)

        if self.rank == 0:
            sd = {
                "A": Foo(),
                "B": Bar(),
            }
        else:
            sd = {
                "B": Bar(),
                "A": Foo(),
            }

        DCP.save(sd, DCP.FileSystemWriter(self.temp_dir))
        DCP.load(sd, DCP.FileSystemReader(self.temp_dir))

    # uncomment to remove deadlock
    # @property
    # def backend(self):
    #     return "cpu:gloo,cuda:nccl"

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_async(self):
        """Tests that the order of keys in the state dict does not matter when loading
        If order was not accounted for, the following test would cause a deadlock.
        """

        from torch.distributed.checkpoint.state_dict import get_model_state_dict
        from torch.distributed.checkpoint.state_dict import StateDictOptions
        from concurrent.futures import ThreadPoolExecutor

        model = TestDummyModel().cuda()
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = FSDP(
            model,
            device_mesh=device_mesh,
            use_orig_params=True,
        )
        msd = get_model_state_dict(model, options=StateDictOptions(cpu_offload=True))

        sd = {"model": msd}
        checkpointer = DCP.FileSystemCheckpointer(self.temp_dir)

        executor = ThreadPoolExecutor(max_workers=1)
        f = executor.submit(checkpointer.save, state_dict=sd)

        print(f.result())


    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_deadlock(self):
        from concurrent.futures import ThreadPoolExecutor

        def foo():
            # the first collective we hit in `checkpoint.save` is `gather_object`, so testing here
            gather_objects = ["foo", 12, {1: 2}, {2:3}]
            output = [None for _ in gather_objects]
            dist.gather_object(
                gather_objects[dist.get_rank()],
                output if dist.get_rank() == 0 else None
            )
            return output

        executor = ThreadPoolExecutor(max_workers=1)
        f = executor.submit(foo)

        print(f.result())

    def test_pg_init(self):
        # same as above but we init pg and use env instead of the temp file

        import os
        import tempfile
        from concurrent.futures import ThreadPoolExecutor

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # backend = "cpu:gloo,cuda:nccl" # no deadlock
        backend = "nccl" # raises exception for Duplicate GPU if default init_method, deadlocks on file
        # backend = "gloo" # no deadlock

        init_file = None
        # uncomment to cause a deadlock if backend == nccl
        # init_file = tempfile.NamedTemporaryFile(delete=False)
        # init_file = f"file://{init_file.name}"

        dist.init_process_group(
            backend=backend,
            rank=self.rank,
            world_size=4,
            init_method=init_file
        )

        if "nccl" in backend:
            torch.cuda.set_device(self.rank)

        def foo():
            gather_objects = ["foo", 12, {1: 2}, {2:3}]
            output = [None for _ in gather_objects]
            dist.gather_object(
                gather_objects[dist.get_rank()],
                output if dist.get_rank() == 0 else None
            )
            return output

        # works
        print("running sync")
        foo()
        print("done running sync")

        # does not work for backend == nccl
        # (which is probably expected but the deadlock when using a file as init method is a potentially an issue)
        print("running async")
        executor = ThreadPoolExecutor(max_workers=1)
        f = executor.submit(foo)

        print(f.result())
        print("done running async")






instantiate_parametrized_tests(TestE2ELoadAndSave)
if __name__ == "__main__":
    run_tests()

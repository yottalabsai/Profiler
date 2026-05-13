# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /root/Profiler/gpt2_optimized_inductor_debug/pu/cpuxmuffbp4w3m5mqvgksmqzfmf23m36fngulvdhtqh2xjpaaorr.py
# Topologically Sorted Source Nodes: [embedding, arange, add, unsqueeze, embedding_1, add_1], Original ATen: [aten.embedding, aten.arange, aten.add, aten.unsqueeze]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   arange => iota
#   embedding => embedding
#   embedding_1 => embedding_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg0_1 : Tensor "i64[4, 128][128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[50257, 768][768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %embedding : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 0), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
#   %embedding_1 : Tensor "bf16[1, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %unsqueeze), kwargs = {})
#   %add_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   return %add_1
triton_poi_fused_add_arange_embedding_unsqueeze_0 = async_compile.triton('triton_poi_fused_add_arange_embedding_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_embedding_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1769472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_embedding_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex // 768
    x0 = (xindex % 768)
    x4 = (xindex % 98304)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.full([XBLOCK], 50257, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 50257), "index out of bounds: 0 <= tmp4 < 50257")
    tmp6 = tl.load(in_ptr1 + (x0 + 768*tmp4), None).to(tl.float32)
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x5), tmp8, None)
''', device_str='cuda')


# kernel path: /root/Profiler/gpt2_optimized_inductor_debug/wv/cwvkg65mjxigtkwadegulbribddkewyalrnyjwdnvvtytscne5b2.py
# Topologically Sorted Source Nodes: [arange_4, add_3, getitem_1, arange_3, add_2, getitem, le], Original ATen: [aten.arange, aten.add, aten.unsqueeze, aten.le]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   arange_3 => iota_3
#   arange_4 => iota_4
#   getitem => unsqueeze_1, unsqueeze_2, unsqueeze_3
#   getitem_1 => unsqueeze_4, unsqueeze_5, unsqueeze_6
#   le => le
# Graph fragment:
#   %iota_4 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_3 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_4, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_3, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 2), kwargs = {})
#   %iota_3 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_2 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_3, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %le : Tensor "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_6, %unsqueeze_3), kwargs = {})
#   return %le
triton_poi_fused_add_arange_le_unsqueeze_1 = async_compile.triton('triton_poi_fused_add_arange_le_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_le_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_le_unsqueeze_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 <= tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride(arg0_1, (4, 128), (128, 1))
        assert_size_stride(arg1_1, (50257, 768), (768, 1))
        assert_size_stride(arg2_1, (1024, 768), (768, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((4, 128, 768), (98304, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [embedding, arange, add, unsqueeze, embedding_1, add_1], Original ATen: [aten.embedding, aten.arange, aten.add, aten.unsqueeze]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_arange_embedding_unsqueeze_0.run(arg0_1, arg1_1, arg2_1, buf0, 393216, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            buf1 = empty_strided_cuda((1, 1, 128, 128), (16384, 1, 128, 1), torch.bool)
            # Topologically Sorted Source Nodes: [arange_4, add_3, getitem_1, arange_3, add_2, getitem, le], Original ATen: [aten.arange, aten.add, aten.unsqueeze, aten.le]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_arange_le_unsqueeze_1.run(buf1, 16384, stream=stream0)
        return (buf0, reinterpret_tensor(buf1, (4, 1, 128, 128), (0, 16384, 128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

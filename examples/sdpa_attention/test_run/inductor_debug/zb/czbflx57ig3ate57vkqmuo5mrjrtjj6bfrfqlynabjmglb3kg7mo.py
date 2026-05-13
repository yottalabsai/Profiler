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


# kernel path: /root/Profiler/examples/sdpa_attention/test_run/inductor_debug/nw/cnw2kz2jvxw3bxs2o4vv5ddhn7duxztss7rkgb77e746qoyfmd3h.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %arg0_1 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %getitem_1 : Tensor "f32[8, 512, 1][512, 1, 4096]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[8, 512, 1][512, 1, 4096]cuda:0" = PlaceHolder[target=buf1]
#   %arg1_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg0_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %getitem_1), kwargs = {})
#   %add : Tensor "f32[8, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[8, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg1_1), kwargs = {})
#   %add_1 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg2_1), kwargs = {})
#   return %getitem_1,%buf1,%add_1
triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_per_fused_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 25169920}}
)
@triton.jit
def triton_per_fused_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None)
    tmp21 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tl.full([1, 1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = (tmp5 / tmp7)
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp14 = tmp0 - tmp8
    tmp15 = tl.full([1, 1], 512.0, tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tl.full([1, 1], 1e-05, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr2 + (r0_1 + 512*x0), tmp24, None)
''', device_str='cuda')


# kernel path: /root/Profiler/examples/sdpa_attention/test_run/inductor_debug/c5/cc5k3ymo3yrba2ftqp2ru4by2ld3bssoslntpxzohgh2isd3n5oy.py
# Topologically Sorted Source Nodes: [linear_3, add, layer_norm_1], Original ATen: [aten._unsafe_view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_2
#   layer_norm_1 => add_3, add_4, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   linear_3 => view_11
# Graph fragment:
#   %mm_3 : Tensor "f32[4096, 512][512, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %arg0_1 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %getitem_7 : Tensor "f32[8, 512, 1][512, 1, 4096]cuda:0" = PlaceHolder[target=getitem_7]
#   %buf14 : Tensor "f32[8, 512, 1][512, 1, 4096]cuda:0" = PlaceHolder[target=buf14]
#   %arg7_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %view_11 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [8, 512, 512]), kwargs = {})
#   %add_2 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %arg0_1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_7), kwargs = {})
#   %add_3 : Tensor "f32[8, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[8, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_2 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg7_1), kwargs = {})
#   %add_4 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg8_1), kwargs = {})
#   return %getitem_7,%buf14,%add_4
triton_per_fused__unsafe_view_add_native_layer_norm_1 = async_compile.triton('triton_per_fused__unsafe_view_add_native_layer_norm_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_native_layer_norm_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 33558528}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_native_layer_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 512*x0), None)
    tmp23 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = tmp2 - tmp10
    tmp17 = tl.full([1, 1], 512.0, tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tl.full([1, 1], 1e-05, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp26, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
        args.clear()
        assert_size_stride(arg0_1, (8, 512, 512), (262144, 512, 1))
        assert_size_stride(arg1_1, (512, ), (1, ))
        assert_size_stride(arg2_1, (512, ), (1, ))
        assert_size_stride(arg3_1, (512, 512), (512, 1))
        assert_size_stride(arg4_1, (512, 512), (512, 1))
        assert_size_stride(arg5_1, (512, 512), (512, 1))
        assert_size_stride(arg6_1, (512, 512), (512, 1))
        assert_size_stride(arg7_1, (512, ), (1, ))
        assert_size_stride(arg8_1, (512, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((8, 512, 512), (262144, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_0.run(arg0_1, arg1_1, arg2_1, buf3, 4096, 512, stream=stream0)
            del arg1_1
            del arg2_1
            buf4 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (4096, 512), (512, 1), 0), reinterpret_tensor(arg3_1, (512, 512), (1, 512), 0), out=buf4)
            del arg3_1
            buf5 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (4096, 512), (512, 1), 0), reinterpret_tensor(arg4_1, (512, 512), (1, 512), 0), out=buf5)
            del arg4_1
            buf6 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (4096, 512), (512, 1), 0), reinterpret_tensor(arg5_1, (512, 512), (1, 512), 0), out=buf6)
            del arg5_1
            del buf3
            # Topologically Sorted Source Nodes: [linear, view, transpose, linear_1, view_1, transpose_1, linear_2, view_2, transpose_2, scaled_dot_product_attention], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf4, (8, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf5, (8, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf6, (8, 8, 512, 64), (262144, 64, 512, 1), 0), None, False)
            del buf4
            del buf5
            buf8 = buf7[0]
            assert_size_stride(buf8, (8, 8, 512, 64), (262144, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf8, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf7
            buf12 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, view_3, linear_3], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf8, (4096, 512), (512, 1), 0), reinterpret_tensor(arg6_1, (512, 512), (1, 512), 0), out=buf12)
            del arg6_1
            del buf8
            buf16 = reinterpret_tensor(buf12, (8, 512, 512), (262144, 512, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [linear_3, add, layer_norm_1], Original ATen: [aten._unsafe_view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_native_layer_norm_1.run(buf16, arg0_1, arg7_1, arg8_1, 4096, 512, stream=stream0)
            del arg0_1
            del arg7_1
            del arg8_1
        return (buf16, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

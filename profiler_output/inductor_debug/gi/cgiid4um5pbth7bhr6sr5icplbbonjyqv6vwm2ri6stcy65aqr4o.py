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


# kernel path: /home/ubuntu/Profiler/profiler_output/inductor_debug/hh/chhxjvgebasknzoeyhgtfs2voun7tkq3jss2hzxclviyqwvkm3ix.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %arg2_1 : Tensor "f32[16, 512][512, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %getitem_1 : Tensor "f32[16, 1][1, 16]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[16, 1][1, 16]cuda:0" = PlaceHolder[target=buf1]
#   %arg0_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg2_1, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg2_1, %getitem_1), kwargs = {})
#   %add : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg0_1), kwargs = {})
#   %add_1 : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg1_1), kwargs = {})
#   return %getitem_1,%buf1,%add_1
triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_per_fused_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '61FD96D882CAA14506751336AFCA96D7E09BFF9C7BCDC89B222AC88CF5D0619B', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 102400}}
)
@triton.jit
def triton_per_fused_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tmp0 - tmp10
    tmp18 = tl.full([1, 1], 512.0, tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tl.full([1, 1], 1e-05, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r0_1 + 512*x0), tmp27, xmask)
''', device_str='cuda')


# kernel path: /home/ubuntu/Profiler/profiler_output/inductor_debug/uw/cuwdn5zsgebp43tydz33t2iehuoulvbwwpx3veyb3wlhk3l2side.py
# Topologically Sorted Source Nodes: [q], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   q => relu
# Graph fragment:
#   %mm : Tensor "f32[16, 512][512, 1]cuda:0" = PlaceHolder[target=mm]
#   %relu : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%mm,), kwargs = {})
#   return %relu
triton_poi_fused_relu_1 = async_compile.triton('triton_poi_fused_relu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '61FD96D882CAA14506751336AFCA96D7E09BFF9C7BCDC89B222AC88CF5D0619B', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 98304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /home/ubuntu/Profiler/profiler_output/inductor_debug/sc/csccrokgv5l5tmavg3zmrdgqjofw2hgpcbpcc7a5xuuiluztsadz.py
# Topologically Sorted Source Nodes: [scores], Original ATen: [aten.mul, aten.amax, aten.sub, aten.div, aten._softmax]
# Source node to ATen node mapping:
#   scores => div_1, exp, sum_1
# Graph fragment:
#   %mm_2 : Tensor "f32[16, 16][16, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %amax_default : Tensor "f32[16, 1][1, 16]cuda:0" = PlaceHolder[target=amax_default]
#   %sum_1 : Tensor "f32[16, 1][1, 16]cuda:0" = PlaceHolder[target=sum_1]
#   %mul_tensor : Tensor "f32[16, 16][16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_2, 1), kwargs = {})
#   %amax_default : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : Tensor "f32[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 22.627416997969522), kwargs = {})
#   %exp : Tensor "f32[16, 16][16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_1 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %amax_default,%sum_1,%div_1
triton_per_fused__softmax_amax_div_mul_sub_2 = async_compile.triton('triton_per_fused__softmax_amax_div_mul_sub_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_amax_div_mul_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '61FD96D882CAA14506751336AFCA96D7E09BFF9C7BCDC89B222AC88CF5D0619B', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 3072}}
)
@triton.jit
def triton_per_fused__softmax_amax_div_mul_sub_2(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.full([1, 1], 1.0, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tmp2 - tmp6
    tmp8 = tl.full([1, 1], 0.044194173824159216, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = libdevice.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp15 = (tmp10 / tmp14)
    tl.store(in_out_ptr0 + (r0_1 + 16*x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: /home/ubuntu/Profiler/profiler_output/inductor_debug/hm/chmz72mnqo7e5btv4oddazw43ckbcceqv6dkipsqags2u4apvqgd.py
# Topologically Sorted Source Nodes: [linear_3, relu_1], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear_3 => add_tensor_1
#   relu_1 => relu_1
# Graph fragment:
#   %arg9_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %mm_default_1 : Tensor "f32[16, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %add_tensor_1 : Tensor "f32[16, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg9_1, %mm_default_1), kwargs = {})
#   %relu_1 : Tensor "f32[16, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   return %relu_1
triton_poi_fused_addmm_relu_3 = async_compile.triton('triton_poi_fused_addmm_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '61FD96D882CAA14506751336AFCA96D7E09BFF9C7BCDC89B222AC88CF5D0619B', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 401408}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 2048)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /home/ubuntu/Profiler/profiler_output/inductor_debug/q2/cq2lgkmy4kyvkslxpybjbkq2mcocyv2eakplixptfohogjsb2xbl.py
# Topologically Sorted Source Nodes: [linear_4, gelu, x_1], Original ATen: [aten.addmm, aten.gelu, aten.add]
# Source node to ATen node mapping:
#   gelu => add_5, erf, mul_4, mul_5, mul_6
#   linear_4 => add_tensor
#   x_1 => add_6
# Graph fragment:
#   %addmm_default : Tensor "f32[16, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_default]
#   %arg11_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %mm_default : Tensor "f32[16, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg11_1, %mm_default), kwargs = {})
#   %mul_4 : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, 0.5), kwargs = {})
#   %mul_5 : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_5 : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_6 : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_5), kwargs = {})
#   %add_6 : Tensor "f32[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%addmm_default, %mul_6), kwargs = {})
#   return %add_6
triton_poi_fused_add_addmm_gelu_4 = async_compile.triton('triton_poi_fused_add_addmm_gelu_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '61FD96D882CAA14506751336AFCA96D7E09BFF9C7BCDC89B222AC88CF5D0619B', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 133120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_gelu_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tl.full([1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp7 = tmp3 * tmp6
    tmp8 = libdevice.erf(tmp7)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp5 * tmp10
    tmp12 = tmp0 + tmp11
    tl.store(in_out_ptr0 + (x2), tmp12, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1 = args
        args.clear()
        assert_size_stride(arg0_1, (512, ), (1, ))
        assert_size_stride(arg1_1, (512, ), (1, ))
        assert_size_stride(arg2_1, (16, 512), (512, 1))
        assert_size_stride(arg3_1, (512, 512), (512, 1))
        assert_size_stride(arg4_1, (512, 512), (512, 1))
        assert_size_stride(arg5_1, (512, 512), (512, 1))
        assert_size_stride(arg6_1, (512, ), (1, ))
        assert_size_stride(arg7_1, (512, ), (1, ))
        assert_size_stride(arg8_1, (2048, 512), (512, 1))
        assert_size_stride(arg9_1, (2048, ), (1, ))
        assert_size_stride(arg10_1, (512, 2048), (2048, 1))
        assert_size_stride(arg11_1, (512, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((16, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_0.run(arg2_1, arg0_1, arg1_1, buf3, 16, 512, stream=stream0)
            del arg0_1
            del arg1_1
            buf4 = empty_strided_cuda((16, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf3, reinterpret_tensor(arg3_1, (512, 512), (1, 512), 0), out=buf4)
            del arg3_1
            buf5 = empty_strided_cuda((16, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [v], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf3, reinterpret_tensor(arg4_1, (512, 512), (1, 512), 0), out=buf5)
            del arg4_1
            del buf3
            buf6 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [q], Original ATen: [aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_relu_1.run(buf6, 8192, stream=stream0)
            buf7 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [q, transpose, matmul], Original ATen: [aten.relu, aten.transpose, aten.mm]
            extern_kernels.mm(buf6, reinterpret_tensor(buf5, (512, 16), (1, 512), 0), out=buf7)
            buf10 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [scores], Original ATen: [aten.mul, aten.amax, aten.sub, aten.div, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_amax_div_mul_sub_2.run(buf10, 16, 16, stream=stream0)
            buf11 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [scores, matmul_1], Original ATen: [aten.mul, aten.sub, aten.div, aten._softmax, aten.mm]
            extern_kernels.mm(buf10, buf5, out=buf11)
            del buf10
            buf12 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.t, aten.addmm]
            extern_kernels.addmm(arg2_1, buf11, reinterpret_tensor(arg5_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf12)
            del arg2_1
            del arg5_1
            buf16 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_0.run(buf12, arg6_1, arg7_1, buf16, 16, 512, stream=stream0)
            del arg6_1
            del arg7_1
            buf17 = empty_strided_cuda((16, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_1, linear_3], Original ATen: [aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.mm(buf16, reinterpret_tensor(arg8_1, (512, 2048), (1, 512), 0), out=buf17)
            del arg8_1
            del buf16
            buf18 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [linear_3, relu_1], Original ATen: [aten.addmm, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_3.run(buf18, arg9_1, 32768, stream=stream0)
            del arg9_1
            buf19 = empty_strided_cuda((16, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3, relu_1, linear_4], Original ATen: [aten.addmm, aten.relu, aten.t]
            extern_kernels.mm(buf18, reinterpret_tensor(arg10_1, (2048, 512), (1, 2048), 0), out=buf19)
            del arg10_1
            del buf18
            buf20 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [linear_4, gelu, x_1], Original ATen: [aten.addmm, aten.gelu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_gelu_4.run(buf20, arg11_1, buf19, 8192, stream=stream0)
            del arg11_1
            del buf19
        return (buf20, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

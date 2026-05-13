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


# kernel path: /root/Profiler/examples/transformer_stack/test_run/inductor_debug/cg/ccgvqpdd6abvx5cc462c54n7qqp5jnxeihogz7t47mfbmectffb5.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %arg0_1 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %getitem_1 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf1]
#   %arg1_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg0_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %getitem_1), kwargs = {})
#   %add : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg1_1), kwargs = {})
#   %add_1 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg2_1), kwargs = {})
#   return %getitem_1,%buf1,%add_1
triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_per_fused_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 3149824}}
)
@triton.jit
def triton_per_fused_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
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


# kernel path: /root/Profiler/examples/transformer_stack/test_run/inductor_debug/px/cpxtfduqyheyn7xyekptcwcqywd32mq3gdbdhmhys24x2c24nayl.py
# Topologically Sorted Source Nodes: [linear, view, transpose, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   linear => view_1
#   matmul => clone
#   transpose => permute_1
#   view => view_2
# Graph fragment:
#   %mm : Tensor "f32[512, 512][512, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_1 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [4, 128, 512]), kwargs = {})
#   %view_2 : Tensor "f32[4, 128, 8, 64][65536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [4, 128, 8, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[4, 8, 128, 64][65536, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f32[4, 8, 128, 64][65536, 8192, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused__unsafe_view_clone_transpose_view_1 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3145728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 128)
    x2 = ((xindex // 8192) % 8)
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 512*x1 + 65536*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /root/Profiler/examples/transformer_stack/test_run/inductor_debug/7k/c7kzsa3erc3nhu55ghbmqhpupr22gljp7g5f26cpk4ulzeiznyrw.py
# Topologically Sorted Source Nodes: [linear_1, view_1, transpose_1, transpose_3, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   linear_1 => view_4
#   matmul => clone_1
#   transpose_1 => permute_3
#   transpose_3 => permute_6
#   view_1 => view_5
# Graph fragment:
#   %mm_1 : Tensor "f32[512, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %view_4 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [4, 128, 512]), kwargs = {})
#   %view_5 : Tensor "f32[4, 128, 8, 64][65536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [4, 128, 8, 64]), kwargs = {})
#   %permute_3 : Tensor "f32[4, 8, 128, 64][65536, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %permute_6 : Tensor "f32[4, 8, 64, 128][65536, 64, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_3, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[4, 8, 64, 128][65536, 8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused__unsafe_view_clone_transpose_view_2 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1048576, 'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 65536*y1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 128*y3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/Profiler/examples/transformer_stack/test_run/inductor_debug/fs/cfshh6phwugwyf4qrinxpgd5rm4xqklkymwcyun6cdz3fsi4se74.py
# Topologically Sorted Source Nodes: [matmul, softmax], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   matmul => view_11
#   softmax => div, exp, sum_1
# Graph fragment:
#   %bmm : Tensor "f32[32, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm]
#   %amax_default : Tensor "f32[4, 8, 128, 1][1024, 128, 1, 4096]cuda:0" = PlaceHolder[target=amax_default]
#   %sum_1 : Tensor "f32[4, 8, 128, 1][1024, 128, 1, 4096]cuda:0" = PlaceHolder[target=sum_1]
#   %view_11 : Tensor "f32[4, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [4, 8, 128, 128]), kwargs = {})
#   %mul_tensor : Tensor "f32[4, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, 1), kwargs = {})
#   %amax_default : Tensor "f32[4, 8, 128, 1][1024, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[4, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[4, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 0.125), kwargs = {})
#   %exp : Tensor "f32[4, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_1,), kwargs = {})
#   %sum_1 : Tensor "f32[4, 8, 128, 1][1024, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : Tensor "f32[4, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %amax_default,%sum_1,%expand_2
triton_per_fused__softmax_amax_mul_sub_view_3 = async_compile.triton('triton_per_fused__softmax_amax_mul_sub_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_amax_mul_sub_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 6291456}}
)
@triton.jit
def triton_per_fused__softmax_amax_mul_sub_view_3(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), None)
    tmp1 = tl.full([1, 1], 1.0, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tmp2 - tmp5
    tmp7 = tl.full([1, 1], 0.125, tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = libdevice.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = (tmp9 / tmp12)
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp13, None)
''', device_str='cuda')


# kernel path: /root/Profiler/examples/transformer_stack/test_run/inductor_debug/zt/cztqw263z3agtarw32hqu2xkxnca6rxp4vqwlwq6fmc65xsadss2.py
# Topologically Sorted Source Nodes: [matmul_1, transpose_4, contiguous], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_3
#   matmul_1 => view_14
#   transpose_4 => permute_7
# Graph fragment:
#   %bmm_1 : Tensor "f32[32, 128, 64][8192, 64, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %view_14 : Tensor "f32[4, 8, 128, 64][65536, 8192, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [4, 8, 128, 64]), kwargs = {})
#   %permute_7 : Tensor "f32[4, 128, 8, 64][65536, 64, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_14, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[4, 128, 8, 64][65536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_clone_transpose_view_4 = async_compile.triton('triton_poi_fused_clone_transpose_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3145728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 8)
    x2 = ((xindex // 512) % 128)
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 8192*x1 + 65536*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /root/Profiler/examples/transformer_stack/test_run/inductor_debug/c2/cc2egbnu6rjbzeq6r4dosaypmgyz5tlkspnq4uusvs5wuzybfqxq.py
# Topologically Sorted Source Nodes: [linear_3, add, layer_norm_1], Original ATen: [aten._unsafe_view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_2
#   layer_norm_1 => add_3, add_4, mul_3, mul_4, rsqrt_1, sub_2, var_mean_1
#   linear_3 => view_17
# Graph fragment:
#   %arg0_1 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %mm_3 : Tensor "f32[512, 512][512, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %getitem_3 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=getitem_3]
#   %buf18 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf18]
#   %arg7_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %view_17 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [4, 128, 512]), kwargs = {})
#   %add_2 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %view_17), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_3), kwargs = {})
#   %add_3 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_3 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_4 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg7_1), kwargs = {})
#   %add_4 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg8_1), kwargs = {})
#   return %getitem_3,%buf18,%add_4
triton_per_fused__unsafe_view_add_native_layer_norm_5 = async_compile.triton('triton_per_fused__unsafe_view_add_native_layer_norm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 4198400}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = tl.full([1, 1], 512, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = (tmp9 / tmp11)
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = tmp2 - tmp12
    tmp20 = tl.full([1, 1], 512.0, tl.float32)
    tmp21 = (tmp18 / tmp20)
    tmp22 = tl.full([1, 1], 1e-05, tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r0_1 + 512*x0), tmp29, xmask)
''', device_str='cuda')


# kernel path: /root/Profiler/examples/transformer_stack/test_run/inductor_debug/pb/cpb2v2xjmyfctxjv4q6qd566ai42duc5fuqhl6zwwrlpb2lpsmsi.py
# Topologically Sorted Source Nodes: [linear_4, gelu], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_5, erf, mul_5, mul_6, mul_7
#   linear_4 => add_tensor_1, view_19
# Graph fragment:
#   %arg10_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %mm_default_1 : Tensor "f32[512, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %add_tensor_1 : Tensor "f32[512, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg10_1, %mm_default_1), kwargs = {})
#   %view_19 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [4, 128, 2048]), kwargs = {})
#   %mul_5 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.5), kwargs = {})
#   %mul_6 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_5 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_5), kwargs = {})
#   return %mul_7
triton_poi_fused_addmm_gelu_view_6 = async_compile.triton('triton_poi_fused_addmm_gelu_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12591104}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 2048)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /root/Profiler/examples/transformer_stack/test_run/inductor_debug/5n/c5n45p7d7ontdtwyxundqe7t7qwnl7trwsu3cs5voudd2lzgedus.py
# Topologically Sorted Source Nodes: [linear_3, add, linear_5, add_1], Original ATen: [aten._unsafe_view, aten.add, aten.addmm, aten.view]
# Source node to ATen node mapping:
#   add => add_2
#   add_1 => add_6
#   linear_3 => view_17
#   linear_5 => add_tensor, view_21
# Graph fragment:
#   %arg0_1 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %mm_3 : Tensor "f32[512, 512][512, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %arg12_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %mm_default : Tensor "f32[512, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %view_17 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [4, 128, 512]), kwargs = {})
#   %add_2 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %view_17), kwargs = {})
#   %add_tensor : Tensor "f32[512, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg12_1, %mm_default), kwargs = {})
#   %view_21 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [4, 128, 512]), kwargs = {})
#   %add_6 : Tensor "f32[4, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_21), kwargs = {})
#   return %add_6
triton_poi_fused__unsafe_view_add_addmm_view_7 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 5244928}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
        args.clear()
        assert_size_stride(arg0_1, (4, 128, 512), (65536, 512, 1))
        assert_size_stride(arg1_1, (512, ), (1, ))
        assert_size_stride(arg2_1, (512, ), (1, ))
        assert_size_stride(arg3_1, (512, 512), (512, 1))
        assert_size_stride(arg4_1, (512, 512), (512, 1))
        assert_size_stride(arg5_1, (512, 512), (512, 1))
        assert_size_stride(arg6_1, (512, 512), (512, 1))
        assert_size_stride(arg7_1, (512, ), (1, ))
        assert_size_stride(arg8_1, (512, ), (1, ))
        assert_size_stride(arg9_1, (2048, 512), (512, 1))
        assert_size_stride(arg10_1, (2048, ), (1, ))
        assert_size_stride(arg11_1, (512, 2048), (2048, 1))
        assert_size_stride(arg12_1, (512, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((4, 128, 512), (65536, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_0.run(arg0_1, arg1_1, arg2_1, buf3, 512, 512, stream=stream0)
            del arg1_1
            del arg2_1
            buf4 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (512, 512), (512, 1), 0), reinterpret_tensor(arg3_1, (512, 512), (1, 512), 0), out=buf4)
            del arg3_1
            buf5 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (512, 512), (512, 1), 0), reinterpret_tensor(arg4_1, (512, 512), (1, 512), 0), out=buf5)
            del arg4_1
            buf6 = empty_strided_cuda((4, 8, 128, 64), (65536, 8192, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, view, transpose, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_1.run(buf4, buf6, 262144, stream=stream0)
            buf7 = reinterpret_tensor(buf4, (4, 8, 64, 128), (65536, 8192, 128, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [linear_1, view_1, transpose_1, transpose_3, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_2.run(buf5, buf7, 2048, 128, stream=stream0)
            del buf5
            buf8 = empty_strided_cuda((32, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, view, transpose, matmul, linear_1, view_1, transpose_1, transpose_3], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf6, (32, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf7, (32, 64, 128), (8192, 128, 1), 0), out=buf8)
            del buf6
            buf12 = reinterpret_tensor(buf8, (4, 8, 128, 128), (131072, 16384, 128, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [matmul, softmax], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_amax_mul_sub_view_3.run(buf12, 4096, 128, stream=stream0)
            buf11 = reinterpret_tensor(buf7, (512, 512), (512, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (512, 512), (512, 1), 0), reinterpret_tensor(arg5_1, (512, 512), (1, 512), 0), out=buf11)
            del arg5_1
            buf13 = reinterpret_tensor(buf3, (4, 8, 128, 64), (65536, 8192, 64, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [linear_2, view_2, transpose_2, matmul_1], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_1.run(buf11, buf13, 262144, stream=stream0)
            buf14 = reinterpret_tensor(buf11, (32, 128, 64), (8192, 64, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [matmul, softmax, matmul_1, linear_2, view_2, transpose_2], Original ATen: [aten.view, aten.mul, aten.sub, aten._softmax, aten._unsafe_view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf12, (32, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf13, (32, 128, 64), (8192, 64, 1), 0), out=buf14)
            del buf12
            buf15 = reinterpret_tensor(buf13, (4, 128, 8, 64), (65536, 512, 64, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [matmul_1, transpose_4, contiguous], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_4.run(buf14, buf15, 262144, stream=stream0)
            buf16 = reinterpret_tensor(buf14, (512, 512), (512, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [matmul_1, transpose_4, contiguous, view_3, linear_3], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf15, (512, 512), (512, 1), 0), reinterpret_tensor(arg6_1, (512, 512), (1, 512), 0), out=buf16)
            del arg6_1
            buf20 = reinterpret_tensor(buf15, (4, 128, 512), (65536, 512, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [linear_3, add, layer_norm_1], Original ATen: [aten._unsafe_view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_native_layer_norm_5.run(arg0_1, buf16, arg7_1, arg8_1, buf20, 512, 512, stream=stream0)
            del arg7_1
            del arg8_1
            buf21 = empty_strided_cuda((512, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3, add, layer_norm_1, linear_4], Original ATen: [aten._unsafe_view, aten.add, aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf20, (512, 512), (512, 1), 0), reinterpret_tensor(arg9_1, (512, 2048), (1, 512), 0), out=buf21)
            del arg9_1
            del buf20
            buf22 = reinterpret_tensor(buf21, (4, 128, 2048), (262144, 2048, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [linear_4, gelu], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_6.run(buf22, arg10_1, 1048576, stream=stream0)
            del arg10_1
            buf23 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_4, gelu, linear_5], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf22, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg11_1, (2048, 512), (1, 2048), 0), out=buf23)
            del arg11_1
            del buf22
            buf24 = reinterpret_tensor(buf16, (4, 128, 512), (65536, 512, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [linear_3, add, linear_5, add_1], Original ATen: [aten._unsafe_view, aten.add, aten.addmm, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_view_7.run(buf24, arg0_1, arg12_1, buf23, 262144, stream=stream0)
            del arg0_1
            del arg12_1
            del buf23
        return (buf24, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

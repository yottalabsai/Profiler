# AOT ID: ['2_inference']
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


# kernel path: /root/Profiler/gpt2_optimized_inductor_debug/yj/cyjbaju5io2nhgdrhy7lwa3nwioreke5tlblqzy6jrtpom3zbbpb.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, add_1, convert_element_type, convert_element_type_1, mul_4, mul_5, rsqrt, sub, var_mean
# Graph fragment:
#   %arg0_1 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %getitem_1 : Tensor "f32[4, s22, 1][s22, 1, 4*s22]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[4, s22, 1][s22, 1, 4*s22]cuda:0" = PlaceHolder[target=buf1]
#   %arg1_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %add : Tensor "f32[4, s22, 1][s22, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[4, s22, 1][s22, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_4 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_5 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg1_1), kwargs = {})
#   %add_1 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg2_1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   return %getitem_1,%buf1,%convert_element_type_1
triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_per_fused_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 2362368}}
)
@triton.jit
def triton_per_fused_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 768, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tmp1 - tmp11
    tmp19 = tl.full([1, 1], 768.0, tl.float32)
    tmp20 = (tmp17 / tmp19)
    tmp21 = tl.full([1, 1], 1e-05, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/Profiler/gpt2_optimized_inductor_debug/eb/ceblw7keuiebtanowq6x4jc6bom6zeg4wvrakfazb76s4j6swsyh.py
# Topologically Sorted Source Nodes: [view_1, split, view_4, transpose_2, view_2, transpose, cat, view_3, transpose_1, cat_1, scaled_dot_product_attention], Original ATen: [aten.view, aten.split, aten.transpose, aten.cat, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   cat => clone
#   cat_1 => clone_1
#   scaled_dot_product_attention => _scaled_dot_product_efficient_attention, constant_pad_nd, expand, full_default, full_default_1, slice_1, where
#   split => split
#   transpose => permute
#   transpose_1 => permute_1
#   transpose_2 => permute_2
#   view_1 => view_1
#   view_2 => view_2
#   view_3 => view_3
#   view_4 => view_4
# Graph fragment:
#   %addmm : Tensor "bf16[4*s22, 2304][2304, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_1 : Tensor "bf16[4, s22, 2304][2304*s22, 2304, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 128, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %view_4 : Tensor "bf16[4, s22, ((3*s22)//32), (768//(((3*s22)//32)))][2304*s22, 2304, (768//(((3*s22)//32))), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [4, 128, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][2304*s22, (768//(((3*s22)//32))), 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %view_2 : Tensor "bf16[4, s22, ((3*s22)//32), (768//(((3*s22)//32)))][2304*s22, 2304, (768//(((3*s22)//32))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_3, [4, 128, -1, 64]), kwargs = {})
#   %permute : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][2304*s22, (768//(((3*s22)//32))), 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][s22*Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), Max(1, (768//(((3*s22)//32)))), Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {})
#   %view_3 : Tensor "bf16[4, s22, ((3*s22)//32), (768//(((3*s22)//32)))][2304*s22, 2304, (768//(((3*s22)//32))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [4, 128, -1, 64]), kwargs = {})
#   %permute_1 : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][2304*s22, (768//(((3*s22)//32))), 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %clone_1 : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][s22*Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), Max(1, (768//(((3*s22)//32)))), Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {})
#   %full_default_1 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[4, 1, s22, s22][s22**2, s22**2, s22, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%arg5_1, %full_default_1, %full_default), kwargs = {})
#   %constant_pad_nd : Tensor "bf16[4, 1, s22, s22 - (Mod(s22, 8)) + 8][s22*Max(1, s22 - (Mod(s22, 8)) + 8), s22*Max(1, s22 - (Mod(s22, 8)) + 8), Max(1, s22 - (Mod(s22, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_36], 0.0), kwargs = {})
#   %slice_1 : Tensor "bf16[4, 1, s22, s22][s22*Max(1, s22 - (Mod(s22, 8)) + 8), s22*Max(1, s22 - (Mod(s22, 8)) + 8), Max(1, s22 - (Mod(s22, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, %sym_size_int_10), kwargs = {})
#   %expand : Tensor "bf16[4, ((3*s22)//32), s22, s22][s22*Max(1, s22 - (Mod(s22, 8)) + 8), 0, Max(1, s22 - (Mod(s22, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_1, [4, %sym_size_int_7, %sym_size_int_2, %sym_size_int_2]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_2, %clone, %clone_1, %expand, False), kwargs = {scale: 0.125})
#   return %buf5
triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_1 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 768)
    x1 = xindex // 768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + 2304*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /root/Profiler/gpt2_optimized_inductor_debug/qf/cqf7mvprbvzno36hzvpxb7iplotm5qrabkzombhatld4osrssmod.py
# Topologically Sorted Source Nodes: [view_1, split, view_4, transpose_2, view_2, transpose, cat, view_3, transpose_1, cat_1, scaled_dot_product_attention], Original ATen: [aten.view, aten.split, aten.transpose, aten.cat, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   cat => clone
#   cat_1 => clone_1
#   scaled_dot_product_attention => _scaled_dot_product_efficient_attention, constant_pad_nd, expand, full_default, full_default_1, slice_1, where
#   split => split
#   transpose => permute
#   transpose_1 => permute_1
#   transpose_2 => permute_2
#   view_1 => view_1
#   view_2 => view_2
#   view_3 => view_3
#   view_4 => view_4
# Graph fragment:
#   %addmm : Tensor "bf16[4*s22, 2304][2304, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_1 : Tensor "bf16[4, s22, 2304][2304*s22, 2304, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 128, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %view_4 : Tensor "bf16[4, s22, ((3*s22)//32), (768//(((3*s22)//32)))][2304*s22, 2304, (768//(((3*s22)//32))), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [4, 128, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][2304*s22, (768//(((3*s22)//32))), 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %view_2 : Tensor "bf16[4, s22, ((3*s22)//32), (768//(((3*s22)//32)))][2304*s22, 2304, (768//(((3*s22)//32))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_3, [4, 128, -1, 64]), kwargs = {})
#   %permute : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][2304*s22, (768//(((3*s22)//32))), 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][s22*Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), Max(1, (768//(((3*s22)//32)))), Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {})
#   %view_3 : Tensor "bf16[4, s22, ((3*s22)//32), (768//(((3*s22)//32)))][2304*s22, 2304, (768//(((3*s22)//32))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [4, 128, -1, 64]), kwargs = {})
#   %permute_1 : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][2304*s22, (768//(((3*s22)//32))), 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %clone_1 : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][s22*Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), Max(1, (768//(((3*s22)//32)))), Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {})
#   %full_default_1 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[4, 1, s22, s22][s22**2, s22**2, s22, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%arg5_1, %full_default_1, %full_default), kwargs = {})
#   %constant_pad_nd : Tensor "bf16[4, 1, s22, s22 - (Mod(s22, 8)) + 8][s22*Max(1, s22 - (Mod(s22, 8)) + 8), s22*Max(1, s22 - (Mod(s22, 8)) + 8), Max(1, s22 - (Mod(s22, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_36], 0.0), kwargs = {})
#   %slice_1 : Tensor "bf16[4, 1, s22, s22][s22*Max(1, s22 - (Mod(s22, 8)) + 8), s22*Max(1, s22 - (Mod(s22, 8)) + 8), Max(1, s22 - (Mod(s22, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, %sym_size_int_10), kwargs = {})
#   %expand : Tensor "bf16[4, ((3*s22)//32), s22, s22][s22*Max(1, s22 - (Mod(s22, 8)) + 8), 0, Max(1, s22 - (Mod(s22, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_1, [4, %sym_size_int_7, %sym_size_int_2, %sym_size_int_2]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_2, %clone, %clone_1, %expand, False), kwargs = {scale: 0.125})
#   return %buf6
triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_2 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 768)
    x1 = xindex // 768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 2304*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /root/Profiler/gpt2_optimized_inductor_debug/u6/cu6m6x2cfvfbnvjtsovankrf6vwbnuhol7hcim7uhr2idnvhgmtp.py
# Topologically Sorted Source Nodes: [view_1, split, view_4, transpose_2, view_2, transpose, cat, view_3, transpose_1, cat_1, scaled_dot_product_attention], Original ATen: [aten.view, aten.split, aten.transpose, aten.cat, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   cat => clone
#   cat_1 => clone_1
#   scaled_dot_product_attention => _scaled_dot_product_efficient_attention, constant_pad_nd, expand, full_default, full_default_1, slice_1, where
#   split => split
#   transpose => permute
#   transpose_1 => permute_1
#   transpose_2 => permute_2
#   view_1 => view_1
#   view_2 => view_2
#   view_3 => view_3
#   view_4 => view_4
# Graph fragment:
#   %arg5_1 : Tensor "b8[4, 1, s22, s22][0, s22**2, s22, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %view_1 : Tensor "bf16[4, s22, 2304][2304*s22, 2304, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 128, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %view_4 : Tensor "bf16[4, s22, ((3*s22)//32), (768//(((3*s22)//32)))][2304*s22, 2304, (768//(((3*s22)//32))), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [4, 128, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][2304*s22, (768//(((3*s22)//32))), 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %view_2 : Tensor "bf16[4, s22, ((3*s22)//32), (768//(((3*s22)//32)))][2304*s22, 2304, (768//(((3*s22)//32))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_3, [4, 128, -1, 64]), kwargs = {})
#   %permute : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][2304*s22, (768//(((3*s22)//32))), 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][s22*Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), Max(1, (768//(((3*s22)//32)))), Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {})
#   %view_3 : Tensor "bf16[4, s22, ((3*s22)//32), (768//(((3*s22)//32)))][2304*s22, 2304, (768//(((3*s22)//32))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [4, 128, -1, 64]), kwargs = {})
#   %permute_1 : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][2304*s22, (768//(((3*s22)//32))), 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %clone_1 : Tensor "bf16[4, ((3*s22)//32), s22, (768//(((3*s22)//32)))][s22*Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), Max(1, (768//(((3*s22)//32)))), Max(1, (768//(((3*s22)//32))))*Max(1, ((3*s22)//32)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {})
#   %full_default_1 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[4, 1, s22, s22][s22**2, s22**2, s22, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%arg5_1, %full_default_1, %full_default), kwargs = {})
#   %constant_pad_nd : Tensor "bf16[4, 1, s22, s22 - (Mod(s22, 8)) + 8][s22*Max(1, s22 - (Mod(s22, 8)) + 8), s22*Max(1, s22 - (Mod(s22, 8)) + 8), Max(1, s22 - (Mod(s22, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_36], 0.0), kwargs = {})
#   %slice_1 : Tensor "bf16[4, 1, s22, s22][s22*Max(1, s22 - (Mod(s22, 8)) + 8), s22*Max(1, s22 - (Mod(s22, 8)) + 8), Max(1, s22 - (Mod(s22, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, %sym_size_int_10), kwargs = {})
#   %expand : Tensor "bf16[4, ((3*s22)//32), s22, s22][s22*Max(1, s22 - (Mod(s22, 8)) + 8), 0, Max(1, s22 - (Mod(s22, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_1, [4, %sym_size_int_7, %sym_size_int_2, %sym_size_int_2]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_2, %clone, %clone_1, %expand, False), kwargs = {scale: 0.125})
#   return %buf7
triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_3 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 278528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_3(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x3 = (xindex % ks1)
    x4 = xindex // ks0
    tmp0 = x0
    tmp1 = ks0
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x3), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp4 = tl.full([1], 0.0, tl.float32)
    tmp5 = tl.full([1], float("-inf"), tl.float32)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp2, tmp6, tmp7)
    tl.store(out_ptr0 + (x0 + 8*x4*((7 + ks0) // 8)), tmp8, xmask)
''', device_str='cuda')


# kernel path: /root/Profiler/gpt2_optimized_inductor_debug/7s/c7sgzcdwrfeknieyvp5s6uwhmlj5lgddbqiwgdnpnmxlosps6jz4.py
# Topologically Sorted Source Nodes: [view_6, add, layer_norm_1], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_126
#   layer_norm_1 => add_131, add_132, convert_element_type_8, convert_element_type_9, mul_121, mul_122, rsqrt_1, sub_59, var_mean_1
#   view_6 => view_7
# Graph fragment:
#   %addmm_1 : Tensor "bf16[512, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %arg0_1 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %getitem_10 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=getitem_10]
#   %buf15 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf15]
#   %arg8_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %view_7 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [4, 128, 768]), kwargs = {})
#   %add_126 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_7, %arg0_1), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_126, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_59 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_8, %getitem_10), kwargs = {})
#   %add_131 : Tensor "f32[4, s22, 1][s22, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, s22, 1][s22, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_131,), kwargs = {})
#   %mul_121 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %rsqrt_1), kwargs = {})
#   %mul_122 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, %arg8_1), kwargs = {})
#   %add_132 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_122, %arg9_1), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_132, torch.bfloat16), kwargs = {})
#   return %getitem_10,%buf15,%convert_element_type_9
triton_per_fused_add_native_layer_norm_view_4 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr2': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 3148800}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x0 + 768*ks0*x1), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([1, 1], 768, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp20 = tmp3 - tmp13
    tmp21 = tl.full([1, 1], 768.0, tl.float32)
    tmp22 = (tmp19 / tmp21)
    tmp23 = tl.full([1, 1], 1e-05, tl.float32)
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 * tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/Profiler/gpt2_optimized_inductor_debug/zb/czbvxldyt3l7teqrcywu4kolziswyogj3h26ioljetasgtmc53ee.py
# Topologically Sorted Source Nodes: [view_8, mul, pow_1, mul_1, add_1, mul_2, tanh, add_2, mul_3], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_1 => add_167
#   add_2 => add_180
#   mul => mul_137
#   mul_1 => mul_144
#   mul_2 => mul_151
#   mul_3 => mul_161
#   pow_1 => pow_1
#   tanh => tanh
#   view_8 => view_9
# Graph fragment:
#   %addmm_2 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %view_9 : Tensor "bf16[4, s22, 3072][3072*s22, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [4, 128, 3072]), kwargs = {})
#   %mul_137 : Tensor "bf16[4, s22, 3072][3072*s22, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, 0.5), kwargs = {})
#   %pow_1 : Tensor "bf16[4, s22, 3072][3072*s22, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_9, 3.0), kwargs = {})
#   %mul_144 : Tensor "bf16[4, s22, 3072][3072*s22, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_167 : Tensor "bf16[4, s22, 3072][3072*s22, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %mul_144), kwargs = {})
#   %mul_151 : Tensor "bf16[4, s22, 3072][3072*s22, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_167, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "bf16[4, s22, 3072][3072*s22, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_151,), kwargs = {})
#   %add_180 : Tensor "bf16[4, s22, 3072][3072*s22, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_161 : Tensor "bf16[4, s22, 3072][3072*s22, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_137, %add_180), kwargs = {})
#   return %mul_161
triton_poi_fused_add_mul_pow_tanh_view_5 = async_compile.triton('triton_poi_fused_add_mul_pow_tanh_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_tanh_view_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp3 * tmp0
    tmp5 = tl.full([1], 0.044715, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp2 * tmp12
    tl.store(in_out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /root/Profiler/gpt2_optimized_inductor_debug/36/c36eckxyvs5uv7m4gdapp3osbim6u2z7ctdwtymchk2gomqew7ok.py
# Topologically Sorted Source Nodes: [view_6, add, view_10, add_3, layer_norm_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_126
#   add_3 => add_203
#   layer_norm_2 => add_208, add_209, convert_element_type_16, convert_element_type_17, mul_182, mul_183, rsqrt_2, sub_79, var_mean_2
#   view_10 => view_11
#   view_6 => view_7
# Graph fragment:
#   %addmm_1 : Tensor "bf16[512, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %arg0_1 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %addmm_3 : Tensor "bf16[512, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %getitem_12 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=getitem_12]
#   %buf22 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf22]
#   %arg14_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %view_7 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [4, 128, 768]), kwargs = {})
#   %add_126 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_7, %arg0_1), kwargs = {})
#   %view_11 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [4, 128, 768]), kwargs = {})
#   %add_203 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_126, %view_11), kwargs = {})
#   %convert_element_type_16 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_203, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_16, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_79 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_16, %getitem_12), kwargs = {})
#   %add_208 : Tensor "f32[4, s22, 1][s22, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[4, s22, 1][s22, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_208,), kwargs = {})
#   %mul_182 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %rsqrt_2), kwargs = {})
#   %mul_183 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_182, %arg14_1), kwargs = {})
#   %add_209 : Tensor "f32[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_183, %arg15_1), kwargs = {})
#   %convert_element_type_17 : Tensor "bf16[4, s22, 768][768*s22, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_209, torch.bfloat16), kwargs = {})
#   return %getitem_12,%buf22,%convert_element_type_17
triton_per_fused_add_native_layer_norm_view_6 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr2': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 3935232}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x0 + 768*ks0*x1), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = tl.full([1, 1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = tmp5 - tmp15
    tmp23 = tl.full([1, 1], 768.0, tl.float32)
    tmp24 = (tmp21 / tmp23)
    tmp25 = tl.full([1, 1], 1e-05, tl.float32)
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr2 + (r0_2 + 768*x0 + 768*ks0*x1), tmp35, r0_mask & xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1 = args
        args.clear()
        arg0_1_size = arg0_1.size()
        s22 = arg0_1_size[1]
        assert_size_stride(arg0_1, (4, s22, 768), (768*s22, 768, 1))
        assert_size_stride(arg1_1, (768, ), (1, ))
        assert_size_stride(arg2_1, (768, ), (1, ))
        assert_size_stride(arg3_1, (2304, ), (1, ))
        assert_size_stride(arg4_1, (768, 2304), (2304, 1))
        assert_size_stride(arg5_1, (4, 1, s22, s22), (0, s22*s22, s22, 1))
        assert_size_stride(arg6_1, (768, ), (1, ))
        assert_size_stride(arg7_1, (768, 768), (768, 1))
        assert_size_stride(arg8_1, (768, ), (1, ))
        assert_size_stride(arg9_1, (768, ), (1, ))
        assert_size_stride(arg10_1, (3072, ), (1, ))
        assert_size_stride(arg11_1, (768, 3072), (3072, 1))
        assert_size_stride(arg12_1, (768, ), (1, ))
        assert_size_stride(arg13_1, (3072, 768), (768, 1))
        assert_size_stride(arg14_1, (768, ), (1, ))
        assert_size_stride(arg15_1, (768, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((4, s22, 768), (768*s22, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
            triton_per_fused_native_layer_norm_0_xnumel = 4*s22
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_0.run(arg0_1, arg1_1, arg2_1, buf3, triton_per_fused_native_layer_norm_0_xnumel, 768, stream=stream0)
            del arg1_1
            del arg2_1
            buf4 = empty_strided_cuda((4*s22, 2304), (2304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [layer_norm, view, addmm], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.addmm(arg3_1, reinterpret_tensor(buf3, (4*s22, 768), (768, 1), 0), arg4_1, alpha=1, beta=1, out=buf4)
            del arg3_1
            del arg4_1
            del buf3
            buf5 = empty_strided_cuda((4, 12, 128, 64), (98304, 64, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1, split, view_4, transpose_2, view_2, transpose, cat, view_3, transpose_1, cat_1, scaled_dot_product_attention], Original ATen: [aten.view, aten.split, aten.transpose, aten.cat, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_1.run(buf4, buf5, 393216, stream=stream0)
            buf6 = empty_strided_cuda((4, 12, 128, 64), (98304, 64, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1, split, view_4, transpose_2, view_2, transpose, cat, view_3, transpose_1, cat_1, scaled_dot_product_attention], Original ATen: [aten.view, aten.split, aten.transpose, aten.cat, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_2.run(buf4, buf6, 393216, stream=stream0)
            ps0 = s22*s22
            buf7 = empty_strided_cuda((4, 1, s22, s22), (8*s22*((7 + s22) // 8), 0, 8*((7 + s22) // 8), 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1, split, view_4, transpose_2, view_2, transpose, cat, view_3, transpose_1, cat_1, scaled_dot_product_attention], Original ATen: [aten.view, aten.split, aten.transpose, aten.cat, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_3_xnumel = 4*s22*s22
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_3.run(arg5_1, buf7, s22, ps0, triton_poi_fused__scaled_dot_product_efficient_attention_cat_constant_pad_nd_expand_scalar_tensor_slice_split_transpose_view_where_3_xnumel, stream=stream0)
            del arg5_1
            # Topologically Sorted Source Nodes: [view_1, split, view_4, transpose_2, view_2, transpose, cat, view_3, transpose_1, cat_1, scaled_dot_product_attention], Original ATen: [aten.view, aten.split, aten.transpose, aten.cat, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf4, (4, 12, 128, 64), (294912, 64, 2304, 1), 0), buf5, buf6, reinterpret_tensor(buf7, (4, (3*s22) // 32, s22, s22), (8*s22*((7 + s22) // 8), 0, 8*((7 + s22) // 8), 1), 0), False, scale=0.125)
            del buf4
            del buf5
            del buf7
            buf9 = buf8[0]
            assert_size_stride(buf9, (4, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf9, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf8
            buf13 = reinterpret_tensor(buf6, (512, 768), (768, 1), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, reshape, view_5, addmm_1], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.addmm(arg6_1, reinterpret_tensor(buf9, (512, 768), (768, 1), 0), arg7_1, alpha=1, beta=1, out=buf13)
            del arg6_1
            del arg7_1
            buf17 = reinterpret_tensor(buf9, (4, 128, 768), (98304, 768, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [view_6, add, layer_norm_1], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_view_4.run(buf13, arg0_1, arg8_1, arg9_1, buf17, s22, 512, 768, stream=stream0)
            del arg8_1
            del arg9_1
            buf18 = empty_strided_cuda((512, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_6, add, layer_norm_1, view_7, addmm_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.addmm]
            extern_kernels.addmm(arg10_1, reinterpret_tensor(buf17, (512, 768), (768, 1), 0), arg11_1, alpha=1, beta=1, out=buf18)
            del arg10_1
            del arg11_1
            del buf17
            buf19 = reinterpret_tensor(buf18, (4, 128, 3072), (393216, 3072, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [view_8, mul, pow_1, mul_1, add_1, mul_2, tanh, add_2, mul_3], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_pow_tanh_view_5.run(buf19, 1572864, stream=stream0)
            buf20 = empty_strided_cuda((512, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_8, mul, pow_1, mul_1, add_1, mul_2, tanh, add_2, mul_3, view_9, addmm_3], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.addmm]
            extern_kernels.addmm(arg12_1, reinterpret_tensor(buf19, (512, 3072), (3072, 1), 0), arg13_1, alpha=1, beta=1, out=buf20)
            del arg12_1
            del arg13_1
            del buf19
            buf24 = empty_strided_cuda((4, 128, 768), (768*s22, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_6, add, view_10, add_3, layer_norm_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_view_6.run(buf13, arg0_1, buf20, arg14_1, arg15_1, buf24, s22, 512, 768, stream=stream0)
            del arg0_1
            del arg14_1
            del arg15_1
            del buf13
            del buf20
        return (buf24, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((4, 1, 128, 128), (0, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

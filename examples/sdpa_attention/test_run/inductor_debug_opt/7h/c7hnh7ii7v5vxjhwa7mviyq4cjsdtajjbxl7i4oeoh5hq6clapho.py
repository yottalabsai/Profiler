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


# kernel path: /root/Profiler/examples/sdpa_attention/test_run/inductor_debug_opt/gh/cghiaxpkyapyectmmmosbn24ylcnd3qcdx3wi2olybiupbs53gla.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x => add, add_1, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %arg2_1 : Tensor "bf16[8, 512, 512][262144, 512, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %getitem_1 : Tensor "f32[8, 512, 1][512, 1, 4096]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[8, 512, 1][512, 1, 4096]cuda:0" = PlaceHolder[target=buf1]
#   %arg0_1 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %add : Tensor "f32[8, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[8, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg0_1), kwargs = {})
#   %add_1 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg1_1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[8, 512, 512][262144, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   return %getitem_1,%buf1,%convert_element_type_1
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 12584960}}
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp22 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = tl.full([1, 1], 512, tl.int32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = (tmp6 / tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp15 = tmp1 - tmp9
    tmp16 = tl.full([1, 1], 512.0, tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tl.full([1, 1], 1e-05, tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp15 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 512*x0), tmp28, None)
''', device_str='cuda')


# kernel path: /root/Profiler/examples/sdpa_attention/test_run/inductor_debug_opt/jz/cjzvqjlfgf3ze7btae2pkhpqpi7egykwszvo66fw7w3cglygpn5r.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.mm]
# Source node to ATen node mapping:
#   linear => mm, permute, view
# Graph fragment:
#   %convert_element_type_1 : Tensor "bf16[8, 512, 512][262144, 512, 1]cuda:0" = PlaceHolder[target=convert_element_type_1]
#   %arg3_1 : Tensor "bf16[512, 512][512, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %view : Tensor "bf16[4096, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_1, [4096, 512]), kwargs = {})
#   %permute : Tensor "bf16[512, 512][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg3_1, [1, 0]), kwargs = {})
#   %mm : Tensor "bf16[4096, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view, %permute), kwargs = {})
#   return %mm
triton_tem_fused_mm_t_view_1 = async_compile.triton('triton_tem_fused_mm_t_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_mm_t_view_1', 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_mm_t_view_1(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4096
    N = 512
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /root/Profiler/examples/sdpa_attention/test_run/inductor_debug_opt/qz/cqzf7rknztfi24pkkz7rjpmgn4owhps7sho25tss25pryzmjduy4.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.mm]
# Source node to ATen node mapping:
#   linear_1 => mm_1, permute_2, view_3
# Graph fragment:
#   %convert_element_type_1 : Tensor "bf16[8, 512, 512][262144, 512, 1]cuda:0" = PlaceHolder[target=convert_element_type_1]
#   %arg4_1 : Tensor "bf16[512, 512][512, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %view_3 : Tensor "bf16[4096, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_1, [4096, 512]), kwargs = {})
#   %permute_2 : Tensor "bf16[512, 512][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg4_1, [1, 0]), kwargs = {})
#   %mm_1 : Tensor "bf16[4096, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_3, %permute_2), kwargs = {})
#   return %mm_1
triton_tem_fused_mm_t_view_2 = async_compile.triton('triton_tem_fused_mm_t_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_mm_t_view_2', 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_mm_t_view_2(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4096
    N = 512
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /root/Profiler/examples/sdpa_attention/test_run/inductor_debug_opt/a3/ca3f5zu47tnhngq34v7kdbkqugdymbde47befinwbhgutafu2mfd.py
# Topologically Sorted Source Nodes: [out, add, out_1], Original ATen: [aten._unsafe_view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_2
#   out => view_11
#   out_1 => add_3, add_4, convert_element_type_10, convert_element_type_11, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %mm_3 : Tensor "bf16[4096, 512][512, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %arg2_1 : Tensor "bf16[8, 512, 512][262144, 512, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %getitem_12 : Tensor "f32[8, 512, 1][512, 1, 4096]cuda:0" = PlaceHolder[target=getitem_12]
#   %buf15 : Tensor "f32[8, 512, 1][512, 1, 4096]cuda:0" = PlaceHolder[target=buf15]
#   %arg7_1 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %view_11 : Tensor "bf16[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [8, 512, 512]), kwargs = {})
#   %add_2 : Tensor "bf16[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %arg2_1), kwargs = {})
#   %convert_element_type_10 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_10, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_10, %getitem_12), kwargs = {})
#   %add_3 : Tensor "f32[8, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[8, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_2 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg7_1), kwargs = {})
#   %add_4 : Tensor "f32[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg8_1), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[8, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   return %getitem_12,%buf15,%convert_element_type_11
triton_per_fused__unsafe_view_add_native_layer_norm_3 = async_compile.triton('triton_per_fused__unsafe_view_add_native_layer_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '42F736BFF93D475DBD33BE6FEA8AEA0059AE5DCB187DBF5D164F739DD163E962', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 16779264}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_native_layer_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 512, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp17 = tmp3 - tmp11
    tmp18 = tl.full([1, 1], 512.0, tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tl.full([1, 1], 1e-05, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp30, None)
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
        assert_size_stride(arg0_1, (512, ), (1, ))
        assert_size_stride(arg1_1, (512, ), (1, ))
        assert_size_stride(arg2_1, (8, 512, 512), (262144, 512, 1))
        assert_size_stride(arg3_1, (512, 512), (512, 1))
        assert_size_stride(arg4_1, (512, 512), (512, 1))
        assert_size_stride(arg5_1, (512, 512), (512, 1))
        assert_size_stride(arg6_1, (512, 512), (512, 1))
        assert_size_stride(arg7_1, (512, ), (1, ))
        assert_size_stride(arg8_1, (512, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((8, 512, 512), (262144, 512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_0.run(arg2_1, arg0_1, arg1_1, buf3, 4096, 512, stream=stream0)
            del arg0_1
            del arg1_1
            buf4 = empty_strided_cuda((4096, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_mm_t_view_1.run(buf3, arg3_1, buf4, 256, 1, 1, stream=stream0)
            del arg3_1
            buf5 = empty_strided_cuda((4096, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_mm_t_view_2.run(buf3, arg4_1, buf5, 256, 1, 1, stream=stream0)
            del arg4_1
            buf6 = empty_strided_cuda((4096, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_mm_t_view_2.run(buf3, arg5_1, buf6, 256, 1, 1, stream=stream0)
            del arg5_1
            del buf3
            # Topologically Sorted Source Nodes: [linear, view, q, linear_1, view_1, k, linear_2, view_2, v, attn_out], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf7 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf4, (8, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf5, (8, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf6, (8, 8, 512, 64), (262144, 64, 512, 1), 0), scale=0.125)
            del buf4
            del buf5
            buf8 = buf7[0]
            assert_size_stride(buf8, (8, 8, 512, 64), (262144, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf8, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf7
            buf13 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, attn_out_1, out], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_mm_t_view_2.run(buf8, arg6_1, buf13, 256, 1, 1, stream=stream0)
            del arg6_1
            del buf8
            buf17 = reinterpret_tensor(buf13, (8, 512, 512), (262144, 512, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [out, add, out_1], Original ATen: [aten._unsafe_view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_native_layer_norm_3.run(buf17, arg2_1, arg7_1, arg8_1, 4096, 512, stream=stream0)
            del arg2_1
            del arg7_1
            del arg8_1
        return (buf17, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((8, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

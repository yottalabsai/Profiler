
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

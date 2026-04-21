"""
run_optimized_profiles.py — Generate profile_optimized.json for all optimized example workloads.

Runs the full two-phase pipeline (nsys capture → ncu replay) for each
optimized workload in examples/<name>/<name>_optimized.py, writing results to:
    examples/<name>/profile_optimized.json

Usage:
    python3 nvidia/scripts/run_optimized_profiles.py [--examples NAME [NAME ...]]

Options:
    --examples      Subset of example names to run (default: all)
    --ncu           Path to ncu binary (default: /opt/nvidia/nsight-compute/2025.4.1/ncu)
    --nsys          Path to nsys binary (default: nsys)
    --no-sudo       Do not prefix ncu with sudo
    --warmup-iters  Number of warm-up iterations (default: 2)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

NVIDIA_ROOT = Path(__file__).resolve().parent.parent  # .../Profiler/nvidia
PROFILER_ROOT = NVIDIA_ROOT.parent                    # .../Profiler
EXAMPLES_ROOT = PROFILER_ROOT / "examples"

sys.path.insert(0, str(PROFILER_ROOT))

from nvidia.operator_profiler.capture.nsys_runner import NsysRunConfig, run_nsys_profile
from nvidia.operator_profiler.mapper.manifest_builder import ManifestBuilder
from nvidia.operator_profiler.mapper.attribution_engine import AttributionEngine
from nvidia.operator_profiler.mapper.kernel_profiler import KernelProfileConfig, KernelProfileOrchestrator
from nvidia.operator_profiler.aggregator.profile_builder import build_profile
from nvidia.operator_profiler.schema.manifest import CaptureManifestMetadata

log = logging.getLogger(__name__)

RUN_WORKLOAD = NVIDIA_ROOT / "scripts" / "run_workload.py"

EXAMPLES = [
    {
        "name": "conv_block",
        "model_name": "ConvBlock-Optimized",
        "compile_backend": "convblock_opt",
    },
    {
        "name": "mlp_activations",
        "model_name": "MlpActivations-Optimized",
        "compile_backend": "transformer_opt",
    },
    {
        "name": "sdpa_attention",
        "model_name": "SdpaAttention-Optimized",
        "compile_backend": "transformer_opt",
    },
    {
        "name": "depthwise_separable_conv",
        "model_name": "DepthwiseSeparableConv-Optimized",
        "compile_backend": "transformer_opt",
    },
    {
        "name": "embedding_projection",
        "model_name": "EmbeddingProjection-Optimized",
        "compile_backend": "transformer_opt",
    },
]


def profile_optimized(
    example_info: dict,
    *,
    ncu_executable: str,
    nsys_executable: str,
    ncu_sudo: bool,
    warmup_iters: int,
) -> Path:
    name = example_info["name"]
    model_name = example_info["model_name"]
    compile_backend = example_info["compile_backend"]

    example_dir = EXAMPLES_ROOT / name
    workload_script = example_dir / f"{name}_optimized.py"

    out_dir = PROFILER_ROOT / "runs" / f"{name}_optimized"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = out_dir / f"{name}_optimized"

    # Both the nvidia package root and the example dir must be on PYTHONPATH.
    # The example dir is needed because optimized files import from their baseline
    # (e.g. `from conv_block import ConvBlock, ...`).
    # Include user site-packages so that torch (installed there) is accessible
    # when ncu runs the replay script under sudo, which drops PATH and site dirs.
    import site as _site
    site_pkgs = _site.getsitepackages() + [_site.getusersitepackages()]
    existing_pp = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(PROFILER_ROOT), str(example_dir)] + site_pkgs
    if existing_pp:
        pythonpath_parts.append(existing_pp)
    pythonpath = ":".join(pythonpath_parts)

    log.info("=" * 60)
    log.info("Example: %s (optimized)", name)
    log.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Phase 1 — nsys capture                                               #
    # ------------------------------------------------------------------ #
    log.info("[1/3] nsys capture → %s.nsys-rep", output_prefix)
    nsys_config = NsysRunConfig(
        script=str(RUN_WORKLOAD),
        script_args=[
            "--workload", str(workload_script),
            "--compile-backend", compile_backend,
            "--warmup-iters", str(warmup_iters),
        ],
        output_path=str(output_prefix),
        nsys_executable=nsys_executable,
        extra_env={"PYTHONPATH": pythonpath},
    )
    rep_path = run_nsys_profile(nsys_config)
    log.info("nsys report: %s", rep_path)

    # ------------------------------------------------------------------ #
    # Phase 2 — build manifest                                             #
    # ------------------------------------------------------------------ #
    log.info("[2/3] building mapping manifest")
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "unknown"

    metadata = CaptureManifestMetadata(
        model_name=model_name,
        torch_version=torch_version,
        compile_mode="inductor",
        nsys_report_path=str(rep_path),
        capture_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    builder = ManifestBuilder(nsys_rep_path=rep_path, metadata=metadata, nsys_executable=nsys_executable)
    manifest = builder.build()

    manifest_path = output_prefix.with_suffix(".manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    log.info("Manifest → %s (%d kernels)", manifest_path, len(manifest.kernels))

    # ------------------------------------------------------------------ #
    # Phase 3 — ncu replay + attribution + profile assembly               #
    # ------------------------------------------------------------------ #
    log.info("[3/3] ncu replay + profile assembly")
    engine = AttributionEngine(manifest)
    operator_records, unattributed = engine.run()

    replay_config = KernelProfileConfig(
        replay_script=str(RUN_WORKLOAD),
        replay_script_args=[
            "--workload", str(workload_script),
            "--compile-backend", compile_backend,
            "--warmup-iters", str(warmup_iters),
        ],
        ncu_executable=ncu_executable,
        ncu_sudo=ncu_sudo,
        # sudo drops PYTHONPATH — pass it explicitly so the replay script can
        # import both the nvidia package and the example's baseline module.
        ncu_extra_env={"PYTHONPATH": pythonpath},
        expected_input_shapes=manifest.capture_metadata.input_shapes,
    )
    orch = KernelProfileOrchestrator(manifest, operator_records, replay_config)
    ncu_output_dir = orch.run()

    profile = build_profile(
        manifest=manifest,
        operator_records=operator_records,
        unattributed_kernels=unattributed,
        model_name=model_name,
        torch_version=torch_version,
        device_name=None,
        ncu_report_path=str(ncu_output_dir),
    )

    profile_path = example_dir / "profile_optimized.json"
    profile_path.write_text(profile.model_dump_json(indent=2))
    log.info("Profile → %s", profile_path)
    return profile_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    all_names = [e["name"] for e in EXAMPLES]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--examples", nargs="+", default=all_names, metavar="NAME",
        help=f"Examples to profile (default: all). Choices: {', '.join(all_names)}",
    )
    parser.add_argument(
        "--ncu", default="/opt/nvidia/nsight-compute/2025.4.1/ncu",
        metavar="PATH", help="Path to ncu binary",
    )
    parser.add_argument("--nsys", default="nsys", metavar="PATH", help="Path to nsys binary")
    parser.add_argument("--no-sudo", action="store_true", help="Do not run ncu under sudo")
    parser.add_argument("--warmup-iters", type=int, default=2)
    args = parser.parse_args()

    unknown = [n for n in args.examples if n not in all_names]
    if unknown:
        parser.error(f"Unknown example(s): {', '.join(unknown)}. Choices: {', '.join(all_names)}")

    selected = [e for e in EXAMPLES if e["name"] in args.examples]

    results: dict[str, str] = {}
    for example_info in selected:
        name = example_info["name"]
        try:
            path = profile_optimized(
                example_info,
                ncu_executable=args.ncu,
                nsys_executable=args.nsys,
                ncu_sudo=not args.no_sudo,
                warmup_iters=args.warmup_iters,
            )
            results[name] = str(path)
        except Exception:
            log.exception("Failed to profile %s (optimized)", name)
            results[name] = "FAILED"

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    for name, path in results.items():
        status = "OK" if path != "FAILED" else "FAILED"
        print(f"  [{status}] {name}: {path}")


if __name__ == "__main__":
    main()

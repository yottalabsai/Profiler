"""
run_all_profiles.py — Generate profile.json for every example workload.

Runs the full two-phase pipeline (nsys capture → ncu replay) for each
workload in scripts/workloads/, writing results to runs/<name>/:
    runs/<name>/<name>.nsys-rep
    runs/<name>/<name>.manifest.json
    runs/<name>/<name>_profile.json

Usage:
    python3 scripts/run_all_profiles.py [--workloads NAME [NAME ...]]

Options:
    --workloads     Subset of workload names to run (default: all)
    --compile-mode  eager | inductor | cudagraphs (default: inductor)
    --warmup-iters  Number of warm-up iterations (default: 2)
    --no-diagnose   Skip DiagnosisAgent (skips Anthropic API call)
    --ncu           Path to ncu binary (default: /opt/nvidia/nsight-compute/2025.4.1/ncu)
    --nsys          Path to nsys binary (default: nsys)
    --no-sudo       Do not prefix ncu with sudo (if perf counters are unrestricted)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is importable whether or not the package is installed.
sys.path.insert(0, str(Path(__file__).parent.parent))

from operator_profiler.capture.nsys_runner import NsysRunConfig, run_nsys_profile
from operator_profiler.mapper.manifest_builder import ManifestBuilder
from operator_profiler.mapper.attribution_engine import AttributionEngine
from operator_profiler.mapper.kernel_profiler import KernelProfileConfig, KernelProfileOrchestrator
from operator_profiler.aggregator.profile_builder import build_profile
from operator_profiler.schema.manifest import CaptureManifestMetadata

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.resolve()

WORKLOADS = [
    "transformer_block",
    "conv_block",
    "mlp_activations",
    "sdpa_attention",
    "depthwise_separable_conv",
    "embedding_projection",
]


def profile_one(
    name: str,
    *,
    compile_mode: str,
    warmup_iters: int,
    ncu_executable: str,
    nsys_executable: str,
    ncu_sudo: bool,
    diagnose: bool,
) -> Path:
    """Run the full pipeline for a single workload. Returns path to profile.json."""
    script = REPO_ROOT / "scripts" / "workloads" / f"{name}.py"
    out_dir = REPO_ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = "".join(part.capitalize() for part in name.split("_"))
    output_prefix = out_dir / name

    log.info("=" * 60)
    log.info("Workload: %s", name)
    log.info("=" * 60)

    run_workload = REPO_ROOT / "scripts" / "run_workload.py"

    # ------------------------------------------------------------------ #
    # Phase 1 — nsys capture                                               #
    # ------------------------------------------------------------------ #
    log.info("[1/3] nsys capture → %s.nsys-rep", output_prefix)
    nsys_config = NsysRunConfig(
        # run_workload.py handles compile, warmup, and emit_nvtx wrapping.
        script=str(run_workload),
        script_args=[
            "--workload", str(script),
            "--compile-backend", compile_mode,
            "--warmup-iters", str(warmup_iters),
        ],
        output_path=str(output_prefix),
        nsys_executable=nsys_executable,
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
        compile_mode=compile_mode,
        nsys_report_path=str(rep_path),
        capture_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    builder = ManifestBuilder(nsys_rep_path=rep_path, metadata=metadata)
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
        # ncu replays the same run_workload.py invocation used for capture.
        replay_script=str(run_workload),
        replay_script_args=[
            "--workload", str(script),
            "--compile-backend", compile_mode,
            "--warmup-iters", str(warmup_iters),
        ],
        ncu_executable=ncu_executable,
        ncu_sudo=ncu_sudo,
        # sudo -E drops PYTHONPATH on most systems; pass it explicitly.
        ncu_extra_env={"PYTHONPATH": str(REPO_ROOT)},
        expected_input_shapes=manifest.capture_metadata.input_shapes,
    )
    orch = KernelProfileOrchestrator(manifest, operator_records, replay_config)
    ncu_output_dir = orch.run()

    diagnosis_agent = None
    if diagnose:
        try:
            from operator_profiler.agents.diagnosis import DiagnosisAgent
            diagnosis_agent = DiagnosisAgent()
            log.info("DiagnosisAgent enabled")
        except Exception as exc:
            log.warning("DiagnosisAgent unavailable (%s); using roofline heuristics", exc)

    profile = build_profile(
        manifest=manifest,
        operator_records=operator_records,
        unattributed_kernels=unattributed,
        model_name=model_name,
        torch_version=torch_version,
        device_name=None,   # auto-detected from manifest
        ncu_report_path=str(ncu_output_dir),
        diagnosis_agent=diagnosis_agent,
    )

    profile_path = output_prefix.with_name(f"{name}_profile.json")
    profile_path.write_text(profile.model_dump_json(indent=2))
    log.info("Profile → %s", profile_path)
    return profile_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workloads", nargs="+", default=WORKLOADS, metavar="NAME",
        help=f"Workloads to profile (default: all). Choices: {', '.join(WORKLOADS)}",
    )
    parser.add_argument("--compile-mode", default="inductor", choices=["eager", "inductor", "cudagraphs"])
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument(
        "--ncu", default="/opt/nvidia/nsight-compute/2025.4.1/ncu",
        metavar="PATH", help="Path to ncu binary",
    )
    parser.add_argument("--nsys", default="nsys", metavar="PATH", help="Path to nsys binary")
    parser.add_argument(
        "--no-sudo", action="store_true",
        help="Do not run ncu under sudo (use if perf counters are unrestricted)",
    )
    parser.add_argument(
        "--no-diagnose", action="store_true",
        help="Skip DiagnosisAgent LLM bottleneck classification",
    )
    args = parser.parse_args()

    unknown = [w for w in args.workloads if w not in WORKLOADS]
    if unknown:
        parser.error(f"Unknown workload(s): {', '.join(unknown)}. Choices: {', '.join(WORKLOADS)}")

    results: dict[str, str] = {}
    for name in args.workloads:
        try:
            path = profile_one(
                name,
                compile_mode=args.compile_mode,
                warmup_iters=args.warmup_iters,
                ncu_executable=args.ncu,
                nsys_executable=args.nsys,
                ncu_sudo=not args.no_sudo,
                diagnose=not args.no_diagnose,
            )
            results[name] = str(path)
        except Exception:
            log.exception("Failed to profile %s", name)
            results[name] = "FAILED"

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    for name, path in results.items():
        status = "OK" if path != "FAILED" else "FAILED"
        print(f"  [{status}] {name}: {path}")


if __name__ == "__main__":
    main()

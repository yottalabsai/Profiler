"""
LLM-backed agent modules for the Operator Profiler pipeline.

  DiagnosisAgent  — LLM-reasoned bottleneck classification over the full
                    KernelMetrics set for each operator, relative to
                    model-wide metric distributions.
"""
from operator_profiler.agents.diagnosis import DiagnosisAgent, DiagnosisResult, ModelStats

__all__ = [
    "DiagnosisAgent",
    "DiagnosisResult",
    "ModelStats",
]

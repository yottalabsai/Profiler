"""
OperatorProfiler — correlates GPU hardware metrics with PyTorch operators.

Pipeline:
    Profiler Capture Stage → Operator Mapper Stage → Metric Aggregation & Report Stage

Quick start:
    from operator_profiler.schema import OperatorAttributedProfile
    from operator_profiler.mapper import ManifestBuilder
    from operator_profiler.aggregator import build_profile
"""
__version__ = "0.1.0"

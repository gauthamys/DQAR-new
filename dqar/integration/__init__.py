"""Integration modules for DQAR with diffusion models."""

from .dit_wrapper import DQARDiTWrapper, DQARConfig
from .samplers import DQARDDIMSampler, DQARDPMSolverSampler, SamplerConfig
from .manager import DQARManager
from .attention_processor import (
    DQARAttentionProcessor,
    DQARJointAttentionProcessor,
    get_processor_class_for_model,
)

__all__ = [
    "DQARDiTWrapper",
    "DQARConfig",
    "DQARDDIMSampler",
    "DQARDPMSolverSampler",
    "SamplerConfig",
    "DQARManager",
    "DQARAttentionProcessor",
    "DQARJointAttentionProcessor",
    "get_processor_class_for_model",
]

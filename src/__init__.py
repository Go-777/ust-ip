"""
Agentic Memory System with Co-evolving Memory and Operation Banks
"""

from .memory_bank import MemoryBank
from .operation_bank import OperationBank

# Lazy import for Executor (depends on llm_utils which requires tiktoken/transformers)
try:
    from .executor import Executor
except ImportError:
    Executor = None

# Lazy imports for torch-dependent modules
try:
    from .controller import PPOController
    from .llm_controller import LLMController
    from .designer import Designer
    from .trainer import BaseTrainer, OfflineTrainer, get_trainer
    _TORCH_AVAILABLE = True
except ImportError:
    PPOController = None
    LLMController = None
    Designer = None
    BaseTrainer = None
    OfflineTrainer = None
    get_trainer = None
    _TORCH_AVAILABLE = False

# Data processing and evaluation modules
try:
    from . import data_processing
    from . import eval
except ImportError:
    data_processing = None
    eval = None

__all__ = [
    'MemoryBank',
    'OperationBank',
    'PPOController',
    'LLMController',
    'Executor',
    'Designer',
    'BaseTrainer',
    'OfflineTrainer',
    'get_trainer',
    # Submodules
    'data_processing',
    'eval',
]

__version__ = "1.2.0"

from mamba.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba.modules.mamba_simple import Mamba
from mamba.models.mixer_seq_simple import MambaLMHeadModel

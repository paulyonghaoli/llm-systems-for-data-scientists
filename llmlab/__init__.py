"""llmlab — the executable core of *LLM Systems for Data Scientists*.

Pure Python + NumPy so the identical code runs in CPython (CI) and in Pyodide
(the browser). Nothing here touches the network; that is a CI gate, not a
convention.
"""

from llmlab.tokenizer import BPETokenizer, merge, pair_counts, pretokenize
from llmlab.tools import Sandbox, ToolSpec, safe_eval, validate_call

__all__ = ["BPETokenizer", "Sandbox", "ToolSpec", "merge", "pair_counts",
           "pretokenize", "safe_eval", "validate_call"]
__version__ = "0.1.0"

from .tools.tool_reason import (
    TOOLS_UNIFIED_FINAL,
    TOOLS_UNIFIED_FINAL_1229,
)


TOOL_MAP = {
    "TOOLS_UNIFIED_FINAL": TOOLS_UNIFIED_FINAL,
    "TOOLS_UNIFIED_FINAL_1229": TOOLS_UNIFIED_FINAL_1229,
}

TOOL_EXECUTORS = {
    "execute_step": None,
    "commit": None,
    "expand": None,
    "fold": None,
    "final_answer": None,
}

from typing import Dict
from .tools.tool_search import Search
from .tools.tool_visit import Visit
from .tools.tool_agent import *
from .tools.tool_agent import expand_deepresearch_tool, fold_deepresearch_tool


search_tool = Search()
visit_tool = Visit()




DEEPRESEARCH_TOOLS = [
    search_tool._get_schema(),
    visit_tool._get_schema(),
    log_step_tool,
    expand_deepresearch_tool, 
    fold_deepresearch_tool, 
]

DEEPRESEARCH_BASE = [
    search_tool._get_schema(),
    visit_tool._get_schema(),
]

TOOL_EXECUTORS = {
    "search": search_tool,
    "visit": visit_tool,
    "execute_step": None,
    "expand": None,
    "fold": None,
    "final_answer": None
}


TOOL_MAP = {
    "DEEPRESEARCH": DEEPRESEARCH_TOOLS,
    "DEEPRESEARCH_BASE": DEEPRESEARCH_BASE
}
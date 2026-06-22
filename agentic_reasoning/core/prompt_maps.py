from .prompts.agent import *

PROMPTS_MAP = {

    # dual agent (synthetic data generation)
    "synthetic_main_agent": SYNTHETIC_MAIN_AGENT_PROMPT,

    # single agent (inference)
    "infer_agent": INFER_AGENT_PROMPT,

    # logger / summarizer
    "visit_logger": VISIT_LOGGER_PROMPT,
    "search_logger": SEARCH_LOGGER_PROMPT,
}

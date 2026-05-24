# Compatibility shim — retriever agent has moved to struct_memory_r1.
from struct_memory_r1.agents.retrieve_agent import (  # noqa: F401
    parse_retriever_output,
    FrozenAnswerAgent,
    set_frozen_answer_agent,
    get_frozen_answer_agent,
    compute_score_struct_retriever_verl,
)

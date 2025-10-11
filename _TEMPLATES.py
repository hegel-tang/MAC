import json

from templates.Multi_Agent import Parser_prompt,EqBuilder_prompt,Calculator_prompt,Solver_prompt,Critic_prompt,Solver_revision_prompt

def _select_reply(item_output, selection_mode="first", join_sep="\n"):
    """
    item_output can be str or list[str].
    selection_mode:
      - "first": use the first generation (default)
      - "concat": concat all generations with join_sep
      - "json": dump the list as JSON string
      - "best": placeholder (same as first unless you implement ranking)
    """
    if isinstance(item_output, list):
        if selection_mode == "first":
            return item_output[0] if len(item_output) > 0 else ""
    else:
        return str(item_output)

def apply_ma_template(item, agent_index, question_key, reply_selection_mode="first"):
    """
    Build prompt string for given agent and item.
      - item: dict loaded from dataset (for agent>0 likely contains 'output' list)
      - agent_index: 0/1/2...
      - question_key: key in item for the question (for agent0 typically 'question')
      - reply_selection_mode: how to handle list outputs from previous agents
    """
    if agent_index == 0:
        question = item.get(question_key, "")
        prompt_str = Solver_prompt[:]
        prompt_str = prompt_str.replace("{question}", str(question))
        return prompt_str

    if agent_index == 1:
        # item["output"] may be a list; pick according to selection mode
        parser_output_raw = item.get("output", "")
        parser_output = _select_reply(parser_output_raw, selection_mode=reply_selection_mode)
        prompt_str = Critic_prompt[:]
        prompt_str = prompt_str.replace("{Solver_output}", parser_output)
        return prompt_str

    if agent_index == 2:
        eq_output_raw = item.get("output", "")
        eq_output = _select_reply(eq_output_raw, selection_mode=reply_selection_mode)
        prompt_str = Solver_revision_prompt[:]
        prompt_str = prompt_str.replace("{Critic_output}", eq_output)
        return prompt_str

    # generic fallback
    return ""
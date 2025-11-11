import json

from templates.Multi_Agent import Parser_prompt,EqBuilder_prompt,Calculator_prompt,Solver_prompt,Critic_prompt,Reviser_prompt

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

def apply_ma_template(data_name, item, agent_index, question_key, reply_selection_mode="first"):
    """
    Build prompt string for given agent and item.
      - item: dict loaded from dataset (for agent>0 likely contains 'output' list)
      - agent_index: 0/1/2...
      - question_key: key in item for the question (for agent0 typically 'question')
      - reply_selection_mode: how to handle list outputs from previous agents
    """
    if data_name == "gsm" :
        if agent_index == 0:
            question = item.get(question_key, "")
            prompt_str = Solver_prompt[:]
            prompt_str = prompt_str.replace("{question}", str(question))
            return prompt_str

        if agent_index == 1:
            # item["output"] may be a list; pick according to selection mode
            solver_output_raw = item.get("output", "")
            solver_output = _select_reply(solver_output_raw, selection_mode=reply_selection_mode)
            origin_question_raw = item.get("question","")
            origin_question = _select_reply(origin_question_raw, selection_mode=reply_selection_mode)
            prompt_str = Critic_prompt[:]
            prompt_str = prompt_str.replace("{question}", origin_question)
            prompt_str = prompt_str.replace("{Solver_output}", solver_output)
            return prompt_str

        if agent_index == 2:
            solver_output_raw = item.get("prev_output_2", "")
            solver_output = _select_reply(solver_output_raw, selection_mode=reply_selection_mode)
            critic_output_raw = item.get("output", "")
            critic_output = _select_reply(solver_output_raw, selection_mode=reply_selection_mode)
            origin_question_raw = item.get("question","")
            origin_question = _select_reply(origin_question_raw, selection_mode=reply_selection_mode)
            prompt_str = Reviser_prompt[:]
            prompt_str = prompt_str.replace("{question}", origin_question)
            prompt_str = prompt_str.replace("{Solver_output}", solver_output)
            prompt_str = prompt_str.replace("{Critic_output}", critic_output)
            return prompt_str
    if data_name == "gsm_hard" :
        if agent_index == 0:
            question = item.get(question_key, "")
            prompt_str = Solver_prompt[:]
            prompt_str = prompt_str.replace("{question}", str(question))
            return prompt_str

        if agent_index == 1:
            # item["output"] may be a list; pick according to selection mode
            solver_output_raw = item.get("output", "")
            solver_output = _select_reply(solver_output_raw, selection_mode=reply_selection_mode)
            origin_question_raw = item.get("question","")
            origin_question = _select_reply(origin_question_raw, selection_mode=reply_selection_mode)
            prompt_str = Critic_prompt[:]
            prompt_str = prompt_str.replace("{question}", origin_question)
            prompt_str = prompt_str.replace("{Solver_output}", solver_output)
            return prompt_str

        if agent_index == 2:
            solver_output_raw = item.get("prev_output_2", "")
            solver_output = _select_reply(solver_output_raw, selection_mode=reply_selection_mode)
            critic_output_raw = item.get("output", "")
            critic_output = _select_reply(critic_output_raw, selection_mode=reply_selection_mode)
            origin_question_raw = item.get("question","")
            origin_question = _select_reply(origin_question_raw, selection_mode=reply_selection_mode)
            prompt_str = Reviser_prompt[:]
            prompt_str = prompt_str.replace("{question}", origin_question)
            prompt_str = prompt_str.replace("{Solver_output}", solver_output)
            prompt_str = prompt_str.replace("{Critic_output}", critic_output)
            return prompt_str
    # generic fallback
    return ""

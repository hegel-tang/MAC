import sys
import os
import time
from functools import wraps
from typing import List
from tqdm import tqdm
from fastchat_conversation import map_to_conv, HF_Conversation
import json
from task_configs import mapping_task_names, prompt_generation, result_format


def apply_template(chat_history, model_name, args, agent_index=None):
    """
    Convert chat_history (list of chats) -> model_inputs (list of prompt strings).
    - chat_history: list, each element is a list of turns (each turn ideally a str).
    - model_name: passed to map_to_conv to pick conversation format.
    - args: used for engine check and optionally verbosity.
    - agent_index: optional, provided for compatibility if different agents require different templates.
    """
    model_inputs = []
    conv = None
    last_model_for_conv = None

    disable_tqdm = getattr(args, "disable_tqdm", False) or True  # preserve your prior default

    for chats in tqdm(chat_history, desc="Applying template", disable=disable_tqdm):
        # only support vllm/hf engines here (same as original)
        if args.engine not in ["vllm", "hf"]:
            model_inputs.append("n/a")
            continue

        # Recreate conv if model_name changed or conv not initialized
        if conv is None or last_model_for_conv != model_name or not isinstance(conv, HF_Conversation):
            conv = map_to_conv(model_name)
            last_model_for_conv = model_name
        else:
            # clear conversation for reuse
            try:
                conv.clear()
            except Exception:
                # fallback: recreate
                conv = map_to_conv(model_name)
                last_model_for_conv = model_name

        # Append each chat turn safely (convert non-string to string)
        for chat_id, chat in enumerate(chats):
            # normalize chat into a string
            if chat is None:
                text = ""
            elif isinstance(chat, str):
                text = chat
            elif isinstance(chat, list):
                # join list elements (often happens if upstream mistakenly passed list)
                text = "\n".join([str(x) for x in chat])
            elif isinstance(chat, dict):
                # common keys: 'text', 'content', 'message'
                text = chat.get("text") or chat.get("content") or chat.get("message") or str(chat)
            else:
                text = str(chat)

            # append to conv with alternating roles assumption as before
            role = conv.roles[chat_id % 2]
            conv.append_message(role, text)

        # add assistant placeholder (so conv.get_prompt() expects model to fill in)
        conv.append_message(conv.roles[1], None)

        # final prompt for this example
        model_inputs.append(conv.get_prompt())

    return model_inputs

def load_eval_data(args, agent_index=None, selected = False, data_name=None, model_name=None):
    """
    return id_strs, chat_history, model_inputs, metadata
    """
    if data_name is None:
        data_name = args.data_name
    if model_name is None:
        model_name = args.model_name

    dataset, id_name = mapping_task_names(data_name, agent_index, selected)
    print(f"Loaded {len(dataset)} examples from {data_name} (agent_index={agent_index})")

    expanded_id_strs = []
    expanded_chat_history = []
    expanded_metadata = {}

    for ind, item in enumerate(dataset):
        orig_session_id = item.get(id_name, f"{data_name}#{ind}")

        # detect upstream outputs: when agent_index > 0, items produced by previous agent should have "output"
        if agent_index is not None and agent_index > 0 and "output" in item:
            outs = item["output"]
            
            # unify type: if string, treat as single-element list
            if isinstance(outs, str):
                outs = [outs]
            if outs is None:
                outs = [""]
           
            # for each generated reply, create a separate expanded sample
            for j, out_j in enumerate(outs):
                # unique session id per generated reply to trace back
                sid = f"{orig_session_id}#gen{j}"

                # We will generate a prompt for this specific reply; to enable prompt_generation to consume it,
                # we create a temporary item copy where "output" is replaced by the single string out_j.
                item_copy = dict(item)  # shallow copy; safe if values are primitives/strings
                item_copy["output"] = out_j
                #print(item_copy)
                # generate prompt using prompt_generation (try new signature first)
                try:
                    prompt = prompt_generation(data_name, item_copy, args, agent_index)
                except TypeError:
                    prompt = prompt_generation(data_name, item_copy, args)
                #print(prompt)
                expanded_id_strs.append(sid)
                expanded_chat_history.append([prompt])

                # metadata: for each key in original item, append corresponding value (duplicate for each expansion)
                for key in item:
                    # set output metadata to the single string out_j
                    if key == "output":
                        expanded_metadata.setdefault(key, []).append(out_j)
                    else:
                        expanded_metadata.setdefault(key, []).append(item.get(key))
                
        else:
            # agent_index == 0 or no "output" field: treat as normal (single sample)
            sid = orig_session_id
            # prompt generation (ask with agent_index so template can adapt)
            try:
                prompt = prompt_generation(data_name, item, args, agent_index)
            except TypeError:
                prompt = prompt_generation(data_name, item, args)

            expanded_id_strs.append(sid)
            expanded_chat_history.append([prompt])
            for key in item:
                expanded_metadata.setdefault(key, []).append(item.get(key))

    # Now expanded_chat_history contains one prompt per (expanded) sample.
    # Apply template to create model_inputs (keep backwards-compatible call)
    try:
        model_inputs = apply_template(expanded_chat_history, model_name, args, agent_index)
    except TypeError:
        model_inputs = apply_template(expanded_chat_history, model_name, args)


    return expanded_id_strs, expanded_chat_history, model_inputs, expanded_metadata


def clear_output(output, model_name):
    """
    You can customize the output clearing logic here based on the model_name.
    """
    if isinstance(output, list):
        output = output[0]
    assert isinstance(output, str), f"the type of output is {type(output)}"
    # print(f"the output is {output}")
    output = output.replace("<|endoftext|>", " ")
    output = output.replace("<pad>", " ")
    output = output.replace("<end_of_turn>", " ")
    output = output.strip()
    return output


def save_outputs(
    args, id_strs, outputs, chat_history, metadata, model_inputs, filepath
):
    formatted_outputs = []
    for ind in range(len(outputs)):
        output_item = {}
        output_item["session_id"] = id_strs[ind]
        output_item["chat_history"] = chat_history[ind]
        output_item["model_input"] = model_inputs[ind]
        output_item["output"] = [clear_output(o, args.model_name) for o in outputs[ind]]
        output_item["generator"] = args.model_name
        output_item["configs"] = {
            "engine": args.engine,
            "repetition_penalty": args.repetition_penalty,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            # "cot": args.cot,
        }
        output_item["dataset"] = args.data_name
        for key in metadata:
            if key in output_item:
                continue
            if ind < len(metadata[key]):
                output_item[key] = metadata[key][ind]
        output_item = result_format(output_item, args)
        formatted_outputs.append(output_item)
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(formatted_outputs, f, indent=2)




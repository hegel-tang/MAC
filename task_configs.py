import os
from datasets import load_dataset
from _TEMPLATES import apply_ma_template


        
def mapping_task_names(data_name,agent_index,selected,baseline=False):
    """
    Mapping the task names to the dataset and id name.
    """
    id_name = "id"
    if agent_index == 0:
        if data_name == "gsm":
            dataset = load_dataset("json", data_files={"test": "gsm8K.json"}, split="test")
        if data_name == "gsm_hard":
            dataset = load_dataset("json", data_files={"test": "gsm_hard.json"}, split="test")
    else:
        if selected:
            path = f"result_dirs/{data_name}/agent{agent_index}_conf_selected.json"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected previous agent outputs at {path} but not found.")
            dataset = load_dataset("json", data_files={"test": path}, split="test")
        else: 
            if baseline:
                path = f"result_dirs/{data_name}/agent{agent_index-1}_baseline_output.json"
            else:
                path = f"result_dirs/{data_name}/agent{agent_index-1}_output.json"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected previous agent outputs at {path} but not found.")
            dataset = load_dataset("json", data_files={"test": path}, split="test")
    return dataset, id_name

def prompt_generation(data_name, data_item, args, agent_index):
    """
    Generate prompt for different tasks.
    """

    if data_name in ["gsm"] or "math" in data_name:
        if agent_index == 0:
            question_key = "question"
        if agent_index == 1:
            question_key = "Parser_output"
        if agent_index == 2:
            question_key = "Eqbuilder_output"
        prompt = apply_ma_template(data_name, data_item, agent_index, question_key = question_key)
    
    return prompt

def result_format(output_item, args):
    """
    Modify the output format for different tasks if needed.
    """
    if args.data_name in ["alpaca_eval"]:
        output_item["output"] = output_item["output"][0] # use str instead of list 
    elif args.data_name in ["zebra-grid"]:
        if "solution" in output_item:
            del output_item["solution"]
    elif args.data_name in ["wildbench_v2-hard"]:
        for key in ["conversation_input", "references", "length", "checklist", "avg_score", "var_score"]:
            if key in output_item:
                del output_item[key]

    else:
        pass 
    return output_item

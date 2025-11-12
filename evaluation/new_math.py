import json
import os
from tabulate import tabulate
import re
import argparse
from typing import Optional
from eval_utils import load_model_results, extract_values_from_json, extract_first_complete_json, model_specific_extraction, model_name_replacement
from livebench_utils import amps_hard_process_results_with_processed_output, is_amps_hard
from eval_utils_padding import sanitize_math_answers, convert_AAAAA_to_A, extract_answer_from_output, boxed_extraction


def eval_model(model, filepath):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0 
    num_total_examples = len(data) 
    no_answer = 0  
    
    reason_lens = []
    parsed_results = [] 
    for item in data:  
        model_dir = item["generator"]
        # Read and Parse the prediction from model output
        parsed_item = item.copy()
        prediction_str = item["output"][min(0, len(item["output"])-1)]
        prediction_json = extract_first_complete_json(prediction_str)
        flag_parsed_answer = True
        if prediction_json is None or "answer" not in prediction_json:
            prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
        if prediction_json is None or "answer" not in prediction_json: 
            try_extracted_answer = model_specific_extraction(model_dir, prediction_str)
            if try_extracted_answer:
                prediction_json["answer"] = try_extracted_answer
            else:
                no_answer += 1 
                flag_parsed_answer = False 
       
        reason = prediction_json.get("reasoning", "")
        correct_answer = item["answer"].replace("#", "").strip()
        model_answer = None 
        if not flag_parsed_answer:
            parsed_item["model_answer"] = {"raw": None, "sanitized": None, "first_number": None} # not matched 
            parsed_item["correct_answer"] = {"raw": correct_answer}
            parsed_item["matched"] = "No answer extracted"
            parsed_results.append(parsed_item) 
            continue
        else:
            model_answer = str(prediction_json["answer"])
            
        # sanitize the answers
        model_answer = convert_AAAAA_to_A(model_answer)
        
        raw_model_answer = model_answer[:]
        model_answer = sanitize_math_answers(model_answer)
        correct_answer = sanitize_math_answers(correct_answer)
        
        first_number_in_model_answer = re.search(r"-?\d+(\.\d+)?", model_answer)
        first_number_in_correct_answer = re.search(r"-?\d+(\.\d+)?", correct_answer)
        correct = False 
        #print(model_answer)
        #print(correct_answer)
        
        if model_answer == correct_answer or correct_answer in model_answer:
            correct = True
        elif "livebench" not in item["dataset"]:
            if first_number_in_correct_answer and first_number_in_model_answer:
                if float(first_number_in_model_answer.group()) == float(first_number_in_correct_answer.group()):
                    correct = True
        elif is_amps_hard(item) and amps_hard_process_results_with_processed_output(correct_answer, model_answer):
            correct = True
        #print(correct)
        #print(solved_examples)
        def str_to_float(s: object, default: Optional[float]=None) -> Optional[float]:
            if s is None:
                return default
            if isinstance(s, (int, float)):
                return float(s)
            if not isinstance(s, str):
                return default
            s = s.strip()
            try:
                return float(s)
            except ValueError:
                pass
            cleaned = re.sub(r"[^\d\.\-eE,]", "", s)  
            if "," in cleaned and "." not in cleaned:
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
            try:
                return float(cleaned)
            except ValueError:
                return default
        if str_to_float(model_answer) is not None and str_to_float(correct_answer) is not None:
            if abs(str_to_float(model_answer) - str_to_float(correct_answer)) < 1:
                correct = True
        if correct:
            solved_examples += 1

        reason_lens.append(len(reason))

        parsed_item["model_answer"] = {"raw": raw_model_answer, "sanitized": model_answer, "first_number": first_number_in_model_answer.group() if first_number_in_model_answer else None}
        parsed_item["correct_answer"] = {"raw": correct_answer, "sanitized": correct_answer, "first_number": first_number_in_correct_answer.group() if first_number_in_correct_answer else None}
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)


 
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = "Eval"
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Model"] = model_name_replacement(result["Model"])
    return result, parsed_results

def eval_model_best(model, filepath, best_N):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0
    num_total_examples = len(data)
    no_answer = 0

    reason_lens = []
    parsed_results = []

    for item in data:
        parsed_item = item.copy()
        model_dir = item.get("generator", None)
        find_response = False

        # 局部 n_for_item，避免修改传入的 best_N
        if best_N == -1:
            n_for_item = len(item.get("output", []))
        else:
            n_for_item = min(best_N, len(item.get("output", [])))

        # 预先初始化，避免 UnboundLocalError
        reason = ""
        raw_model_answer = None
        model_answer = None
        model_answer_sanitized = None
        correct = False

        # 始终初始化正确答案（在循环外），以便后面安全引用
        correct_answer_raw = item.get("answer", "")
        correct_answer_raw = correct_answer_raw.replace("#", "").strip()
        correct_answer_sanitized = sanitize_math_answers(correct_answer_raw)
        first_number_in_correct_answer = re.search(r"-?\d+(\.\d+)?", correct_answer_sanitized)

        first_number_in_model_answer = None

        outputs = item.get("output", [])
        for i in range(n_for_item):
            # 保护性取值，防止 index error（尽管 n_for_item 已用 len 限定）
            prediction_str = outputs[i] if i < len(outputs) else ""
            # 解析预测
            prediction_json = extract_first_complete_json(prediction_str)
            if prediction_json is None or "answer" not in prediction_json:
                prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
            if prediction_json is None or "answer" not in prediction_json:
                try_extracted_answer = model_specific_extraction(model_dir, prediction_str)
                if try_extracted_answer:
                    prediction_json = {"answer": try_extracted_answer}
                else:
                    # 该条预测无法解析出答案，继续下一条
                    continue

            # 成功从该条预测中拿到答案
            find_response = True
            reason = prediction_json.get("reasoning", "") or ""

            model_answer = str(prediction_json["answer"])
            model_answer = convert_AAAAA_to_A(model_answer)
            raw_model_answer = model_answer[:]
            model_answer_sanitized = sanitize_math_answers(model_answer)

            first_number_in_model_answer = re.search(r"-?\d+(\.\d+)?", model_answer_sanitized)

            # 比对逻辑（保留你原有的策略）
            if correct_answer_sanitized and correct_answer_sanitized in model_answer_sanitized:
                correct = True
            if model_answer_sanitized == correct_answer_sanitized:
                correct = True
            elif "livebench" not in item.get("dataset", ""):
                if first_number_in_correct_answer and first_number_in_model_answer:
                    try:
                        if float(first_number_in_model_answer.group()) == float(first_number_in_correct_answer.group()):
                            correct = True
                    except Exception:
                        pass
            elif is_amps_hard(item) and amps_hard_process_results_with_processed_output(correct_answer_sanitized, model_answer_sanitized):
                correct = True

            if correct:
                solved_examples += 1
                # best 模式：找到正确答案就停止检查该样本的其余预测
                break

        # 内层循环后收尾
        if not find_response:
            no_answer += 1
            parsed_item["model_answer"] = {"raw": None, "sanitized": None, "first_number": None}
        else:
            parsed_item["model_answer"] = {
                "raw": raw_model_answer,
                "sanitized": model_answer_sanitized,
                "first_number": (first_number_in_model_answer.group() if first_number_in_model_answer else None)
            }

        # reason 已被初始化，安全追加
        reason_lens.append(len(reason))
        parsed_item["reasoning"] = reason
        parsed_item["correct_answer"] = {
            "raw": correct_answer_raw,
            "sanitized": correct_answer_sanitized,
            "first_number": (first_number_in_correct_answer.group() if first_number_in_correct_answer else None)
        }
        parsed_item["matched"] = bool(correct)
        parsed_results.append(parsed_item)

    # 汇总结果（防止 reason_lens 为空）
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = "best"
    result["Acc"] = f"{(solved_examples / num_total_examples * 100) if num_total_examples>0 else 0:.2f}"
    result["No answer"] = f"{(no_answer / num_total_examples * 100) if num_total_examples>0 else 0:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{(sum(reason_lens) / len(reason_lens)) if len(reason_lens)>0 else 0:.2f}"
    result["Model"] = model_name_replacement(result["Model"])
    return result, parsed_results


def eval_model_first_answered(model, filepath, best_N):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0 
    num_total_examples = len(data) 
    no_answer = 0  
    
    reason_lens = []
    parsed_results = [] 
    for item in data:  
        model_dir = item["generator"]
        # Read and Parse the prediction from model output
        parsed_item = item.copy()
        if best_N == -1:
            best_N = len(item["output"])
        best_N = min(best_N, len(item["output"]))
        for i in range(best_N):
            flag_parsed_answer = True
            prediction_str = item["output"][i]
            prediction_json = extract_first_complete_json(prediction_str)
            if prediction_json is None or "answer" not in prediction_json:
                prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
            if prediction_json is None or "answer" not in prediction_json: 
                try_extracted_answer = model_specific_extraction(model_dir, prediction_str)
                if try_extracted_answer:
                    prediction_json["answer"] = try_extracted_answer
                else:
                    flag_parsed_answer = False
                    continue
            if flag_parsed_answer:
                break 
        if not flag_parsed_answer:
            no_answer += 1 
            parsed_item["model_answer"] = {"raw": None, "sanitized": None, "first_number": None}    
            parsed_item["correct_answer"] = {"raw": item["answer"].replace("#", "").strip()}
            parsed_item["matched"] = "No answer extracted"
            parsed_results.append(parsed_item)
            continue
        reason = prediction_json.get("reasoning", "")
        correct_answer = item["answer"].replace("#", "").strip()
        model_answer = str(prediction_json["answer"])
        model_answer = convert_AAAAA_to_A(model_answer)
        raw_model_answer = model_answer[:]
        model_answer = sanitize_math_answers(model_answer)
        correct_answer = sanitize_math_answers(correct_answer)
        
        first_number_in_model_answer = re.search(r"-?\d+(\.\d+)?", model_answer)
        first_number_in_correct_answer = re.search(r"-?\d+(\.\d+)?", correct_answer)
        correct = False 
        if correct_answer in model_answer:
            correct = True
        if model_answer == correct_answer:
            correct = True
        elif "livebench" not in item["dataset"]:
            if first_number_in_correct_answer and first_number_in_model_answer:
                if float(first_number_in_model_answer.group()) == float(first_number_in_correct_answer.group()):
                    correct = True
        elif is_amps_hard(item) and amps_hard_process_results_with_processed_output(correct_answer, model_answer):
            correct = True
        if correct:
            solved_examples += 1
            
        reason_lens.append(len(reason))
        parsed_item["reasoning"] = reason
        parsed_item["model_answer"] = {"raw": raw_model_answer, "sanitized": model_answer, "first_number": first_number_in_model_answer.group() if first_number_in_model_answer else None}
        parsed_item["correct_answer"] = {"raw": correct_answer, "sanitized": correct_answer, "first_number": first_number_in_correct_answer.group() if first_number_in_correct_answer else None}
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)

 
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = "First_answered"
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    result["Model"] = model_name_replacement(result["Model"])
    return result, parsed_results

def gen_results(run_name_folders, mode="eval", best_N=-1):
    model_results = load_model_results(run_name_folders)

    columns = ["Model", "Mode", "Acc", "No answer", "Total"]
    rows = []
    for model_name, filepath in model_results.items(): 
        if mode == "best":
            result, parsed_results = eval_model_best(model_name, filepath, best_N)
        elif mode == "first_answered":
            result, parsed_results = eval_model_first_answered(model_name, filepath, best_N)
        else:
            result, parsed_results = eval_model(model_name, filepath) 
        # save the parsed_results to the same filepath with a  new prefix 
        parsed_results_filepath = filepath.replace("result_dirs", "result_dirs_parsed")
        # create folders if not exist
        os.makedirs(os.path.dirname(parsed_results_filepath), exist_ok=True)
        # save 
        with open(parsed_results_filepath, "w") as f:
            json.dump(parsed_results, f, indent=2)
        rows.append(result)

    # sort the rows by puzzle accuracy
    rows = sorted(rows, key=lambda x: -float(x["Acc"]))
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] for row in rows]

    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))

    
    with open(f"result_dirs/{data_name}.summary.md", "w") as f:
        f.write(tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center"))

    # write to json file 
    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on livebench math dataset")
    parser.add_argument("--dataset", type=str, default="livebench-math", help="Dataset name")
    parser.add_argument("--mode", type=str, default="eval", help="Mode of the evaluation")
    parser.add_argument("--best_N", type=int, default=-1, help="Number of best results to consider")
    args = parser.parse_args()

    data_name = args.dataset
    mode = args.mode
 
    run_name_folders = {
        mode: f"result_dirs/{data_name}"
    } 
    if mode == "eval":
        gen_results(run_name_folders)
    elif mode == "best":
        gen_results(run_name_folders, mode="best", best_N=args.best_N)
    elif mode == "first_answered":
        gen_results(run_name_folders, mode="first_answered", best_N=args.best_N)


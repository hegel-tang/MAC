
  import os
import sys
import json
import gc
import time
import argparse
import subprocess
import shlex
from typing import List, Dict, Optional
from tqdm import tqdm

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from unified_utils import load_eval_data, save_outputs
from global_configs import HF_TEMPLATED_MODELS, IM_END_MODELS
from hf_models import DecoderOnlyModelManager
from transformers import AutoTokenizer

# --- Helper Classes ---

class LLMController:
    """
    Handles Model Loading, Unloading, and Generation (vLLM & HF).
    """
    def __init__(self, args, agent_idx, model_name):
        self.args = args
        self.agent_idx = agent_idx
        self.model_name = model_name
        self.llm = None
        self.lora_request = None
        self.engine_type = args.engine

    def _infer_maybe_lora(self, model_name):
        if os.path.exists(model_name):
            if os.path.exists(f"{model_name}/adapter_config.json"):
                return model_name, model_name 
            else:
                return model_name, None
        else:
            from huggingface_hub import hf_hub_download, snapshot_download
            try:
                adapter_config_path = hf_hub_download(repo_id=model_name, filename="adapter_config.json")
                adapter_path = snapshot_download(repo_id=model_name)
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                return adapter_config["base_model_name_or_path"], adapter_path
            except Exception:
                return model_name, None

    def load(self):
        print(f"[{self.agent_idx}] Loading model: {self.model_name} ({self.engine_type})")
        if self.engine_type == "vllm":
            from vllm import LLM
            base_model, lora_path = self._infer_maybe_lora(self.model_name)
            
            if lora_path:
                from vllm.lora.request import LoRARequest
                self.lora_request = LoRARequest(lora_path.split("/")[-1], 1, lora_path)
            
            trial_util = self.args.gpu_memory_utilization
            for attempt in range(3):
                try:
                    self.llm = LLM(
                        model=base_model,
                        tokenizer=self.model_name if self.args.tokenizer_name == "auto" else self.args.tokenizer_name,
                        tensor_parallel_size=self.args.tensor_parallel_size,
                        download_dir=self.args.download_dir,
                        dtype=self.args.dtype,
                        tokenizer_mode=self.args.tokenizer_mode,
                        max_model_len=None if self.args.max_model_len == -1 else self.args.max_model_len,
                        trust_remote_code=True,
                        gpu_memory_utilization=trial_util,
                        enable_lora=(self.lora_request is not None),
                        max_num_seqs=128,
                        enable_sleep_mode=True
                    )
                    break
                except Exception as e:
                    if "memory" in str(e).lower():
                        trial_util = max(0.2, trial_util - 0.2)
                        print(f"[{self.agent_idx}] Memory error, retrying with util={trial_util}...")
                    else:
                        raise e
        elif self.engine_type == "hf":
            self.llm = DecoderOnlyModelManager(self.model_name, self.model_name, cache_dir=self.args.download_dir,
                                               bf16=self.args.hf_bf16, gptq=self.args.hf_gptq)
            self.llm.load_model()
        return self.llm

    def unload(self):
        if self.llm is None: return
        print(f"[{self.agent_idx}] Unloading model...")
        if self.engine_type == "vllm":
            try:
                self.llm.sleep(level=2)
            except: pass
        self.llm = None
        self.lora_request = None
        import torch
        try:
            torch.cuda.empty_cache()
        except: pass
        gc.collect()
        self._wait_for_gpu_free()

    def _wait_for_gpu_free(self, required_free_gb=None, timeout=60):
        if required_free_gb is None: required_free_gb = 5.0 
        start = time.time()
        while time.time() - start < timeout:
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"])
                free_gb = int(out.decode().strip()) / 1024.0
                if free_gb >= required_free_gb: return
            except:
                time.sleep(5)
                return
            time.sleep(2)

    def get_stop_tokens(self):
        stop_words = []
        stop_token_ids = []
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.model_name in IM_END_MODELS:
                for token in ["<|im_end|>", "<|eot_id|>"]:
                    if token in hf_tokenizer.get_vocab():
                        stop_token_ids.append(hf_tokenizer.get_vocab()[token])
            if self.model_name in HF_TEMPLATED_MODELS:
                stop_token_ids.append(hf_tokenizer.eos_token_id)
        except: pass
        return stop_words, stop_token_ids

    def generate(self, inputs, n_samples, max_tokens, temperature, top_p):
        if self.llm is None: self.load()
        outputs_text = []
        
        if self.engine_type == "vllm":
            from vllm import SamplingParams
            stop_words, stop_ids = self.get_stop_tokens()
            sampling_params = SamplingParams(
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=self.args.repetition_penalty,
                max_tokens=max_tokens,
                stop=stop_words,
                stop_token_ids=stop_ids,
                n=n_samples
            )
            for cur_id in tqdm(range(0, len(inputs), self.args.batch_size), desc=f"Agent {self.agent_idx} Gen"):
                batch_inputs = inputs[cur_id : cur_id + self.args.batch_size]
                batch_res = self.llm.generate(batch_inputs, sampling_params, use_tqdm=False, lora_request=self.lora_request)
                batch_out = [[o.text for o in x.outputs] for x in batch_res]
                outputs_text.extend(batch_out)
                yield batch_out, cur_id, batch_inputs
                
        elif self.engine_type == "hf":
            gen_args = { "num_outputs": n_samples, "max_output_tokens": max_tokens, "temperature": temperature, "top_p": top_p }
            for cur_id in tqdm(range(0, len(inputs), self.args.batch_size), desc=f"Agent {self.agent_idx} Gen (HF)"):
                batch_inputs = inputs[cur_id : cur_id + self.args.batch_size]
                batch_out = self.llm.infer_generate(batch_inputs, args=gen_args)
                outputs_text.extend(batch_out)
                yield batch_out, cur_id, batch_inputs


class SelectionManager:
    """
    Handles confidence calculation and selection.
    """
    @staticmethod
    def _run_worker(args, agent_idx, input_file, model_name, output_file):
        conf_gpu_id = "0"
        print(f"[{agent_idx}] Computing confidence (Worker)...")
        cmd = [
            sys.executable,
            "MAC/compute_conf_worker.py", 
            conf_gpu_id,
            input_file,
            model_name,
            output_file,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = conf_gpu_id
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            raise RuntimeError(f"compute_conf_worker failed for Agent {agent_idx}")

    @staticmethod
    def select_best_solver_path_with_dual_lookup(args, agent_idx, evaluator_conf_path, solver_original_conf_path):
        """
        Dual Lookup Selection:
        1. Read 'evaluator_conf_path' (Agent 1 Short Gen) to find the INDEX of the best path.
           (Because Critic Evaluation is more reliable for selection).
        2. Read 'solver_original_conf_path' (Agent 0 Original Gen) to get the CONFIDENCE SCORE at that index.
           (Because we need to compare apples to apples in the final step: Raw Solver Conf vs Raw Reviser Conf).
        """
        with open(evaluator_conf_path, "r", encoding="utf-8") as f:
            eval_confs = json.load(f)
            
        with open(solver_original_conf_path, "r", encoding="utf-8") as f:
            solver_confs = json.load(f)
            
        # Structure Check:
        # Solver (Agent 0) generated `args.num_outputs` candidates per question.
        # Evaluator (Agent 1) generated `args.sample_num` samples for EACH Solver candidate.
        # So `eval_confs` should be roughly `args.sample_num` times larger (or structured differently) than `solver_confs`.
        
        # NOTE: compute_conf_worker usually returns a flat list where each entry corresponds to one generation call.
        # Agent 0 Gen: 1 input -> N outputs. Worker calculates N scores.
        # Agent 1 Gen: N inputs -> M samples each. Worker calculates N*M scores.
        
        # Let's align based on the assumption that lists are ordered by Question -> Candidate.
        
        selected_items = []
        
        # Number of Solver Candidates per Question
        n_solver_cands = args.num_outputs
        
        # We iterate through the SOLVER'S confidence list (since we want to pick 1 out of N solver cands)
        # But we loop carefully to match indices.
        
        # Pointers
        ptr_eval = 0
        
        # We assume solver_confs is a flat list of all candidates generated by Agent 0.
        # It is grouped by Question implicitly: [Q1_C1, Q1_C2... Q1_CN, Q2_C1...]
        
        # We iterate in chunks of `n_solver_cands` (one Question's worth of Solver candidates)
        for i in range(0, len(solver_confs), n_solver_cands):
            solver_chunk = solver_confs[i : i + n_solver_cands]
            if not solver_chunk: continue
            
            best_eval_score = -1.0
            best_idx_in_chunk = -1
            
            # For each candidate in this question, look at Evaluator scores
            for j in range(len(solver_chunk)):
                # The Evaluator generated `args.sample_num` samples for this specific candidate.
                # In the flat `eval_confs` list, these take up `args.sample_num` slots.
                
                # Check bounds
                if ptr_eval >= len(eval_confs): break
                
                # Extract the chunk of Evaluator scores for Solver Candidate j
                # Note: `compute_conf_worker` output usually groups samples for one prompt in `confidence_list`
                # IF Agent 1 generated M samples per prompt.
                
                eval_item = eval_confs[ptr_eval]
                confs = eval_item.get("confidence_list", [])
                
                # Calculate mean confidence of Critic's evaluation (Short Gen)
                avg_eval_conf = sum(confs) / len(confs) if confs else 0.0
                
                if avg_eval_conf > best_eval_score:
                    best_eval_score = avg_eval_conf
                    best_idx_in_chunk = j
                
                ptr_eval += 1
            
            if best_idx_in_chunk != -1:
                # SELECTION: We pick the solver candidate at `best_idx_in_chunk`
                chosen_solver_item = solver_chunk[best_idx_in_chunk]
                
                # LOOKUP: We retrieve the ORIGINAL score from Agent 0
                original_conf_list = chosen_solver_item.get("confidence_list", [])
                original_score = sum(original_conf_list)/len(original_conf_list) if original_conf_list else 0.0
                
                # Construct the selected item for next stage
                # We save the Agent 0 item (which contains the answer/model_input)
                # And we inject the `original_solver_score` for later comparison.
                chosen_solver_item["original_solver_score"] = original_score
                
                # Optional: We can also save `evaluator_score` for debugging
                chosen_solver_item["evaluator_score"] = best_eval_score
                
                selected_items.append(chosen_solver_item)

        out_path = f"{args.output_folder}/{args.data_name}/agent{agent_idx}_input_selected.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(selected_items, f, ensure_ascii=False, indent=2)
        return out_path

    @staticmethod
    def select_best_output(args, agent_idx, conf_json_path):
        """
        Scenario: Agent 1 Final Full Gen.
        Goal: Select the best OUTPUT based on its own confidence.
        """
        with open(conf_json_path, "r", encoding="utf-8") as f:
            confidence_dict = json.load(f)
            
        selected_items = []
        chunk_size = args.num_outputs 
        
        for i in range(0, len(confidence_dict), chunk_size):
            chunk = confidence_dict[i : i + chunk_size]
            if not chunk: continue
            
            best_val = -1.0
            best_idx = -1
            
            for j, rec in enumerate(chunk):
                confs = rec.get("confidence_list", [])
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                if avg_conf > best_val:
                    best_val = avg_conf
                    best_idx = j
                    
            if best_idx != -1:
                best_item = chunk[best_idx]
                best_item["final_confidence_score"] = best_val
                selected_items.append(best_item)
                
        out_path = f"{args.output_folder}/{args.data_name}/agent{agent_idx}_final_selected.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(selected_items, f, ensure_ascii=False, indent=2)
        return out_path


# --- Final Comparison Logic ---
def run_final_confidence_comparison(args, input_selection_file, reviser_file):
    """
    input_selection_file: Contains the chosen Agent 0 output (with 'original_solver_score').
    reviser_file: Contains Agent 1's Final Full Output (with 'final_confidence_score').
    """
    print("\n=== Running Scheme 2: Confidence Comparison (Solver Original vs Reviser Final) ===")
    
    with open(input_selection_file, 'r', encoding='utf-8') as f:
        solver_data = json.load(f)
    with open(reviser_file, 'r', encoding='utf-8') as f:
        reviser_data = json.load(f)
        
    solver_map = {x['session_id']: x for x in solver_data}
    
    final_results = []
    count_solver = 0
    count_reviser = 0
    
    for rev_item in reviser_data:
        sid = rev_item.get('session_id')
        if sid not in solver_map: continue
            
        sol_item = solver_map[sid]
        
        # CORRECTED LOGIC: 
        # Conf(Solver) is the Original Agent 0 Confidence (Baseline)
        conf_sol = sol_item.get("original_solver_score", 0.0)
        # Conf(Reviser) is Agent 1's Final Confidence
        conf_rev = rev_item.get("final_confidence_score", 0.0)
        
        # Formula
        if conf_rev > conf_sol + args.confidence_delta:
            final_output = rev_item.get("output") # Accept Reviser
            source = "reviser"
            count_reviser += 1
        else:
            # We revert to Solver.
            # sol_item comes from `agent0_output.json`, so its "output" field IS the Solver's answer.
            # However, `output` is a list `[text]`.
            final_output = sol_item.get("output")
            source = "solver"
            count_solver += 1
            
        final_entry = {
            "session_id": sid,
            "output": final_output, 
            "final_answer": rev_item.get("answer", ""), # GT
            "selected_source": source,
            "solver_original_conf": conf_sol,
            "reviser_final_conf": conf_rev,
            "delta": args.confidence_delta
        }
        final_results.append(final_entry)
        
    print(f"Comparison Done. Reviser Accepted: {count_reviser}, Solver Kept: {count_solver}")
    
    out_path = f"{args.output_folder}/{args.data_name}/MA_final_answer.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)


# --- Main Pipeline ---

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default="vllm", type=str)
    parser.add_argument('--output_folder', default="./result_dirs/", type=str)
    parser.add_argument('--download_dir', default=None, type=str)
    parser.add_argument('--model_name', default="/home/ubuntu/gemma-3-4b", type=str)
    parser.add_argument('--tokenizer_name', default="auto", type=str)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--dtype', type=str, default="auto")
    parser.add_argument('--tokenizer_mode', type=str, default="auto")
    parser.add_argument('--data_name', default="gsm", type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_outputs', default=8, type=int) # Agent 0 Candidates
    parser.add_argument('--sample_num', default=8, type=int)  # Agent 1 Short Gen Samples
    parser.add_argument('--top_p',default=0.9, type=float)
    parser.add_argument('--temperature',default=0.7, type=float)
    parser.add_argument('--repetition_penalty',default=1, type=float)
    parser.add_argument('--max_tokens',default=4096, type=int)
    parser.add_argument('--max_model_len',default=-1, type=int)
    parser.add_argument('--num_shards', default=1, type=int)
    parser.add_argument('--shard_id', default=0, type=int)
    parser.add_argument('--start_index',default=0, type=int) 
    parser.add_argument('--end_index',default=-1, type=int)
    parser.add_argument('--filepath',default="auto", type=str)
    parser.add_argument('--cache_filepath', default=None, type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--hf_bf16', action='store_true')
    parser.add_argument('--hf_gptq', action='store_true')
    parser.add_argument('--gpu_memory_utilization', default=0.8, type=float)
    parser.add_argument('--agent_num',default=3,type=int)
    parser.add_argument('--agent_model_names', default="", type=str)
    parser.add_argument('--unload_after_agent', action='store_true')
    parser.add_argument('--confidence_delta', default=0.05, type=float)
    return parser.parse_args()

def get_agent_model_list(args):
    if args.agent_model_names:
        candidate = [x.strip() for x in args.agent_model_names.split(",") if x.strip()]
    else:
        candidate = []
    final_list = []
    for i in range(args.agent_num):
        if i < len(candidate): final_list.append(candidate[i])
        else: final_list.append(args.model_name)
    return final_list

def determine_filepath(args, agent_idx):
    if args.filepath == "auto":
        prefix = f"agent{agent_idx}"
        return f"{args.output_folder}/{args.data_name}/{prefix}_output.json"
    else:
        base, ext = os.path.splitext(args.filepath)
        return f"{base}.agent{agent_idx}{ext}"

def main():
    args = parse_args()
    if args.download_dir == "default": args.download_dir = None
    os.makedirs(f"{args.output_folder}/{args.data_name}", exist_ok=True)
    
    agent_models = get_agent_model_list(args)
    
    input_selection_file = None # Stores Chosen Agent 0 output (with original score)
    reviser_final_file = None   # Stores Agent 1 Final output (with final score)
    solver_conf_path = None     # Path to Agent 0's confidence file
    
    for agent_idx in range(args.agent_num):
        current_model_name = agent_models[agent_idx]
        print(f"\n{'='*20} Agent {agent_idx} ({current_model_name}) {'='*20}")
        output_filepath = determine_filepath(args, agent_idx)
        controller = LLMController(args, agent_idx, current_model_name)
        
        # === Agent 0: Generate -> Compute RAW Confidence ===
        if agent_idx == 0:
            id_strs, chat_history, model_inputs, metadata = load_eval_data(args, agent_idx, selected=False, model_name=current_model_name)
            
            outputs = []
            if os.path.exists(output_filepath) and not args.overwrite:
                 with open(output_filepath, 'r') as f: existing = json.load(f)
                 outputs = [[x["output"]] if isinstance(x["output"], str) else x["output"] for x in existing]
                 print(f"Skipping {len(outputs)} existing entries.")
            
            inputs_to_run = model_inputs[len(outputs):]
            if inputs_to_run:
                gen_iter = controller.generate(inputs_to_run, args.num_outputs, args.max_tokens, args.temperature, args.top_p)
                for batch_out, _, _ in gen_iter:
                    outputs.extend(batch_out)
                    save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, output_filepath, model_name=current_model_name)
            
            # --- NEW STEP: Compute Agent 0 Original Confidence ---
            print(f"[{agent_idx}] Computing Solver Original Confidence...")
            controller.unload() # Unload before conf calc
            
            solver_conf_path = output_filepath.replace(".json", "_conf.json")
            SelectionManager._run_worker(args, agent_idx, output_filepath, current_model_name, solver_conf_path)
            
            if args.unload_after_agent: controller.unload()

        # === Agent > 0 (Reviser): Eval Agent 0 -> Select -> Full Gen ===
        else:
            # 1. Load Agent 0's RAW outputs (Evaluator Input)
            id_strs, chat_history, model_inputs, metadata = load_eval_data(args, agent_idx, selected=False, model_name=current_model_name)
            
            # 2. Short Generation (Evaluation)
            print(f"[{agent_idx}] Phase 1: Short Generation (Evaluating Agent {agent_idx-1} outputs)...")
            temp_short_gen_path = output_filepath.replace(".json", "_short_eval.json")
            
            outputs = []
            if os.path.exists(temp_short_gen_path) and not args.overwrite:
                 with open(temp_short_gen_path, 'r') as f: existing = json.load(f)
                 outputs = [[x["output"]] if isinstance(x["output"], str) else x["output"] for x in existing]
            
            inputs_to_run = model_inputs[len(outputs):]
            if inputs_to_run:
                gen_iter = controller.generate(inputs_to_run, args.sample_num, max_tokens=64, temperature=args.temperature, top_p=args.top_p)
                for batch_out, _, _ in gen_iter:
                    outputs.extend(batch_out)
                    save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, temp_short_gen_path, model_name=current_model_name)
            
            controller.unload()
            
            # 3. Compute Confidence on Short Gen
            evaluator_conf_path = temp_short_gen_path.replace(".json", "_conf.json")
            SelectionManager._run_worker(args, agent_idx, temp_short_gen_path, current_model_name, evaluator_conf_path)
            
            # 4. DUAL LOOKUP SELECTION
            # Use Evaluator Score to Select Path; Fetch Original Solver Score for Record
            print(f"[{agent_idx}] Phase 2: Dual Lookup Selection...")
            if solver_conf_path is None:
                # Fallback if Agent 0 file calculation failed or manual run separation
                solver_conf_path = output_filepath.replace(f"agent{agent_idx}", f"agent{agent_idx-1}").replace(".json", "_conf.json")

            input_selection_file = SelectionManager.select_best_solver_path_with_dual_lookup(args, agent_idx, evaluator_conf_path, solver_conf_path)
            print(f"Selected Solver inputs (with original scores) saved to: {input_selection_file}")
            
            # 5. Full Generation (Refinement)
            print(f"[{agent_idx}] Phase 3: Full Generation (Refining Selected Answer)...")
            controller.load()
            
            with open(input_selection_file, 'r', encoding='utf-8') as f:
                selected_data = json.load(f)
            
            # Extract inputs directly from the selection file
            # These inputs are essentially "Agent 0 Output + History"
            inputs_sel = [x["model_inputs"] for x in selected_data] 
            id_strs_sel = [x["session_id"] for x in selected_data]
            chat_history_sel = [x.get("chat_history", []) for x in selected_data]
            
            final_outputs = []
            if os.path.exists(output_filepath) and not args.overwrite:
                 with open(output_filepath, 'r') as f: existing = json.load(f)
                 final_outputs = [[x["output"]] if isinstance(x["output"], str) else x["output"] for x in existing]
            
            inputs_to_run_final = inputs_sel[len(final_outputs):]
            
            if inputs_to_run_final:
                # Generate 1 (or num_outputs) full answer per selected candidate
                gen_iter = controller.generate(inputs_to_run_final, args.num_outputs, args.max_tokens, args.temperature, args.top_p)
                for batch_out, cur_id, batch_in in gen_iter:
                    final_outputs.extend(batch_out)
                    curr_ids = id_strs_sel[len(final_outputs)-len(batch_out) : len(final_outputs)]
                    save_outputs(args, id_strs_sel, final_outputs, chat_history_sel, {"dataset": ["gsm"]*len(id_strs_sel)}, inputs_sel, output_filepath, model_name=current_model_name)

            controller.unload()

            # 6. Compute Final Confidence (For comparison)
            if agent_idx == args.agent_num - 1:
                print(f"[{agent_idx}] Computing Final Confidence for Reviser...")
                conf_final_path = output_filepath.replace(".json", "_final_conf.json")
                SelectionManager._run_worker(args, agent_idx, output_filepath, current_model_name, conf_final_path)
                reviser_final_file = SelectionManager.select_best_output(args, agent_idx, conf_final_path)

    # --- Final Step: Confidence Comparison ---
    if input_selection_file and reviser_final_file:
        run_final_confidence_comparison(args, input_selection_file, reviser_final_file)
    else:
        print("Comparison skipped (files missing).")

if __name__ == "__main__":
    main()


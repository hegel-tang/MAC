import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Avoid tokenizers parallelism warning when processes are forked
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import requests
from typing import List
import argparse
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import sys
from unified_utils import load_eval_data, save_outputs
from global_configs import HF_TEMPLATED_MODELS, IM_END_MODELS
from hf_models import DecoderOnlyModelManager
from transformers import AutoTokenizer
import subprocess
import shlex
import gc
import time
#from ma_confidence import compute_confidence_from_file
# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default="vllm", type=str)
    parser.add_argument('--output_folder', default="./result_dirs/", type=str)
    parser.add_argument('--download_dir', default=None, type=str)
    parser.add_argument('--model_name', default="/home/ubuntu/gemma-3-4b", type=str)
    parser.add_argument('--model_pretty_name', default=None, type=str)
    parser.add_argument('--tokenizer_name', default="auto", type=str)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--dtype', type=str, default="auto")
    parser.add_argument('--tokenizer_mode', type=str, default="auto")
    parser.add_argument('--data_name', default="gsm", type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_outputs', default=1, type=int)
    parser.add_argument('--top_p',default=0.9, type=float)
    parser.add_argument('--temperature',default=0.7, type=float)
    parser.add_argument('--repetition_penalty',default=1, type=float)
    parser.add_argument('--max_tokens',default=4096, type=int)
    parser.add_argument('--max_model_len',default=-1, type=int)
    parser.add_argument('--num_shards', default=1, type=int)
    parser.add_argument('--shard_id', default=0, type=int)
    parser.add_argument('--start_index',default=0, type=int) # 0 means from the beginning of the list
    parser.add_argument('--end_index',default=-1, type=int) # -1 means to the end of the list
    parser.add_argument('--filepath',default="auto", type=str)

    parser.add_argument('--cache_filepath', default=None, type=str)

    parser.add_argument('--follow_up_mode', default="N/A", type=str) # N/A means not a follow up
    parser.add_argument('--follow_up_file', default=None, type=str) # if you have an existing file

    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no_repeat_ngram_size', default=0, type=int)
    parser.add_argument('--hf_bf16', action='store_true')
    parser.add_argument('--hf_gptq', action='store_true')
    parser.add_argument('--gpu_memory_utilization', default=0.8, type=float)

    parser.add_argument('--use_hf_conv_template', action='store_true')
    parser.add_argument('--use_imend_stop', action='store_true')

    # only for MT-bench; not useful for other benchmarks
    # parser.add_argument('--cot', type=str, default="True")
    parser.add_argument('--run_name', type=str, default="")

    parser.add_argument('--agent_num',default=1,type=int)
    # Comma-separated list of model names/paths for each agent (length will be truncated/padded to agent_num)
    parser.add_argument('--agent_model_names', default="SmolLM3-3B", type=str,
                        help='Comma-separated model names/paths for agents. If empty, uses --model_name for all agents')
    # If set, unload vllm model from memory after an agent finishes generating
    parser.add_argument('--unload_after_agent', action='store_true', help='Unload vllm model after each agent finishes to save GPU memory')
    return parser.parse_args()

def infer_maybe_lora(model_name):
    if os.path.exists(model_name):
        if os.path.exists(f"{model_name}/adapter_config.json"):
            adapter_config_path = f"{model_name}/adapter_config.json"
            adapter_path = model_name
            lora_model = True
        else:
            lora_model = False
    else:
        # try hugging face
        from huggingface_hub import hf_hub_download, snapshot_download
        try:
            adapter_config_path = hf_hub_download(repo_id=model_name, filename="adapter_config.json")
            adapter_path = snapshot_download(repo_id=model_name)
            lora_model = True
        except Exception as e:
            lora_model = False
    if lora_model:
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name_or_path = adapter_config["base_model_name_or_path"]
        lora_model = adapter_path
    else:
        base_model_name_or_path = model_name
        lora_model = None
    return base_model_name_or_path, lora_model

def sanitize_args(args):
    if args.download_dir == "default":
        args.download_dir = None
    return args

if __name__ == "__main__":
    args = parse_args()
    args = sanitize_args(args)
    
    # make sure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    llm_list = []
    lora_requests = []  # parallel list to llm_list, store lora_request or None for each agent

    # Build agent models (currently all agents use args.model_name; if you want different models per agent,
    # adjust this section to read a list of model names)
    print("loading model(s) for each agent!")

    llm_list = []
    lora_requests = []

    # Keep args.tokenizer_name as 'auto' to allow per-agent tokenizer selection.

    if args.engine == "vllm":
        # Per-agent lazy loading: do not create all LLMs at startup to save memory.
        from vllm import LLM
        max_model_len = None if args.max_model_len == -1 else args.max_model_len

        # build agent model name list
        if args.agent_model_names:
            candidate_names = [x.strip() for x in args.agent_model_names.split(",") if x.strip()]
        else:
            candidate_names = []
        # pad/truncate to agent_num
        agent_model_names = []
        for i in range(args.agent_num):
            if i < len(candidate_names):
                agent_model_names.append(candidate_names[i])
            else:
                agent_model_names.append(args.model_name)

        # We store None in llm_list initially; will load per-agent LLM when agent runs
        for i in range(args.agent_num):
            llm_list.append(None)
            lora_requests.append(None)

        # attach helper for lazy loading
        def load_agent_llm(agent_idx):
            model_name_for_agent = agent_model_names[agent_idx]
            base_model_name_or_path, lora_model_name_or_path = infer_maybe_lora(model_name_for_agent)
            lora_req = None
            if lora_model_name_or_path:
                from vllm.lora.request import LoRARequest
                lora_req = LoRARequest(lora_model_name_or_path.split("/")[-1], 1, lora_model_name_or_path)
            # try to instantiate LLM, with simple backoff reducing gpu_memory_utilization
            trial_util = args.gpu_memory_utilization
            last_exc = None
            for attempt in range(3):
                try:
                    llm_instance = LLM(
                        model=base_model_name_or_path,
                        tokenizer=model_name_for_agent if args.tokenizer_name == "auto" else args.tokenizer_name,
                        tensor_parallel_size=args.tensor_parallel_size,
                        download_dir=args.download_dir,
                        dtype=args.dtype,
                        tokenizer_mode=args.tokenizer_mode,
                        max_model_len=max_model_len,
                        trust_remote_code=True,
                        gpu_memory_utilization=trial_util,
                        enable_lora=(lora_req is not None),
                        max_num_seqs=128,
                        enable_sleep_mode=True
                    )
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    # if memory related, try to reduce requested utilization
                    msg = str(e)
                    if "Free memory on device" in msg or "less than desired GPU memory utilization" in msg:
                        trial_util = max(0.2, trial_util - 0.2)
                        continue
                    else:
                        break
            if last_exc is not None:
                raise last_exc
            llm_list[agent_idx] = llm_instance
            lora_requests[agent_idx] = lora_req
            return llm_instance, lora_req

        # Helper to unload an agent's llm
        def unload_agent_llm(agent_idx):
            inst = llm_list[agent_idx]
            if inst is None:
                print(f"unload_agent_llm: no instance for agent {agent_idx}, nothing to unload")
                return

            print(f"unload_agent_llm: unloading agent {agent_idx} LLM instance...")
            try:
                inst.sleep(level=2)
                print(f"unload_agent_llm: called close() on agent {agent_idx} LLM")
            except Exception as e:
                print(f"unload_agent_llm: exception calling close() on agent {agent_idx}: {e}")

            # remove references
            llm_list[agent_idx] = None
            lora_requests[agent_idx] = None

            # best-effort GPU memory cleanup
            try:
                import torch
                torch.cuda.empty_cache()
                print("unload_agent_llm: torch.cuda.empty_cache() called")
            except Exception:
                print("unload_agent_llm: torch not available or cuda empty cache failed")
            gc.collect()

            # Wait for GPU memory to be freed (best-effort).
            def _get_gpu_mem_info(gpu_index=0):
                try:
                    out = subprocess.check_output([
                        "nvidia-smi",
                        "--query-gpu=memory.total,memory.free",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(gpu_index),
                    ])
                    line = out.decode().strip().splitlines()[0]
                    total_kb, free_kb = [int(x) for x in line.split(",")]
                    total_gb = total_kb / 1024.0
                    free_gb = free_kb / 1024.0
                    return total_gb, free_gb
                except Exception as e:
                    print(f"unload_agent_llm: failed to query nvidia-smi: {e}")
                    return None, None

            def _wait_for_gpu_free(required_free_gb, timeout=60, poll_interval=2, gpu_index=0):
                start = time.time()
                while time.time() - start < timeout:
                    total, free = _get_gpu_mem_info(gpu_index)
                    if free is None:
                        # cannot query, fallback to fixed sleep
                        print("unload_agent_llm: cannot query GPU free memory, falling back to sleep(20)")
                        time.sleep(20)
                        return True
                    print(f"unload_agent_llm: waiting for free GPU >= {required_free_gb:.2f}GB; current free={free:.2f}GB")
                    if free >= required_free_gb:
                        return True
                    time.sleep(poll_interval)
                return False

            total_gb, free_gb = _get_gpu_mem_info(0)
            if total_gb is not None:
                requested_util = float(getattr(args, "gpu_memory_utilization", 0.5))
                required_free = max(1.0, total_gb * requested_util)
                ok = _wait_for_gpu_free(required_free, timeout=60, poll_interval=2, gpu_index=0)
                if not ok:
                    print("unload_agent_llm: GPU did not free up in time, sleeping fallback 20s")
                    time.sleep(20)
                else:
                    print("unload_agent_llm: GPU memory freed to required level")
            else:
                # couldn't query via nvidia-smi: fallback to fixed sleep
                print("unload_agent_llm: nvidia-smi unavailable, sleeping fallback 20s")
                time.sleep(20)

    elif args.engine == "hf":
        llm = DecoderOnlyModelManager(args.model_name, args.model_name, cache_dir=args.download_dir,
                                    bf16=args.hf_bf16, gptq=args.hf_gptq)
        llm.load_model()
        for i in range(args.agent_num):
            llm_list.append(llm)
            lora_requests.append(None)
    else:
        raise ValueError(f"Unsupported engine: {args.engine}")
    print(f"Loaded {len(llm_list)} agent models.")    

    # (Per-agent stop/token logic moved into agent loop so templates and stops match each agent's model.)

    # We'll store each agent's outputs here: a list where each element is the outputs list-of-lists for that agent
    outputs_per_agent = []

    # For each agent, perform generation. Agent 0 uses original dataset; agent k>0 uses outputs_per_agent[k-1]
    for agent_idx in range(len(llm_list)):
        print(f"\n=== Running agent {agent_idx} ===")
        # Lazy-load per-agent LLM if using vllm
        if args.engine == "vllm":
            if llm_list[agent_idx] is None:
                agent_llm, agent_lora_request = load_agent_llm(agent_idx)
            else:
                agent_llm = llm_list[agent_idx]
                agent_lora_request = lora_requests[agent_idx]
        else:
            agent_llm = llm_list[agent_idx]
            agent_lora_request = lora_requests[agent_idx]
        # determine which model name this agent uses
        if args.engine == "vllm":
            model_name_for_agent = agent_model_names[agent_idx]
        else:
            model_name_for_agent = args.model_name

        # pass per-agent model_name into load_eval_data so templates (map_to_conv) use correct model
        id_strs_orig, chat_history_orig, model_inputs_orig, metadata_orig = load_eval_data(args, agent_idx, selected=False, model_name=model_name_for_agent,baseline=True)

        # token/stopping token logic per-agent
        stop_words = []
        include_stop_str_in_output = False
        stop_token_ids = []
        try:
            if model_name_for_agent in IM_END_MODELS:
                hf_tokenizer = AutoTokenizer.from_pretrained(model_name_for_agent, trust_remote_code=True)
                potential_end_tokens = ["<|im_end|>", "<|eot_id|>"]
                for potential_end_token in potential_end_tokens:
                    if potential_end_token in hf_tokenizer.get_vocab():
                        stop_token_ids += [hf_tokenizer.get_vocab()[potential_end_token]]
            if model_name_for_agent in HF_TEMPLATED_MODELS:
                hf_tokenizer = AutoTokenizer.from_pretrained(model_name_for_agent, trust_remote_code=True)
                stop_token_ids.append(hf_tokenizer.eos_token_id)
        except Exception:
            stop_token_ids = []
        
        if agent_idx == 0:
            id_strs = id_strs_orig[:]  # session ids
            chat_history = chat_history_orig[:]
            model_inputs = model_inputs_orig[:]  # prompts
            metadata = {k: v[:] for k, v in metadata_orig.items()}
        else:
            model_inputs = model_inputs_orig[:]
            
            if len(model_inputs) != len(id_strs_orig):
                min_len = min(len(model_inputs), len(id_strs_orig))
                model_inputs = model_inputs[:min_len]
                id_strs = id_strs_orig[:min_len]
                chat_history = chat_history_orig[:min_len]
                metadata = {k: v[:min_len] for k, v in metadata_orig.items()}
            else:
                id_strs = id_strs_orig[:]
                chat_history = chat_history_orig[:]
                metadata = {k: v[:] for k, v in metadata_orig.items()}
        
        # decide start_index and end_index by num_shards and shard_id (same logic as before)
        if args.num_shards > 1:
            num_data = len(id_strs)
            shard_size = num_data // args.num_shards
            start_index = args.shard_id * shard_size
            end_index = (args.shard_id + 1) * shard_size
            if args.shard_id == args.num_shards - 1:
                end_index = num_data
        else:
            start_index = args.start_index
            end_index = args.end_index

        # Decide the output filepath for this agent
        if args.filepath == "auto":
            if end_index == -1 and start_index == 0:
                filepath = f"{args.output_folder}/{args.data_name}/agent{agent_idx}_baseline_output.json" if agent_idx > 0 else f"{args.output_folder}/{args.data_name}/agent0_baseline_output.json"
            else:
                filepath = f"{args.output_folder}/{args.data_name}/agent{agent_idx}.{start_index}-{end_index}_output.json" if agent_idx > 0 else f"{args.output_folder}/{args.data_name}/agent0.{start_index}-{end_index}_output.json"
        else:
            # if explicit filepath given, append agent suffix to avoid overwrite
            base, ext = os.path.splitext(args.filepath)
            filepath = f"{base}.agent{agent_idx}{ext}"

            output_folder = "/".join(filepath.split("/")[:-1])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

        # Clip indices and slice inputs
        if end_index < 0 or end_index > len(model_inputs):
            end_index = len(model_inputs)
        model_inputs = model_inputs[start_index:end_index]
        id_strs = id_strs[start_index:end_index]
        chat_history = chat_history[start_index:end_index]
        metadata = {key: metadata[key][start_index:end_index] for key in metadata}

        print(f"Agent {agent_idx} will run on indices [{start_index}:{end_index}] -> {len(model_inputs)} items")
        print(f"Agent {agent_idx} output filepath: {filepath}")

        # Load existing outputs for this agent if present and not overwrite
        outputs = []
        if os.path.exists(filepath) and not args.overwrite:
            with open(filepath, "r", encoding="utf-8") as f:
                formatted_outputs = json.load(f)
            for output_item in formatted_outputs:
                outputs.append([output_item["output"]] if type(output_item["output"]) == str else output_item["output"])
            num_skipped = len(outputs)
            print(f"Agent {agent_idx}: found existing file, skipped first {num_skipped} examples")
        else:
            num_skipped = 0

        # Load cache file if provided (same as before)
        cache_outputs = {}
        if args.cache_filepath is not None:
            if os.path.exists(args.cache_filepath):
                with open(args.cache_filepath, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                for output_item in cache_data:
                    if type(output_item.get("output")) == list and len(output_item["output"]) > 0 and len(output_item["output"][0]) > 0:
                        cache_outputs[output_item["session_id"]] = output_item
            print(f"Agent {agent_idx}: Loaded {len(cache_outputs)} non-empty outputs from cache: {args.cache_filepath}")

        # Prepare generation loop for this agent
        todo_inputs = model_inputs[num_skipped:]
        if len(todo_inputs) == 0:
            print(f"Agent {agent_idx}: no new inputs to process.")
            # still append the existing outputs (maybe empty) to outputs_per_agent
            outputs_per_agent.append(outputs)
            try:
                print(f"Agent {agent_idx}: unloading agent LLM to free GPU memory (unload_after_agent=True)")
                unload_agent_llm(agent_idx)
            except Exception as e:
                print(f"Agent {agent_idx}: unload_agent_llm raised exception: {e}")
            
            continue

        # generation
        if args.engine == "vllm":
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_tokens=args.max_tokens,
                stop=stop_words,
                stop_token_ids=stop_token_ids,
                include_stop_str_in_output=include_stop_str_in_output,
                n=args.num_outputs
            )

            # generate in batches
            for cur_id in tqdm(range(0, len(todo_inputs), args.batch_size), desc=f"Agent {agent_idx} generating"):
                batch_inputs = todo_inputs[cur_id:cur_id+args.batch_size]
                # For vllm, pass lora_request if present
                batch_outputs = agent_llm.generate(batch_inputs, sampling_params, use_tqdm=False, lora_request=agent_lora_request)
                # each x in batch_outputs corresponds to an input; extract x.outputs -> list of generated objects; use .text
                outputs.extend([[o.text for o in x.outputs] for x in batch_outputs])
                # save incremental results for safety
                save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath, model_name=model_name_for_agent)

            # final save
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath, model_name=model_name_for_agent)
            

        elif args.engine == "hf":
            # A generic HF generation wrapper — adapt if your DecoderOnlyModelManager API differs.
            # We assume llm.generate returns a list matching batch_inputs, where each item is a list of generated strings.
            for cur_id in tqdm(range(0, len(todo_inputs), args.batch_size), desc=f"Agent {agent_idx} generating (hf)"):
                batch_inputs = todo_inputs[cur_id:cur_id+args.batch_size]
                gen_args = {
                    "num_outputs": args.num_outputs,
                    "max_output_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,    
                }

                batch_outputs = agent_llm.infer_generate(batch_inputs, args=gen_args)
                outputs.extend(batch_outputs)
                
                save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath, model_name=model_name_for_agent)

            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath, model_name=model_name_for_agent)

        else:
            raise ValueError(f"Unsupported engine: {args.engine}")


        print(f"Agent {agent_idx} finished. Generated {len(outputs)} items.")

        # Optionally unload vllm model instance for this agent to free GPU memory
        if args.engine == "vllm":
        
            try:
                print(f"Agent {agent_idx}: unloading agent LLM to free GPU memory (unload_after_agent=True)")
                unload_agent_llm(agent_idx)
            except Exception as e:
                print(f"Agent {agent_idx}: unload_agent_llm raised exception: {e}")
            

    print("\nAll agents finished. Outputs saved per agent in:", args.output_folder)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
#from ma_confidence import compute_confidence_from_file
# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default="vllm", type=str)
    parser.add_argument('--output_folder', default="./result_dirs/gsm/", type=str)
    parser.add_argument('--download_dir', default=None, type=str)
    parser.add_argument('--model_name', default="/home/ubuntu/gemma-3-4b", type=str)
    parser.add_argument('--model_pretty_name', default=None, type=str)
    parser.add_argument('--tokenizer_name', default="auto", type=str)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--dtype', type=str, default="auto")
    parser.add_argument('--tokenizer_mode', type=str, default="auto")
    parser.add_argument('--data_name', default="gsm", type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_outputs', default=3, type=int)
    parser.add_argument('--top_p',default=0.9, type=float)
    parser.add_argument('--temperature',default=0.7, type=float)
    parser.add_argument('--repetition_penalty',default=1, type=float)
    parser.add_argument('--max_tokens',default=7500, type=int)
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
    parser.add_argument('--gpu_memory_utilization', default=0.7, type=float)

    parser.add_argument('--use_hf_conv_template', action='store_true')
    parser.add_argument('--use_imend_stop', action='store_true')

    # only for MT-bench; not useful for other benchmarks
    # parser.add_argument('--cot', type=str, default="True")
    parser.add_argument('--run_name', type=str, default="")

    parser.add_argument('--agent_num',default=3,type=int)
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
    # for i in range(args.agent_num):
    #     if args.tokenizer_name == "auto":
    #         args.tokenizer_name = args.model_name
    #     if args.engine == "vllm":
    #         from vllm import LLM, SamplingParams
    #         max_model_len = None if args.max_model_len == -1 else args.max_model_len
    #         base_model_name_or_path, lora_model_name_or_path = infer_maybe_lora(args.model_name)
    #         if lora_model_name_or_path:
    #             from vllm.lora.request import LoRARequest
    #             lora_request = LoRARequest(lora_model_name_or_path.split("/")[-1], 1, lora_model_name_or_path)
    #         else:
    #             lora_request = None
    #         llm = LLM(model=base_model_name_or_path, tokenizer=args.tokenizer_name, tensor_parallel_size=args.tensor_parallel_size,
    #                         download_dir=args.download_dir, dtype=args.dtype, tokenizer_mode=args.tokenizer_mode,
    #                         max_model_len=max_model_len, trust_remote_code=True,
    #                         gpu_memory_utilization=args.gpu_memory_utilization,
    #                         enable_lora=(lora_request is not None)
    #                         )
    #         llm_list.append(llm)
    #         lora_requests.append(lora_request)
    #     elif args.engine == "hf":
    #         # note: DecoderOnlyModelManager 的 generate API may be different — adapt if necessary.
    #         llm = DecoderOnlyModelManager(args.model_name, args.model_name, cache_dir=args.download_dir,
    #                                     bf16=args.hf_bf16, gptq=args.hf_gptq)
    #         llm.load_model()
    #         llm_list.append(llm)
    #         lora_requests.append(None)
    #     else:
    #         raise ValueError(f"Unsupported engine: {args.engine}")
    llm_list = []
    lora_requests = []

    if args.tokenizer_name == "auto":
        args.tokenizer_name = args.model_name

    if args.engine == "vllm":
        # only create one vllm instance and one LoRA request (if applicable), then reuse
        from vllm import LLM
        max_model_len = None if args.max_model_len == -1 else args.max_model_len

        base_model_name_or_path, lora_model_name_or_path = infer_maybe_lora(args.model_name)

        if lora_model_name_or_path:
            from vllm.lora.request import LoRARequest
            shared_lora_request = LoRARequest(lora_model_name_or_path.split("/")[-1], 1, lora_model_name_or_path)
        else:
            shared_lora_request = None

        # create single shared LLM instance
        shared_llm = LLM(
            model=base_model_name_or_path,
            tokenizer=args.tokenizer_name,
            tensor_parallel_size=args.tensor_parallel_size,
            download_dir=args.download_dir,
            dtype=args.dtype,
            tokenizer_mode=args.tokenizer_mode,
            max_model_len=max_model_len,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=(shared_lora_request is not None),
            max_num_seqs=128
        )

        # reuse the same instance/reference for each agent
        for i in range(args.agent_num):
            llm_list.append(shared_llm)
            lora_requests.append(shared_lora_request)

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

    # token/stopping token logic (done once; assumes model_name same for all agents)
    stop_words = []
    include_stop_str_in_output = False
    stop_token_ids = []
    if args.model_name in IM_END_MODELS:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        potential_end_tokens = ["<|im_end|>" , "<|eot_id|>"]
        for potential_end_token in potential_end_tokens:
            if potential_end_token in hf_tokenizer.get_vocab():
                stop_token_ids += [hf_tokenizer.get_vocab()[potential_end_token]]
    if args.model_name in HF_TEMPLATED_MODELS:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        stop_token_ids.append(hf_tokenizer.eos_token_id)

    # We'll store each agent's outputs here: a list where each element is the outputs list-of-lists for that agent
    outputs_per_agent = []

    # For each agent, perform generation. Agent 0 uses original dataset; agent k>0 uses outputs_per_agent[k-1]
    for agent_idx in range(len(llm_list)):
        print(f"\n=== Running agent {agent_idx} ===")
        agent_llm = llm_list[agent_idx]
        agent_lora_request = lora_requests[agent_idx]
        id_strs_orig, chat_history_orig, model_inputs_orig, metadata_orig = load_eval_data(args,agent_idx)
        
        if agent_idx == 0:
            id_strs = id_strs_orig[:]  # session ids
            chat_history = chat_history_orig[:]
            model_inputs = model_inputs_orig[:]  # prompts
            metadata = {k: v[:] for k, v in metadata_orig.items()}
        else:
            
            model_inputs = model_inputs_orig[:]
            
            if len(model_inputs) != len(id_strs_orig):
                # adjust id_strs/chat_history/metadata to match new length if needed
                # Here we will align to min length and trim everything to that
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
            if "/" in args.model_name and args.model_pretty_name is None:
                pretty_base = args.model_name.split("/")[-1]
            else:
                pretty_base = args.model_pretty_name or (args.model_name.split("/")[-1] if args.model_name else "model")
            # file per agent
            if end_index == -1 and start_index == 0:
                filepath = f"{args.output_folder}/agent{agent_idx}_output.json"
            else:
                filepath = f"{args.output_folder}/agent{agent_idx}.{start_index}-{end_index}_output.json"
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
                save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

            # final save
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

            if agent_idx > 0:
                conf_gpu_id = "1"  
                tmp_conf_out = f"{args.output_folder}/agent{agent_idx}_conf.json"

                python_exe = shlex.quote(sys.executable) if 'shlex' in globals() else sys.executable

                cmd = [
                    python_exe,
                    "MAC/compute_conf_worker.py",
                    conf_gpu_id,
                    filepath,
                    args.model_name,
                    tmp_conf_out,
                ]

                env = os.environ.copy()
                if conf_gpu_id.strip() == "":
                    env["CUDA_VISIBLE_DEVICES"] = ""
                else:
                    env["CUDA_VISIBLE_DEVICES"] = str(conf_gpu_id)

                ret = subprocess.run(cmd, env=env)
                if ret.returncode != 0:
                    raise RuntimeError(f"compute_conf_worker failed (returncode={ret.returncode})")

                with open(tmp_conf_out, "r", encoding="utf-8") as f:
                    confidence_dict = json.load(f)
                # ----- END: separate-process confidence calc -----

                ind_list = []
                group_best_val = -1.0     
                group_best_idx = None

                for i, rec in enumerate(confidence_dict):
                    confs = rec["confidence_list"]
                    avg = sum(confs) / len(confs) if len(confs) > 0 else 0.0

                    if avg > group_best_val:
                        group_best_val = avg
                        group_best_idx = i

                    if i % args.num_outputs == args.num_outputs - 1:
                        ind_list.append(group_best_idx)
                        group_best_val = -1.0
                        group_best_idx = None
                selected_items = []
                for idx in ind_list:
                    if idx is None:
                        continue
                    if 0 <= idx < len(confidence_dict):
                        selected_items.append(confidence_dict[idx])

                selected_output_file = f"{args.output_folder}/agent{agent_idx+1}_input.json"

                os.makedirs(os.path.dirname(selected_output_file) or ".", exist_ok=True)

                with open(selected_output_file, "w", encoding="utf-8") as f:
                    json.dump(selected_items, f, ensure_ascii=False, indent=2)

        # elif args.engine == "hf":
        #     # A generic HF generation wrapper — adapt if your DecoderOnlyModelManager API differs.
        #     # We assume llm.generate returns a list matching batch_inputs, where each item is a list of generated strings.
        #     for cur_id in tqdm(range(0, len(todo_inputs), args.batch_size), desc=f"Agent {agent_idx} generating (hf)"):
        #         batch_inputs = todo_inputs[cur_id:cur_id+args.batch_size]
        #         try:
        #             # try a common API
        #             batch_results = agent_llm.generate(batch_inputs,
        #                                                num_return_sequences=args.num_outputs,
        #                                                max_new_tokens=args.max_tokens,
        #                                                temperature=args.temperature,
        #                                                top_p=args.top_p)
        #             # expect batch_results to be list-like; convert to list-of-lists of strings
        #             for res in batch_results:
        #                 if isinstance(res, list):
        #                     outputs.append(res)
        #                 elif isinstance(res, str):
        #                     outputs.append([res])
        #                 else:
        #                     # fallback: stringify
        #                     outputs.append([str(res)])
        #         except Exception as e:
        #             # If generate API not supported, fall back to calling a simple .predict or .batch_decode as available
        #             print(f"Agent {agent_idx} HF generation failed with error: {e}")
        #             raise

        #         save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

        #     save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

        # else:
        #     raise ValueError(f"Unsupported engine: {args.engine}")

        # done for agent -> append outputs to outputs_per_agent
        print(f"Agent {agent_idx} finished. Generated {len(outputs)} items.")

    print("\nAll agents finished. Outputs saved per agent in:", args.output_folder)
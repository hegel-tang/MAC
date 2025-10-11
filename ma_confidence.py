import json
import os
import tempfile
from typing import List, Dict, Optional, Union


import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def confidence_logprob_sum(logprob_sum: torch.Tensor, attention_mask: torch.Tensor, V: int):
    """
    Compute the per-sequence confidence from a tensor of per-position sums of log-probabilities.

    This implements the KL(U || p) based metric used in the script:
      conf_pos = -1/V * sum_j log p_j - log(V)
    Then the per-sequence confidence is the mean of conf_pos over valid positions.

    Args:
        logprob_sum: Tensor of shape (batch, seq_len) containing for each position the
                     sum over the vocabulary of log p(j) (i.e. sum_j log p_j).
        attention_mask: Tensor of shape (batch, seq_len) (0/1) marking valid tokens.
        V: vocabulary size.

    Returns:
        A python list with one float per sequence in the batch.
    """
    logprob_sum = logprob_sum.contiguous()
    attention_mask = attention_mask.contiguous()
    V_tensor = torch.tensor(V, dtype=logprob_sum.dtype, device=logprob_sum.device)
    conf = -1.0 / V * logprob_sum - torch.log(V_tensor)
    valid_conf = conf * attention_mask
    # Avoid division by zero; if a sequence has zero valid tokens, return nan for that sequence
    denom = attention_mask.sum(dim=-1)
    denom = denom.to(torch.float32)
    # mask denom zeros
    denom_zero = denom == 0
    denom = denom + denom_zero.to(denom.dtype)  # avoid div0
    batch_confidence_list = (valid_conf.sum(dim=-1) / denom).tolist()
    # set nan for those with zero denom
    for i, z in enumerate(denom_zero.tolist()):
        if z:
            batch_confidence_list[i] = float('nan')
    return batch_confidence_list


def _load_examples(filepath: str, output_field_name: str = "output") -> List[Dict]:
    """Load examples from .json or .parquet into a list of dicts."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        with open(filepath, "r") as f:
            return json.load(f)

    elif ext == ".parquet":
        df = pd.read_parquet(filepath)
        if output_field_name in df.columns and df[output_field_name].dtype == object and isinstance(df.iloc[0][output_field_name], str):
            df[output_field_name] = df[output_field_name].apply(json.loads)
        return df.to_dict(orient="records")

    else:
        raise ValueError(f"Unsupported input format: {ext}")


@torch.no_grad()
def compute_confidence_from_examples(
    examples: List[Dict],
    model_dir: str,
    tokenizer: Optional[AutoTokenizer] = None,
    llm: Optional[AutoModelForCausalLM] = None,
    batch_size: int = 4,
    input_field_name: str = "model_input",
    output_field_name: str = "output",
    prefix_tokens: Optional[int] = 64,
    device: Optional[Union[str, torch.device]] = None,
    resume_from_output: Optional[str] = None,
) -> List[Dict]:
    """
    Compute confidence scores for `examples` using the KL(U || p) metric, only over the
    first `prefix_tokens` of each output (after tokenization).

    Args:
        examples: list of dicts, each dict must contain the `input_field_name` (prompt)
                  and `output_field_name` (list of output strings).
        model_dir: path to model/tokenizer (HuggingFace-compatible directory/name).
        tokenizer: optional tokenizer (if provided, will not be loaded from model_dir).
        llm: optional model instance (if provided, will not be loaded from model_dir).
        batch_size: base batch size for processing outputs.
        input_field_name: field name containing the prompt string.
        output_field_name: field name containing the list of output strings.
        prefix_tokens: number of tokens from the start of the output to use; if None or <=0,
                       use the entire tokenized output.
        device: device string or torch.device; if None use 'cuda' if available else 'cpu'.
        resume_from_output: optional path to an output file to load previously computed results;
                            if provided, the function will skip already processed items and
                            append new results to the returned list.

    Returns:
        A list of dicts: each input dict extended with keys `confidence_list` (list of floats)
        and `processed_index` (int) in the same order as `examples`.
    """

    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Load model/tokenizer if not provided
    if llm is None:
        llm = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, padding=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        llm.config.pad_token_id = tokenizer.pad_token_id
        llm.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "right"
    llm.eval()

    # Prepare output list and resume if requested
    if resume_from_output and os.path.exists(resume_from_output):
        with open(resume_from_output, "r") as f:
            try:
                to_write = json.load(f)
            except json.JSONDecodeError:
                to_write = []
    else:
        to_write = []

    total_items = len(examples)
    processed_items = len(to_write)

    # Group thresholds (texts measured by raw length) — check large first
    for index in range(processed_items, total_items):
        item = examples[index]
        new_item = {k: v for k, v in item.items()}

        # Encode prompt
        input_encoded = tokenizer(
            item[input_field_name], return_tensors="pt", padding=False, truncation=True, add_special_tokens=False
        )
        input_ids = input_encoded["input_ids"].reshape(-1)
        input_attention_mask = input_encoded["attention_mask"].reshape(-1)
        input_length = int(input_attention_mask.sum().item())

        outputs = item[output_field_name]

        # Classify outputs by raw length (fix ordering)
        groups = {"small": {"outputs": [], "indices": []}, "medium": {"outputs": [], "indices": []}, "large": {"outputs": [], "indices": []}}
        for idx, text in enumerate(outputs):
            if len(text) > 6 * 1024:
                groups["large"]["outputs"].append(text)
                groups["large"]["indices"].append(idx)
            elif len(text) > 3 * 1024:
                groups["medium"]["outputs"].append(text)
                groups["medium"]["indices"].append(idx)
            else:
                groups["small"]["outputs"].append(text)
                groups["small"]["indices"].append(idx)

        final_confidences = [None] * len(outputs)
        group_batch_sizes = {"small": batch_size, "medium": max(1, batch_size // 2), "large": max(1, batch_size // 4)}

        for group_name in ["small", "medium", "large"]:
            group_texts = groups[group_name]["outputs"]
            group_indices = groups[group_name]["indices"]
            if not group_texts:
                continue

            current_batch_size = group_batch_sizes[group_name]

            group_tokenized = tokenizer(group_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
            group_outputs_ids = group_tokenized["input_ids"]
            group_outputs_attention_mask = group_tokenized["attention_mask"]

            # Build full sequences
            full_ids_list = []
            full_attention_mask_list = []
            for i in range(group_outputs_ids.size(0)):
                combined_ids = torch.cat((input_ids, group_outputs_ids[i]), dim=0)
                combined_attention_mask = torch.cat((input_attention_mask, group_outputs_attention_mask[i]), dim=0)
                full_ids_list.append(combined_ids)
                full_attention_mask_list.append(combined_attention_mask)

            full_ids = torch.stack(full_ids_list)
            full_attention_mask = torch.stack(full_attention_mask_list)

            group_confidences = []
            num_batches = (full_ids.shape[0] + current_batch_size - 1) // current_batch_size

            for batch_idx in range(num_batches):
                torch.cuda.empty_cache()
                start_idx = batch_idx * current_batch_size
                end_idx = min((batch_idx + 1) * current_batch_size, full_ids.shape[0])
                batch_ids = full_ids[start_idx:end_idx].to(device)
                batch_attention_mask = full_attention_mask[start_idx:end_idx].to(device)

                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
                    logits = llm(batch_ids, attention_mask=batch_attention_mask).logits
                    # Keep only output part
                    logits = logits[:, input_length:, :]

                    # Only use prefix_tokens of the output
                    if prefix_tokens is not None and prefix_tokens > 0:
                        logits = logits[:, :prefix_tokens, :]

                    log_probs = F.log_softmax(logits, dim=-1)
                    # sum over vocab to get the same intermediate quantity as the KL-based metric
                    logprob_sum = log_probs.sum(dim=-1).to("cpu").to(torch.float32)

                # Align output attention mask and move to CPU for the CPU-based metric
                batch_output_attention_mask = group_outputs_attention_mask[start_idx:end_idx]
                if prefix_tokens is not None and prefix_tokens > 0:
                    batch_output_attention_mask = batch_output_attention_mask[:, :prefix_tokens]

                batch_confidence_list = confidence_logprob_sum(logprob_sum, batch_output_attention_mask, llm.config.vocab_size)
                group_confidences.extend(batch_confidence_list)

            # Put back into final list
            for i, orig_idx in enumerate(group_indices):
                final_confidences[orig_idx] = group_confidences[i]

        if any(conf is None for conf in final_confidences):
            # keep going but warn
            print(f"Warning: Some confidences were not computed for item at index {index}.")

        new_item["confidence_list"] = final_confidences
        new_item["processed_index"] = index
        to_write.append(new_item)

        # If resume path given, write to it incrementally to allow resuming
        if resume_from_output:
            try:
                os.makedirs(os.path.dirname(resume_from_output), exist_ok=True)
                with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(resume_from_output)) as tmp_file:
                    json.dump(to_write, tmp_file, indent=4, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
                    temp_name = tmp_file.name
                os.replace(temp_name, resume_from_output)
            except Exception as e:
                print(f"Error writing to resume file: {e}")
                # continue but do not crash

    return to_write


def compute_confidence_from_file(
    filepath: str,
    model_dir: str,
    batch_size: int = 4,
    input_field_name: str = "model_input",
    output_field_name: str = "output",
    prefix_tokens: Optional[int] = 64,
    resume_output_file: Optional[str] = None,
) -> List[Dict]:
    """
    Convenience wrapper that loads examples from a file and calls compute_confidence_from_examples.

    Returns the same list of dicts extended with confidence information.
    """
    examples = _load_examples(filepath, output_field_name)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("  总显存:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
    print("  已分配显存:", round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GB")
    return compute_confidence_from_examples(
        examples=examples,
        model_dir=model_dir,
        batch_size=batch_size,
        input_field_name=input_field_name,
        output_field_name=output_field_name,
        prefix_tokens=prefix_tokens,
        resume_from_output=resume_output_file,
    )

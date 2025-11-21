Parser_prompt="""
TASK:
You are Parser. Input: {question}.

OUTPUT:
Return JSON only, with the following schema:
{
  "known": [
    {"symbol": <string>, "value": <number>, "unit": <string|null>, "note": <string|null>}
  ],
  "unknown": [
    {"name": <string>, "note": <string|null>}
  ],
  "explicit_conditions": [...],
}

RULES:
1) Do NOT emit any non-JSON text or commentary — only the JSON described above.
2) Extract all explicit numeric values, units, clearly stated conditions, and the problem goal.
3) Keep each field concise and machine-parseable.

FEW-SHOT EXAMPLE:
Input (user message): {"raw_text":"A container holds 12 apples. Pack 7 such containers. How many apples in total?"}
Expected Output:
{
  "known": [
    {"symbol": "n_per_box", "value": 12, "unit": "apples", "note": "apples per container"},
    {"symbol": "boxes", "value": 7, "unit": "containers", "note": "number of containers"}
  ],
  "unknown": [
    {"name": "total_apples", "note": "total number of apples"}
  ],
  "explicit_conditions": ["each container holds the same number of apples"],
}
"""

EqBuilder_prompt = """
TASK:
You are Equation-Builder. Input: the Parser output JSON {Parser_output}.

OUTPUT:
Return JSON only, with the following schema:
{
  "equations": [
    {"eq": <string>, "unknown":[<string>,...], "known": [ <string> : <int>,...], "derivation": <string>}
  ],
  "model_priority": [<int_index_order>],
}

RULES:
1) Only return the JSON above — no extra natural-language text.
2) Translate the Parser's extracted facts into one or more candidate mathematical models (equations or arithmetic expressions).
3) If multiple modeling choices exist, enumerate all candidate models and set "model_priority" to a list of indices indicating preferred order (e.g., [0,1] means equations[0] is highest priority).
4) Do NOT introduce new numeric values that contradict Parser output.
5) Keep "derivation" to a single short sentence describing how the equation was formed.

FEW-SHOT EXAMPLE:
Input:
{
  "known": [
    {"symbol": "n_per_box", "value": 12, "unit": "apples", "note": "apples per container"},
    {"symbol": "boxes", "value": 7, "unit": "containers", "note": "number of containers"}
  ],
  "unknown": [
    {"name": "total_apples", "note": "total number of apples"}
  ],
  "explicit_conditions": ["each container holds the same number of apples"],
}
Expected Output:
{
  "equations": [
    {"eq": "total_apples = n_per_box * boxes = 12 * 7 ", "unknown":["total_apples"], "known": [ "n_per_box" : 12, "boxes" : 7], "derivation": "Multiply apples per container by number of containers"}
  ],
  "model_priority": [0],
}

"""

Calculator_prompt="""
TASK:
You are Calculator. Input: Equation-Builder's "equations" list {Eqbuilder_output}.

OUTPUT:
Return JSON only, with the following schema:
{
  "answer": <number|null>,
  "numeric_steps": [<string>, ...],
}

RULES:
1) Only perform arithmetic/equation evaluation; do not add new assumptions or narrative.
2) Return reproducible numeric steps — each step is an explicit calculation string.
3) If evaluation succeeds, set "answer" to the numeric result. The answer is usually the result of the last step of calculation.
4) Do NOT output any non-JSON text.

FEW-SHOT EXAMPLE:
Input: 
{
  "equations": [
    {"eq": "total_apples = n_per_box * boxes = 12 * 7", "vars": ["total_apples", "n_per_box", "boxes"], "derivation": "Multiply apples per container by number of containers"}
  ],
  "model_priority": [0],
}
Expected Output:
{
  "answer": 84,
  "numeric_steps": ["12 * 7 = 84"],
}

"""

# Solver_prompt = """

# ## Question: 

# {question}


# ## Instruction 

# Please solve this question directly by providing your answer.
# Please show your final answer in the `answer` field. without explanation in the following json format. 

# ```json
# {
#     "answer": "___"
# }
# ```
# """

# Solver_prompt = """
# You are SOLVER. Goal: answer the question <{question}> by reasoning first and then giving the final numeric answer.

# Strict rules (MUST follow):
# 1. Output MUST be a single valid JSON (UTF-8) with exactly these fields (no extra fields, no comments, no surrounding text):
# {
#     "reasoning": "1) ...\n2) ...\n... (numbered steps)",
#     "final_check": "digit-by-digit recomputation or verification lines (if any arithmetic).",
#     "self_check": "one-line self-check summary (yes/no + short note).",
#     "answer": "___" // final answer as a pure numeric string (integer or decimal), e.g. "42" or "3.14"
# }
# 2. reasoning MUST be numbered steps (format: "1) ", "2) ", ...). Each step should be short and explicit (no more than 2 lines per step).
# 3. If any arithmetic is performed, you MUST include a digit-by-digit recomputation in final_check. Example formats:
#    - Addition: "123 + 456 = 579 (digits: 3+6=9, 2+5=7, 1+4=5)"
#    - Multiplication: "12 * 34 = 408 (steps: 12*4=48 -> write 8 carry 4; 12*3=36 + carry4 = 40 -> 408)"
# 4. self_check must be a one-line confirmation, e.g., "yes — verified digit-by-digit" or "no — uncertain about rounding".
# 5. Use only standard ASCII double quotes for JSON. Do not output any extra text outside the JSON.
# 6. If the question has no numeric answer or cannot be answered as a number, set "answer" to the string "NaN" and explain briefly in reasoning.
# """
Solver_prompt =  """
You are SOLVER. 

Problem: <{question}>

Please answer this question by first reasoning and then providing your answer.
Present your reasoning and solution in the following json format. 
Please show your final answer in the `answer` field, e.g.,`"answer": "42"`.
The answer must be a pure number - either an integer or decimal number.
Output JSON ONLY in this exact format:

```json
{
    "reasoning": "___",
    "answer": "___"
}
```

"""

# Critic_prompt = """You are CRITIC.

# Input: the original question <{question}> and the Solver's JSON output <{Solver_output}>.

# Task:
# 1) Check the Solver's "reasoning" for logical, arithmetic, or formatting errors, and check that numeric calculations are correct (recompute arithmetic).
# 2) For each issue found, output an item with (error_type, explanation, suggested_fix, step_reference). Use the provided error_type taxonomy below.
# 3) Provide an overall confidence score (float 0.0–1.0) representing how confident you are that the final answer is correct.
# 4) Set `verdict` to `"accept"` if issues is empty (i.e., solver output is correct and well-formed), else `"revise"`.
# 5) If no issues, set "issues" to an empty array.

# Error type taxonomy (use one of these):
# - "arithmetic"
# - "logic"
# - "missing_step"
# - "ambiguous"
# - "invalid_json"
# - "malformed_output"


# Output JSON ONLY in this exact format:
# {
#   "issues": [
#     {"error_type": "arithmetic|logic|missing_step|ambiguous|invalid_json|malformed_output",
#      "explanation": "...",
#      "suggested_fix": "...",
#      "step_reference": "step 2" // or null if not applicable
#     }
#   ],
#   "confidence": 0.85,
#   "confidence_explanation": "one-sentence justification for the numeric confidence",
#   "verdict": "accept" | "revise"
# }

# Notes:
# - If there are no issues, set "issues" to an empty array, "confidence" to a number, and "verdict" to "accept".
# - Be specific in explanations (point to the step number where the problem occurs).
# - Do not include any extra fields."""

# Critic_prompt ="""
# You are CRITIC.
# Input: the original question <{question}> and the Solver's JSON output <{Solver_output}>.
# Task:
# 1) Check the Solver's "reasoning" for logical, arithmetic, or formatting errors, and check that numeric calculations are correct (recompute arithmetic).
# 2) For each issue found, output an item with (error_type, explanation, suggested_fix).
# 3) If no issues, set "null" to "error_type".

# Output valid JSON ONLY in this format:

# {
#   "issues": [
#     {"error_type": "arithmetic|logic|missing_step|ambiguous|null", "explanation": "...", "suggested_fix": "..."}
#   ],
#   "confidence": 0.85,
#   "verdict": "accept" | "revise"
# }
# """

Critic_prompt = """
You are CRITIC. Input: the original question <{question}> and SOLVER's JSON output <{Solver_output}>. Task: rigorously check the solver output and return a JSON-only report.

Strict procedure (MUST execute):
1. First validate that SOLVER output is valid JSON and follows this schema: contains exactly reasoning, final_check, self_check, answer fields. answer must be a pure numeric string or "NaN". If schema validation fails, return a single-item issues array with error_type "invalid_json" and provide the minimal offending snippet (do not rewrite solver text).
2. If schema is valid, perform these checks in order:
   a) arithmetic: Recompute any arithmetic shown in final_check digit-by-digit. If recomputed value != answer, report an "arithmetic" issue and include the original value, the recomputation steps, and the corrected value. Your explanation MUST include reproducible digit-by-digit details (e.g., per-digit sums/carries).
   b) logic: If the reasoning contains a clear logical error (conclusion contradicts arithmetic or steps are self-contradictory), report "logic", quote the exact step text, and provide a one-line correction or counterexample.
   c) missing_step: Use only if a required claim cannot be located in the solver's numbered steps (e.g., missing assumption). First attempt fuzzy matching; only if matching fails report missing_step and suggest exactly what numbered step or text to add.
   d) ambiguous: If a step is too vague to reproduce (e.g., "rounded" but no rounding rule), report "ambiguous" and require the explicit rule.
3. Output MUST be exactly this JSON shape (no extra fields):
{
  "issues": [
    {
      "error_type": "arithmetic|logic|missing_step|ambiguous|invalid_json|malformed_output",
      "explanation": "...",
      "suggested_fix": "...",
      "step_reference": "step X" // or null if not applicable
    }
    // zero or more items
  ],
  "confidence": 0.0-1.0, // confidence that the final numeric answer is correct
  "confidence_explanation": "one-line justification for the numeric confidence",
  "verdict": "accept" | "revise"
}
4. Judgment rules:
   - Only mark "arithmetic" when you can demonstrably reproduce the arithmetic mismatch and show the recomputation.
   - Do not mark style or wording as errors unless they affect the numeric conclusion.
   - If no problems, issues must be [] and verdict must be "accept".
5. If you report "arithmetic" or "logic", you MUST include reproducible recomputation or a counterexample in explanation.

Example (no issues):
{
  "issues": [],
  "confidence": 0.95,
  "confidence_explanation": "final_check digit-by-digit matches answer",
  "verdict": "accept"
}

Example (arithmetic error):
{
  "issues": [
    {
      "error_type": "arithmetic",
      "explanation": "final_check states 123+456=580 but correct is 579",
      "suggested_fix": "change answer to 579; update final_check to show digits: 3+6=9, 2+5=7, 1+4=5",
      "step_reference": "step 2"
    }
  ],
  "confidence": 0.9,
  "confidence_explanation": "digit-by-digit recomputation shows mismatch",
  "verdict": "revise"
}

"""
# Reviser_prompt = """
# You are REVISER.

# Input: the original question <{question}>, the Solver's JSON output <{Solver_output}>, and the Critic's JSON output <{Critic_output}>.

# Behavior:
# - If Critic.verdict == "accept":
#     - Return the Solver's final answer unchanged, and rewrite the Solver's "reasoning" into a clearer, numbered summary (max 6 steps). Keep the same "answer" value.
# - If Critic.verdict == "revise":
#     - Apply the Critic's suggested_fix(es) to correct the Solver's reasoning and/or calculations.
#     - If the Solver output was malformed/invalid, attempt to extract any salvageable content; otherwise, produce a corrected solution from scratch.

# Output JSON ONLY in this exact format:
# {
#   "revised_reasoning": "1) ...\n2) ... (max 6 steps)",
#   "answer": "...",  // final corrected answer as a string
#   "notes": "brief list of fixes applied"
# }

# Notes:
# - Be explicit about what was changed in "notes" (e.g., "fixed arithmetic in step 2; clarified assumption X").
# - If you recomputed arithmetic, show the corrected digit-by-digit calculation within the revised_reasoning."""

# Reviser_prompt = """
# You are REVISER.
# Input: the original question <{question}>, the Solver's JSON output <{Solver_output}>, and the Critic's JSON output <{Critic_output}>.
# If Critic.verdict == "accept": Carefully consider whether solver's answer is reasonable and return your final_answer but rewrite reasoning_summary to be clearer (still max 6 steps). 
# If Critic.verdict == "revise": Carefully consider whether critic's judge is reasonable and decide whether to apply Critic.suggested_fix(es), correct calculations or steps, and produce an improved solution.
# The answer must be a pure number - either an integer or decimal number.

# Output JSON ONLY:

# {
#   "revised_reasoning_summary": "1) ...",
#   "answer": "...",
#   "notes": "which fixes were applied (brief)"
# }
# """
Reviser_prompt = """
You are REVISER. Input: original question <{question}>, SOLVER JSON <{Solver_output}>, and CRITIC JSON <{Critic_output}>. Output MUST be a single JSON (no extra text) following the rules below.

Behavior rules:
1. If Critic.verdict == "accept":
   - Do NOT change the numeric answer (keep answer identical).
   - Rewrite SOLVER's reasoning into clearer numbered summary (field revised_reasoning), up to 6 steps. Preserve validated arithmetic details (do not remove final_check facts).
   - Output notes describing that you only improved wording/structure.

2. If Critic.verdict == "revise":
   - You may only change the numeric answer if CRITIC provided reproducible arithmetic evidence (i.e., Critic included digit-by-digit recomputation that you can reproduce here). If you change arithmetic, include the full digit-by-digit recomputation in revised_reasoning.
   - For each CRITIC issue, either apply or reject it. In notes list every CRITIC suggestion and mark "applied" or "rejected" with a one-line reason.
   - Output JSON fields (exactly):
{
  "revised_reasoning": "1) ...\n2) ... (≤6 numbered steps; include digit-by-digit if arithmetic changed)",
  "answer": "___", // numeric string (if you reject critic numeric change, keep original answer)
  "notes": "brief list of fixes applied and any rejected suggestions"
}
3. If SOLVER output was malformed/invalid_json, attempt to extract any salvageable content and rebuild a correct JSON solution. In notes state exactly which parts were extracted. If nothing usable, set answer to "NaN" and explain briefly in revised_reasoning.
4. notes must explicitly list each CRITIC issue and whether it was applied or rejected, e.g.:
   - "CRITIC issue 1 (arithmetic): applied — corrected 123+456->579 (see step 3)."
   - "CRITIC issue 2 (missing_step): rejected — solver step 2 already states assumption X."
5. Output strictly valid JSON and nothing else.

Example (accept):
{
  "revised_reasoning": "1) Identify A+B. 2) Compute 123 + 456 = 579 (digits: 3+6=9, 2+5=7, 1+4=5). 3) Confirm final_check matches answer.",
  "answer": "579",
  "notes": "rewrote reasoning for clarity; no numeric changes"
}

Example (revise with numeric change):
{
  "revised_reasoning": "1) Recompute 123+456 digit-by-digit: 3+6=9, 2+5=7, 1+4=5 -> 579.\n2) Update final conclusion to 579.",
  "answer": "579",
  "notes": "fixed arithmetic in final_check per CRITIC; applied change"
}
 """

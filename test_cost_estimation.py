"""
Precise cost estimation test.
Uses REAL prompt templates from the codebase (Selector, Executor, Judge, QA Response)
to measure actual token consumption with enable_thinking=False.

Tests 5 cases per role with realistic prompt lengths.
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any

import openai
import httpx

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# === Real prompt templates from src/ ===

SELECTOR_PROMPT_TEMPLATE = """\
You are a memory management skill selector. Given the input text chunk and retrieved memories, \
select the most appropriate skill(s) from the available skills list.

## Input Text Chunk
{session_text}

## Retrieved Memories (current memory bank state)
{retrieved_memories}

## Available Skills
1. [insert_new_memory] (Action: INSERT)
   Description: When new factual information appears that is not yet stored in the memory bank, insert a concise summary.

2. [update_existing_memory] (Action: UPDATE)
   Description: When existing memory contains outdated or incomplete information that is corrected or extended by new input, update it.

3. [delete_outdated_memory] (Action: DELETE)
   Description: When stored memory is explicitly contradicted or no longer relevant based on newinput, delete it.

4. [noop] (Action: NOOP)
   Description: When the input text does not contain new information or does not require any memory changes, do nothing.

## Instructions
- Analyze the input text and existing memories to determine what memory operations are needed.
- Select 1-1 skill(s) that should be applied to this input.
- Consider: Does new information need to be stored? Do existing memories need updating? \
Is any stored information now outdated or contradicted?
- If no memory operation is needed (input is trivial/redundant), select "noop".

## Output Format
Return a JSON object with the selected skill names:
```json
{{"selected_skills": ["skill_name_1"]}}
```

Only use skill names from the Available Skills list above."""


EXECUTOR_PROMPT_TEMPLATE = """\
You are a memory management executor. Apply the selected skills to the input text \
chunk and retrieved memories, then output memory actions.

Input Text Chunk:
{session_text}

Retrieved Memories (0-based index):
{retrieved_memories}

Selected Skills:
[Skill 1] insert_new_memory
Description: When new factual information appears that is not yet stored in the memory bank, insert a concise summary.
Allowed action: INSERT
Instructions:
- Extract key facts (who, what, when, where) from the input text
- Create a concise but complete memory item
- Include essential context and temporal markers

Guidelines:
- Apply any skill as needed; a skill may be used multiple times.
- Read the input text chunk carefully line by line and apply any skill as needed.
- Only use action types supported by the selected skills.
- MEMORY_INDEX is 0-based and must reference the retrieved memories list.
- Output only action blocks in the format below.
- Do not include explanations or REASONING lines.
Output format (repeat as needed). Use ONE block per action and separate blocks with a blank line:

INSERT block:
ACTION: INSERT
MEMORY_ITEM: <concise but complete summary with essential details>

UPDATE block:
ACTION: UPDATE
MEMORY_INDEX: <0-based index>
UPDATED_MEMORY: <concise but complete merged summary with essential updates>

DELETE block:
ACTION: DELETE
MEMORY_INDEX: <0-based index>
"""


JUDGE_PROMPT_TEMPLATE = """\
You are an expert judge evaluating the quality of an answer for a QA task.
Your goal is to determine whether the model's answer correctly and sufficiently
answers the given question.

Read the following information carefully:

[Question]
{question}

[Ground Truth Answers]
{ground_truth}

[Model Answer]
{model_answer}

Your evaluation criteria:
1. Correctness:
   - Is the model answer factually consistent with ANY of the correct answers?
   - Does it avoid contradictions or introducing false information?

2. Relevance:
   - Does the answer address the question directly without unnecessary content?

3. Completeness:
   - Does the answer include all essential information needed to fully answer the question?
   - Partial answers are allowed but should receive lower scores.

Scoring Rules:
- Score = 1.0 if the answer is fully correct.
- Score = 0.5 if the answer is partially correct but incomplete or slightly inaccurate.
- Score = 0.0 if the answer is incorrect, irrelevant, or contradicts the ground truth.

Output Format (STRICT):
Return your output as a JSON dictionary with two fields:
{{
    "explanation": "<brief explanation of your reasoning>",
    "score": <0.0 | 0.5 | 1.0>
}}

Be concise and objective. Do not include anything outside the JSON."""


QA_RESPONSE_PROMPT_TEMPLATE = """\
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.


History Chats:

{memory_context}

Current Date: 2023-06-15
Question: {question}
Short Answer:"""


# === Realistic test data (simulating real training prompts) ===

def build_realistic_session_text():
    """Generate a realistic session text (~800 tokens, typical for locomo)."""
    return """[2023-05-15]
Caroline: Hey Melanie, how are you doing? I just got back from the LGBTQ support group meeting today. It was really eye-opening.
Melanie: Oh that's great Caroline! How did it go? I remember you were nervous about going.
Caroline: It went really well actually. I met some amazing people there. One woman shared her transition journey and it really resonated with me. I've been thinking a lot about my own identity lately.
Melanie: I'm so proud of you for going. You deserve to explore who you are in a safe space.
Caroline: Thank you. Also, I've been researching some things - I found a really good counseling certification program that I might apply to. I want to help others who are going through similar things.
Melanie: That sounds perfect for you! You've always been so empathetic and good at listening.
Caroline: Speaking of goals, I've also been looking into adoption agencies. Michael and I have been talking about it for a while now.
Melanie: Oh wow, that's a big step! How exciting. By the way, I ran a charity race last Sunday - raised $500 for the local shelter!
Caroline: That's amazing Mel! You're always doing something active. How's the painting going?
Melanie: Good! I finished a sunrise piece last year that I'm really proud of. Been experimenting with acrylics more."""


def build_realistic_memories():
    """Generate realistic retrieved memories (~5 items, typical for training)."""
    return [
        "Caroline attended an LGBTQ support group meeting on 7 May 2023 and found it eye-opening.",
        "Melanie completed a sunrise painting in 2022 using acrylic techniques she learned online.",
        "Caroline is interested in pursuing a counseling certification to help others with identity issues.",
        "Caroline and Michael have been discussing adoption and she has been researching agencies.",
        "Melanie ran a charity race on the Sunday before 25 May 2023, raising $500 for the local animal shelter."
    ]


def build_test_cases(n=5):
    """Build n test cases with realistic prompt lengths."""
    cases = []
    session = build_realistic_session_text()
    memories = build_realistic_memories()
    
    questions = [
        "When did Caroline go to the LGBTQ support group?",
        "What fields would Caroline pursue in education?",
        "When did Melanie paint a sunrise?",
        "What did Caroline research?",
        "How much did Melanie raise in the charity race?",
    ]
    ground_truths = [
        "7 May 2023",
        "Psychology, counseling certification",
        "2022",
        "Adoption agencies",
        "$500 for the local animal shelter",
    ]
    predictions = [
        "7 May 2023",
        "counseling certification",
        "last year",
        "adoption agencies",
        "about $500",
    ]
    
    for i in range(n):
        idx = i % len(questions)
        cases.append({
            "session_text": session,
            "memories": memories,
            "question": questions[idx],
            "ground_truth": ground_truths[idx],
            "prediction": predictions[idx],
        })
    return cases


def call_api(client, model, prompt, no_think=True):
    """Call API with thinking disabled. Return (response, in_tokens, out_tokens)."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "extra_body": {"enable_thinking": False},
    }
    completion = client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content
    usage = completion.usage
    return (content, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)


def run_role_test(client, model, prompts, role_name):
    """Run prompts for a specific role, return stats."""
    print(f"\n{'─'*60}")
    print(f"  {role_name} | Model: {model} | N={len(prompts)}")
    print(f"{'─'*60}")
    
    total_in = 0
    total_out = 0
    results = []
    t0 = time.time()
    
    for i, prompt in enumerate(prompts):
        try:
            resp, in_t, out_t = call_api(client, model, prompt)
            total_in += in_t
            total_out += out_t
            results.append({"in": in_t, "out": out_t, "resp": resp[:80]})
            print(f"    [{i+1}/{len(prompts)}] {in_t} in + {out_t} out | {resp[:60]}")
            time.sleep(0.3)
        except Exception as e:
            print(f"    [{i+1}/{len(prompts)}] ERROR: {e}")
            results.append({"in": 0, "out": 0, "error": str(e)})
            time.sleep(1)
    
    elapsed = time.time() - t0
    avg_in = total_in / len(prompts) if prompts else 0
    avg_out = total_out / len(prompts) if prompts else 0
    
    print(f"  => Total: {total_in}+{total_out}={total_in+total_out} tok | {elapsed:.1f}s")
    print(f"  => Avg per call: {avg_in:.0f} in + {avg_out:.0f} out")
    
    return {
        "role": role_name, "model": model, "n": len(prompts),
        "total_in": total_in, "total_out": total_out,
        "avg_in": round(avg_in, 1), "avg_out": round(avg_out, 1),
        "elapsed": round(elapsed, 1),
    }


def compute_cost_estimate(stats):
    """Compute full training cost from measured stats."""
    print("\n" + "=" * 60)
    print("  COST ESTIMATION (based on measured token consumption)")
    print("=" * 60)
    
    # Training params (optimized)
    inner_epochs = 50
    batch_size = 4
    sessions_per_conv = 19
    qa_per_conv = 200
    qa_ratio = 0.3
    
    # Calls per epoch
    selector_calls_per_epoch = batch_size * sessions_per_conv  # 76
    executor_calls_per_epoch = batch_size * sessions_per_conv  # 76
    qa_calls_per_epoch = int(batch_size * qa_per_conv * qa_ratio)  # 240
    judge_calls_per_epoch = qa_calls_per_epoch  # same as QA if using judge
    
    total_selector = selector_calls_per_epoch * inner_epochs  # 3,800
    total_executor = executor_calls_per_epoch * inner_epochs  # 3,800
    total_qa = qa_calls_per_epoch * inner_epochs  # 12,000
    total_judge = judge_calls_per_epoch * inner_epochs  # 12,000
    
    # Measured averages
    sel = stats["selector"]
    exe = stats["executor"]
    judge = stats["judge"]
    qa = stats["qa_response"]
    
    # Pricing (per million tokens)
    prices = {
        "qwen3.5-flash": {"in": 0.2, "out": 0.8},
        "qwen3.5-plus": {"in": 0.8, "out": 0.8},
        "qwen3.7-plus": {"in": 2.0, "out": 8.0},
    }
    
    print(f"\n  ── Training Parameters ──")
    print(f"  inner_epochs={inner_epochs}, batch_size={batch_size}")
    print(f"  sessions/conv={sessions_per_conv}, qa/conv={qa_per_conv}, qa_ratio={qa_ratio}")
    
    print(f"\n  ── Measured Token Consumption (avg per call) ──")
    print(f"  {'Role':<15} {'Model':<16} {'In tok':<10} {'Out tok':<10}")
    print(f"  {'─'*51}")
    print(f"  {'Selector':<15} {'Flash':<16} {sel['avg_in']:<10.0f} {sel['avg_out']:<10.0f}")
    print(f"  {'Executor':<15} {'Plus':<16} {exe['avg_in']:<10.0f} {exe['avg_out']:<10.0f}")
    print(f"  {'QA Response':<15} {'Plus':<16} {qa['avg_in']:<10.0f} {qa['avg_out']:<10.0f}")
    print(f"  {'Judge':<15} {'3.7-Plus':<16} {judge['avg_in']:<10.0f} {judge['avg_out']:<10.0f}")
    
    # === Plan A: No Judge (f1 metric) ===
    print(f"\n  ══════════════════════════════════════════")
    print(f"  方案A: reward_metric=f1 (无Judge)")
    print(f"  ══════════════════════════════════════════")
    
    sel_in_total = total_selector * sel['avg_in'] / 1_000_000
    sel_out_total = total_selector * sel['avg_out'] / 1_000_000
    exe_in_total = total_executor * exe['avg_in'] / 1_000_000
    exe_out_total = total_executor * exe['avg_out'] / 1_000_000
    qa_in_total = total_qa * qa['avg_in'] / 1_000_000
    qa_out_total = total_qa * qa['avg_out'] / 1_000_000
    
    sel_cost = sel_in_total * prices["qwen3.5-flash"]["in"] + sel_out_total * prices["qwen3.5-flash"]["out"]
    exe_cost = exe_in_total * prices["qwen3.5-plus"]["in"] + exe_out_total * prices["qwen3.5-plus"]["out"]
    qa_cost = qa_in_total * prices["qwen3.5-plus"]["in"] + qa_out_total * prices["qwen3.5-plus"]["out"]
    
    print(f"\n  {'Role':<12} {'Calls':<8} {'In(M)':<8} {'Out(M)':<8} {'Cost(¥)':<10}")
    print(f"  {'─'*46}")
    print(f"  {'Selector':<12} {total_selector:<8} {sel_in_total:<8.2f} {sel_out_total:<8.4f} ¥{sel_cost:<8.1f}")
    print(f"  {'Executor':<12} {total_executor:<8} {exe_in_total:<8.2f} {exe_out_total:<8.4f} ¥{exe_cost:<8.1f}")
    print(f"  {'QA Resp':<12} {total_qa:<8} {qa_in_total:<8.2f} {qa_out_total:<8.4f} ¥{qa_cost:<8.1f}")
    total_a = sel_cost + exe_cost + qa_cost
    total_calls_a = total_selector + total_executor + total_qa
    print(f"  {'─'*46}")
    print(f"  {'TOTAL A':<12} {total_calls_a:<8} {sel_in_total+exe_in_total+qa_in_total:<8.2f} {sel_out_total+exe_out_total+qa_out_total:<8.4f} ¥{total_a:<8.1f}")
    
    # === Plan B: With Judge ===
    print(f"\n  ══════════════════════════════════════════")
    print(f"  方案B: reward_metric=llm_judge (含Judge)")
    print(f"  ══════════════════════════════════════════")
    
    j_in_total = total_judge * judge['avg_in'] / 1_000_000
    j_out_total = total_judge * judge['avg_out'] / 1_000_000
    judge_cost = j_in_total * prices["qwen3.7-plus"]["in"] + j_out_total * prices["qwen3.7-plus"]["out"]
    
    print(f"\n  {'Role':<12} {'Calls':<8} {'In(M)':<8} {'Out(M)':<8} {'Cost(¥)':<10}")
    print(f"  {'─'*46}")
    print(f"  {'Plan A':<12} {total_calls_a:<8} {'─':<8} {'─':<8} ¥{total_a:<8.1f}")
    print(f"  {'Judge':<12} {total_judge:<8} {j_in_total:<8.2f} {j_out_total:<8.4f} ¥{judge_cost:<8.1f}")
    total_b = total_a + judge_cost
    total_calls_b = total_calls_a + total_judge
    print(f"  {'─'*46}")
    print(f"  {'TOTAL B':<12} {total_calls_b:<8} {'─':<8} {'─':<8} ¥{total_b:<8.1f}")
    
    # === Time estimate ===
    print(f"\n  ── 训练时间估算 ──")
    # Assume effective rpm=100 for Plus/3.7-Plus, rpm=300 for Flash
    # Bottleneck is Plus calls (executor + QA)
    plus_calls = total_executor + total_qa
    flash_calls = total_selector
    judge_37_calls = total_judge
    
    # Concurrent: selector is fast and parallel. Bottleneck = Plus sequential
    time_a_hours = plus_calls / (100 * 60)  # 100 rpm
    time_b_hours = time_a_hours + judge_37_calls / (100 * 60)
    
    print(f"  假设有效rpm=100 (Plus/3.7-Plus), rpm=300 (Flash)")
    print(f"  方案A: {plus_calls} Plus调用 / 6000/h = {time_a_hours:.1f}h ≈ {time_a_hours/24:.1f}天")
    print(f"  方案B: +{judge_37_calls} Judge调用 → {time_b_hours:.1f}h ≈ {time_b_hours/24:.1f}天")
    
    # Total tokens this test
    test_tokens = sum(s["total_in"] + s["total_out"] for s in stats.values())
    print(f"\n  ── 本次测试消耗 ──")
    print(f"  Total: {test_tokens:,} tokens")
    print(f"  免费额度剩余: ~{1_000_000 - 13373 - 23240 - test_tokens:,} tokens")
    
    return {"plan_a_cost": round(total_a, 1), "plan_b_cost": round(total_b, 1),
            "plan_a_hours": round(time_a_hours, 1), "plan_b_hours": round(time_b_hours, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--api-base", type=str, default=DASHSCOPE_BASE)
    parser.add_argument("--n-cases", type=int, default=5, help="Cases per role")
    parser.add_argument("--output", type=str, default="./results/cost_estimation.json")
    args = parser.parse_args()

    cases = build_test_cases(args.n_cases)
    
    client = openai.OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
        http_client=httpx.Client(verify=False),
    )

    # Build prompts for each role using REAL templates
    selector_prompts = []
    executor_prompts = []
    judge_prompts = []
    qa_prompts = []
    
    for c in cases:
        mem_text = "\n".join(f"{i}. {m}" for i, m in enumerate(c["memories"]))
        
        selector_prompts.append(SELECTOR_PROMPT_TEMPLATE.format(
            session_text=c["session_text"],
            retrieved_memories=mem_text,
        ))
        executor_prompts.append(EXECUTOR_PROMPT_TEMPLATE.format(
            session_text=c["session_text"],
            retrieved_memories=mem_text,
        ))
        judge_prompts.append(JUDGE_PROMPT_TEMPLATE.format(
            question=c["question"],
            ground_truth=c["ground_truth"],
            model_answer=c["prediction"],
        ))
        qa_prompts.append(QA_RESPONSE_PROMPT_TEMPLATE.format(
            memory_context=mem_text,
            question=c["question"],
        ))

    # Run tests
    stats = {}
    stats["selector"] = run_role_test(client, "qwen3.5-flash", selector_prompts, "Selector")
    stats["executor"] = run_role_test(client, "qwen3.5-plus", executor_prompts, "Executor")
    stats["qa_response"] = run_role_test(client, "qwen3.5-plus", qa_prompts, "QA Response")
    stats["judge"] = run_role_test(client, "qwen3.7-plus", judge_prompts, "Judge")
    
    # Compute cost
    cost = compute_cost_estimate(stats)
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {"stats": stats, "cost_estimate": cost}
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
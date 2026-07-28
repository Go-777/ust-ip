"""
Test per-role model selection with DashScope free tier.

Purpose: Verify that Qwen3.5-Flash works for Selector and Qwen3.5-Plus for Judge/Executor.
Also tests with thinking disabled (chat_prefix to skip reasoning) to reduce output tokens.

Usage:
    python test_model_selection.py --api-key sk-xxx
    python test_model_selection.py --api-key sk-xxx --bad-cases-file ./data/bad_cases_extended.json
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

# Test models
MODELS = {
    "qwen3.5-flash": "Qwen3.5-Flash (¥0.2/M in, ¥0.8/M out)",
    "qwen3.5-plus": "Qwen3.5-Plus (¥0.8/M in, ¥0.8/M out)",
    "qwen3.7-plus": "Qwen3.7-Plus (¥2/M in, ¥8/M out)",
}


def load_test_cases(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def build_selector_prompt(case: Dict) -> str:
    return f"""You are a Skill Selector for a memory-augmented QA system.
Given a question and retrieved memories, select the BEST skill to apply.

Question: {case['question']}
Retrieved Memories: {json.dumps(case.get('retrieved_memories', []))}

Available Skills:
1. insert - Store new information into memory bank
2. update - Update existing memory with new details  
3. delete - Remove outdated or incorrect memory
4. noop - No action needed

Select ONE skill. Output ONLY the skill name (insert/update/delete/noop). No explanation."""


def build_judge_prompt(case: Dict) -> str:
    return f"""You are a QA judge. Rate the prediction quality on a scale of 0-10.

Question: {case['question']}
Ground Truth: {case['ground_truth']}
Prediction: {case['prediction']}

Criteria:
- 10: Perfect match
- 7-9: Substantially correct with minor differences
- 4-6: Partially correct
- 1-3: Mostly wrong
- 0: Completely wrong

Output ONLY a JSON: {{"score": <int>, "reason": "<brief>"}}"""


def call_api(client: openai.OpenAI, model: str, prompt: str, no_think: bool = False) -> tuple:
    """Call API, return (response, in_tokens, out_tokens)."""
    messages = [{"role": "user", "content": prompt}]
    
    # For Qwen3 models: use extra_body to disable thinking for shorter output
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    if no_think:
        kwargs["extra_body"] = {"enable_thinking": False}
    
    completion = client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content
    usage = completion.usage
    return (content, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)


def run_test(client: openai.OpenAI, model: str, prompts: List[str], 
             description: str, no_think: bool = False) -> Dict:
    """Run a batch of prompts against a model."""
    suffix = " [no_think]" if no_think else " [with_think]"
    print(f"\n{'─' * 55}")
    print(f"  {description}{suffix}")
    print(f"  Model: {model} | Prompts: {len(prompts)} | no_think={no_think}")
    print(f"{'─' * 55}")

    results = []
    total_in = 0
    total_out = 0
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        try:
            resp, in_t, out_t = call_api(client, model, prompt, no_think=no_think)
            total_in += in_t
            total_out += out_t
            results.append({"idx": i, "response": resp, "in": in_t, "out": out_t, "ok": True})
            print(f"    [{i+1}/{len(prompts)}] {in_t}+{out_t}tok | {resp[:60]}")
            time.sleep(0.3)
        except Exception as e:
            results.append({"idx": i, "error": str(e), "ok": False})
            print(f"    [{i+1}/{len(prompts)}] FAILED: {e}")
            time.sleep(1)

    elapsed = time.time() - t0
    ok_count = sum(1 for r in results if r["ok"])
    print(f"  => {ok_count}/{len(prompts)} ok | {total_in}+{total_out}={total_in+total_out} tokens | {elapsed:.1f}s")
    
    return {
        "model": model, "description": description, "no_think": no_think,
        "prompts": len(prompts), "ok": ok_count,
        "in_tokens": total_in, "out_tokens": total_out,
        "elapsed": round(elapsed, 1), "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--api-base", type=str, default=DASHSCOPE_BASE)
    parser.add_argument("--bad-cases-file", type=str, default="./data/bad_cases_extended.json")
    parser.add_argument("--output", type=str, default="./results/model_selection_test.json")
    parser.add_argument("--max-cases", type=int, default=20, help="Max cases to test")
    args = parser.parse_args()

    cases = load_test_cases(args.bad_cases_file)[:args.max_cases]
    print(f"Loaded {len(cases)} test cases from {args.bad_cases_file}")

    client = openai.OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
        http_client=httpx.Client(verify=False),
    )

    # Build prompts
    selector_prompts = [build_selector_prompt(c) for c in cases]
    judge_prompts = [build_judge_prompt(c) for c in cases]

    all_results = {}

    # === Test 1: Selector Flash with no_think ===
    all_results["selector_flash_nothink"] = run_test(
        client, "qwen3.5-flash", selector_prompts, "Selector: Flash", no_think=True)

    # === Test 2: Selector Plus with no_think (baseline) ===
    all_results["selector_plus_nothink"] = run_test(
        client, "qwen3.5-plus", selector_prompts, "Selector: Plus", no_think=True)

    # === Test 3: Judge Plus with no_think ===
    all_results["judge_plus_nothink"] = run_test(
        client, "qwen3.5-plus", judge_prompts, "Judge: Plus", no_think=True)

    # === Test 4: Judge 3.7-Plus with no_think ===
    all_results["judge_37plus_nothink"] = run_test(
        client, "qwen3.7-plus", judge_prompts, "Judge: 3.7-Plus", no_think=True)

    # === Summary ===
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)

    # Selector comparison
    print("\n  ── Selector: Flash vs Plus (no_think) ──")
    flash_r = [r["response"] for r in all_results["selector_flash_nothink"]["results"] if r["ok"]]
    plus_r = [r["response"] for r in all_results["selector_plus_nothink"]["results"] if r["ok"]]
    agree = sum(1 for a, b in zip(flash_r, plus_r) if a.strip().lower() == b.strip().lower())
    total = min(len(flash_r), len(plus_r))
    print(f"    Agreement: {agree}/{total} ({100*agree/max(total,1):.0f}%)")
    print(f"    Flash tokens: {all_results['selector_flash_nothink']['in_tokens']}+{all_results['selector_flash_nothink']['out_tokens']}")
    print(f"    Plus tokens:  {all_results['selector_plus_nothink']['in_tokens']}+{all_results['selector_plus_nothink']['out_tokens']}")
    if agree < total:
        print(f"    Disagreements:")
        for i, (a, b) in enumerate(zip(flash_r, plus_r)):
            if a.strip().lower() != b.strip().lower():
                print(f"      [{i}] Flash='{a.strip()[:30]}' vs Plus='{b.strip()[:30]}'")

    # Judge comparison
    print("\n  ── Judge: Plus vs 3.7-Plus (no_think) ──")
    jp = all_results["judge_plus_nothink"]["results"]
    j37 = all_results["judge_37plus_nothink"]["results"]
    print(f"    Plus tokens:    {all_results['judge_plus_nothink']['in_tokens']}+{all_results['judge_plus_nothink']['out_tokens']}")
    print(f"    3.7Plus tokens: {all_results['judge_37plus_nothink']['in_tokens']}+{all_results['judge_37plus_nothink']['out_tokens']}")
    
    # Parse scores
    plus_scores = []
    p37_scores = []
    for rp, r37 in zip(jp, j37):
        try:
            sp = json.loads(rp["response"].strip())["score"] if rp["ok"] else None
        except:
            sp = None
        try:
            s37 = json.loads(r37["response"].strip())["score"] if r37["ok"] else None
        except:
            s37 = None
        plus_scores.append(sp)
        p37_scores.append(s37)
    
    valid_pairs = [(a, b) for a, b in zip(plus_scores, p37_scores) if a is not None and b is not None]
    if valid_pairs:
        corr = sum(1 for a, b in valid_pairs if abs(a - b) <= 2)
        print(f"    Score agreement (±2): {corr}/{len(valid_pairs)} ({100*corr/len(valid_pairs):.0f}%)")
        print(f"    Plus avg:    {sum(a for a,_ in valid_pairs)/len(valid_pairs):.1f}")
        print(f"    3.7Plus avg: {sum(b for _,b in valid_pairs)/len(valid_pairs):.1f}")
        print(f"    Sample scores (Plus vs 3.7Plus):")
        for i, (a, b) in enumerate(valid_pairs[:8]):
            print(f"      [{i}] Plus={a} vs 3.7Plus={b}")

    # Total tokens
    total_tok = sum(r["in_tokens"] + r["out_tokens"] for r in all_results.values())
    print(f"\n  Total tokens consumed this run: {total_tok:,}")
    print(f"  Estimated free tier remaining: ~{1_000_000 - total_tok - 23240:,} tokens")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
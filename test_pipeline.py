"""
Mini Pipeline Test: Run a few training steps to measure actual API consumption.
Also analyze memory bank quality from eval results.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"

import json
import time
import pickle
import numpy as np
from typing import Dict, List, Any

# ============================================================
# PART 1: Measure API cost with a mini training loop
# ============================================================

def measure_api_cost():
    """Run 1 episode (1 conversation, 19 sessions) and track API token usage."""
    import tiktoken
    
    # Load data
    with open("data/locomo10.json", "r") as f:
        data = json.load(f)
    
    # Use train conversation 0
    conversation = data[0]
    conv_data = conversation.get("conversation", {})
    # Extract sessions from conversation dict (session_1, session_2, ...)
    session_keys = sorted(
        [k for k in conv_data.keys() if k.startswith('session_') and not k.endswith('_date_time')],
        key=lambda x: int(x.split('_')[1])
    )
    qa_list = conversation.get("qa", [])
    
    print(f"\n{'='*60}")
    print(f"API Cost Measurement - 1 Episode")
    print(f"{'='*60}")
    print(f"Conversation: {conversation.get('sample_id', 'conv-0')}")
    print(f"Sessions: {len(session_keys)}")
    print(f"QA questions: {len(qa_list)}")
    
    # Estimate token costs per step
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Build session texts (full-session mode)
    session_texts = []
    for sk in session_keys:
        turns = conv_data[sk]
        if isinstance(turns, list):
            text = "\n".join([
                f"{t.get('speaker', 'user')}: {t.get('text', t.get('content', ''))}"
                for t in turns
            ])
        else:
            text = str(turns)
        session_texts.append(text)
    
    # Measure token lengths
    session_token_lengths = [len(enc.encode(s)) for s in session_texts]
    qa_question_lengths = [len(enc.encode(qa.get("question", ""))) for qa in qa_list]
    qa_answer_lengths = [len(enc.encode(str(qa.get("answer", qa.get("ground_truth", ""))))) for qa in qa_list]
    
    print(f"\n--- Session Token Stats ---")
    print(f"  Avg: {np.mean(session_token_lengths):.0f}")
    print(f"  Max: {max(session_token_lengths)}")
    print(f"  Min: {min(session_token_lengths)}")
    print(f"  Total: {sum(session_token_lengths)}")
    
    print(f"\n--- QA Token Stats ---")
    print(f"  Questions avg: {np.mean(qa_question_lengths):.0f}")
    print(f"  Answers avg: {np.mean(qa_answer_lengths):.0f}")
    
    # Estimate per-step API calls
    # Selector prompt: system + session_text + retrieved_memories + op_descriptions
    # Executor prompt: system + session_text + operation_instruction + retrieved_memories
    
    selector_system_tokens = 500  # system prompt
    memory_context_tokens = 200 * 5  # top-5 memories, ~200 tokens each
    op_description_tokens = 100 * 10  # ~10 ops, ~100 tokens each
    
    selector_input_per_step = []
    executor_input_per_step = []
    
    for st in session_token_lengths:
        sel_input = selector_system_tokens + st + memory_context_tokens + op_description_tokens
        exe_input = 500 + st + memory_context_tokens + 200  # system + session + memories + instruction
        selector_input_per_step.append(sel_input)
        executor_input_per_step.append(exe_input)
    
    selector_output = 200  # avg output tokens per selector call
    executor_output = 500  # avg output tokens per executor call
    
    # QA eval: question + retrieved_memories + system prompt
    qa_input_per_question = [500 + ql + memory_context_tokens for ql in qa_question_lengths]
    qa_output = 200  # avg output
    
    # Total for 1 episode
    total_selector_input = sum(selector_input_per_step)
    total_executor_input = sum(executor_input_per_step)
    total_selector_output = len(session_keys) * selector_output
    total_executor_output = len(session_keys) * executor_output
    total_qa_input = sum(qa_input_per_question)
    total_qa_output = len(qa_list) * qa_output
    
    print(f"\n--- Per Episode Token Estimates ---")
    print(f"  Selector: input={total_selector_input:,}, output={total_selector_output:,}")
    print(f"  Executor: input={total_executor_input:,}, output={total_executor_output:,}")
    print(f"  QA eval:  input={total_qa_input:,}, output={total_qa_output:,}")
    
    total_input = total_selector_input + total_executor_input + total_qa_input
    total_output = total_selector_output + total_executor_output + total_qa_output
    
    print(f"\n  Total per episode: input={total_input:,}, output={total_output:,}")
    
    # Cost calculation (Qwen3.6-Plus: input 2元/M, output 8元/M)
    cost_input = total_input / 1_000_000 * 2
    cost_output = total_output / 1_000_000 * 8
    cost_per_episode = cost_input + cost_output
    
    print(f"\n--- Cost Per Episode (Qwen3.6-Plus pricing) ---")
    print(f"  Input cost: ¥{cost_input:.4f}")
    print(f"  Output cost: ¥{cost_output:.4f}")
    print(f"  Total: ¥{cost_per_episode:.4f}")
    
    # Scale to full training
    print(f"\n{'='*60}")
    print(f"Full Training Cost Projection")
    print(f"{'='*60}")
    
    configs = [
        ("默认(inner=100, batch=4, QA=1.0)", 100, 4, 1.0, 10),
        ("优化(inner=50, batch=4, QA=0.3)", 50, 4, 0.3, 10),
        ("轻量(inner=30, batch=2, QA=0.3)", 30, 2, 0.3, 10),
        ("极简(inner=10, batch=1, QA=0.3)", 10, 1, 0.3, 5),
    ]
    
    for name, inner, batch, qa_ratio, outer in configs:
        ep_cost_sel_exe = (total_selector_input + total_executor_input) / 1_000_000 * 2 + \
                          (total_selector_output + total_executor_output) / 1_000_000 * 8
        ep_cost_qa = (total_qa_input * qa_ratio) / 1_000_000 * 2 + \
                     (total_qa_output * qa_ratio) / 1_000_000 * 8
        ep_total = ep_cost_sel_exe + ep_cost_qa
        total_cost = ep_total * batch * inner * outer
        total_calls = (len(session_keys) * 2 + int(len(qa_list) * qa_ratio)) * batch * inner * outer
        # time estimate: 100 rpm
        time_hours = total_calls / 100 / 60
        print(f"  {name}:")
        print(f"    Cost: ¥{total_cost:.1f} | Calls: {total_calls:,} | Time: {time_hours:.1f}h ({time_hours/24:.1f}d)")
    
    return {
        "session_token_lengths": session_token_lengths,
        "qa_count": len(qa_list),
        "cost_per_episode": cost_per_episode
    }


# ============================================================
# PART 2: Analyze memory bank quality from eval results
# ============================================================

def analyze_memory_quality():
    """Analyze the quality of stored memories from eval results."""
    print(f"\n\n{'='*60}")
    print(f"Memory Bank Quality Analysis")
    print(f"{'='*60}")
    
    # Load both memory banks
    memory_dir = "results/memories"
    for fname in sorted(os.listdir(memory_dir)):
        if not fname.endswith(".pkl"):
            continue
        
        with open(os.path.join(memory_dir, fname), "rb") as fp:
            data = pickle.load(fp)
        
        bank = data["memory_bank"]
        memories = bank["memories"]
        
        conv_id = fname.split("sample_")[1].split("_mode")[0]
        print(f"\n--- {conv_id} ({len(memories)} memories) ---")
        
        # Analyze memory content
        contents = [m["content"] for m in memories]
        lengths = [len(m["content"]) for m in memories]
        
        print(f"  Content length: avg={np.mean(lengths):.0f}, max={max(lengths)}, min={min(lengths)}")
        
        # Check for common quality issues
        empty_count = sum(1 for c in contents if len(c.strip()) < 20)
        duplicate_count = len(contents) - len(set(contents))
        
        # Check metadata
        ops_used = {}
        for m in memories:
            meta = m.get("metadata", {})
            op = meta.get("operation", meta.get("op_name", "unknown"))
            ops_used[op] = ops_used.get(op, 0) + 1
        
        print(f"  Empty/trivial: {empty_count}")
        print(f"  Duplicates: {duplicate_count}")
        print(f"  Operations used: {ops_used}")
        
        # Sample some memories
        print(f"\n  Sample memories (first 3):")
        for i, m in enumerate(memories[:3]):
            content = m["content"][:150]
            meta = m.get("metadata", {})
            print(f"    [{i}] op={meta.get('operation', '?')}: {content}...")
        
        print(f"\n  Sample memories (last 3):")
        for i, m in enumerate(memories[-3:]):
            content = m["content"][:150]
            meta = m.get("metadata", {})
            idx = len(memories) - 3 + i
            print(f"    [{idx}] op={meta.get('operation', '?')}: {content}...")


# ============================================================
# PART 3: Analyze QA-Memory alignment
# ============================================================

def analyze_qa_memory_alignment():
    """Check if memories contain info needed to answer QA questions."""
    print(f"\n\n{'='*60}")
    print(f"QA-Memory Alignment Analysis")
    print(f"{'='*60}")
    
    with open("data/locomo10.json", "r") as f:
        data = json.load(f)
    
    # Test conversations are index 8 and 9
    for test_idx in [8, 9]:
        if test_idx >= len(data):
            continue
        conv = data[test_idx]
        conv_id = conv.get("conversation_id", f"conv-{test_idx}")
        qa_list = conv.get("qa", [])
        
        # Load corresponding memory bank
        pkl_pattern = f"conv-{49 + (test_idx - 8)}"  # conv-49, conv-50
        memory_file = None
        for fname in os.listdir("results/memories"):
            if pkl_pattern in fname and fname.endswith(".pkl"):
                memory_file = os.path.join("results/memories", fname)
                break
        
        if not memory_file:
            print(f"\n  No memory bank found for {conv_id}")
            continue
        
        with open(memory_file, "rb") as fp:
            mb_data = pickle.load(fp)
        
        memories = mb_data["memory_bank"]["memories"]
        memory_texts = [m["content"].lower() for m in memories]
        all_memory_text = " ".join(memory_texts)
        
        print(f"\n--- {conv_id}: {len(qa_list)} QA, {len(memories)} memories ---")
        
        # Check keyword overlap for each QA
        categories = {}
        for qa in qa_list:
            cat = qa.get("category", 1)
            answer = str(qa.get("answer", qa.get("ground_truth", "")))
            question = qa.get("question", "")
            
            # Simple keyword check: are answer keywords in memory?
            answer_words = set(answer.lower().split())
            # Remove stop words
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "it", "he", "she", "they", "this", "that"}
            answer_keywords = answer_words - stop_words
            
            if answer_keywords:
                found_in_memory = sum(1 for w in answer_keywords if w in all_memory_text)
                coverage = found_in_memory / len(answer_keywords)
            else:
                coverage = 0
            
            if cat not in categories:
                categories[cat] = {"total": 0, "coverage": []}
            categories[cat]["total"] += 1
            categories[cat]["coverage"].append(coverage)
        
        for cat in sorted(categories.keys()):
            info = categories[cat]
            avg_cov = np.mean(info["coverage"]) if info["coverage"] else 0
            low_cov = sum(1 for c in info["coverage"] if c < 0.3)
            print(f"  Category {cat}: {info['total']} QAs, avg keyword coverage={avg_cov:.2f}, low_coverage(<0.3)={low_cov}")


if __name__ == "__main__":
    stats = measure_api_cost()
    analyze_memory_quality()
    analyze_qa_memory_alignment()
    print("\n\nDone.")
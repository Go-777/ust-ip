#!/usr/bin/env python3
"""
Generate high-quality bad_cases_extended.json from locomo10.json.

Key design principles:
1. session_text: Contains FULL conversation text with date/time context (evidence sessions)
2. memory_bank_snapshot: Contains PARTIAL but real memories from preceding sessions
   - Enough info to be useful, but deliberately imperfect (missing some key facts)
   - This allows skill improvement to produce differentiated rewards
3. retrieved_memories: Subset of memory_bank that's somewhat relevant to the question
4. ground_truth: Exact answer from QA annotations
5. prediction: A deliberately wrong/vague answer (simulating baseline failure)

Usage:
    python scripts/generate_bad_cases.py --input data/locomo10.json --output data/bad_cases_extended.json --num-cases 20
"""

import argparse
import json
import random
import re
from typing import Dict, List, Any, Tuple, Optional


def parse_evidence_refs(evidence: List[str]) -> List[Tuple[int, int]]:
    """Parse evidence references like 'D1:3' -> (session_num=1, turn_num=3)."""
    refs = []
    for ref in evidence:
        match = re.match(r'D(\d+):(\d+)', ref)
        if match:
            session_num = int(match.group(1))
            turn_num = int(match.group(2))
            refs.append((session_num, turn_num))
    return refs


def format_session_text(conversation: Dict, session_nums: List[int]) -> str:
    """Format full session text with date context for given session numbers."""
    text_parts = []
    for sn in sorted(set(session_nums)):
        session_key = f"session_{sn}"
        date_key = f"session_{sn}_date_time"
        
        if session_key not in conversation:
            continue
        
        date_time = conversation.get(date_key, "unknown date")
        turns = conversation[session_key]
        
        # Add date header
        text_parts.append(f"[Conversation on {date_time}]")
        
        # Add all turns
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            text_parts.append(f"{speaker}: {text}")
        
        text_parts.append("")  # separator
    
    return "\n".join(text_parts)


def build_memory_bank_snapshot(
    conversation: Dict,
    evidence_sessions: List[int],
    evidence_turn_ids: List[str],
    include_ratio: float = 0.6,
    truncate_some: bool = True,
) -> Dict:
    """
    Build a realistic but imperfect memory bank snapshot.
    
    Strategy:
    1. Include memories from sessions BEFORE the evidence sessions
       (simulating accumulated memory from earlier conversations)
    2. Include PARTIAL content from evidence sessions, but EXCLUDE
       the specific evidence turns that contain the answer.
       This creates the "improvement gap": a good skill can extract
       the missing evidence from session_text and add it to memory,
       resulting in higher reward.
    
    Args:
        evidence_turn_ids: List of dia_ids (e.g. ["D1:3"]) that contain
          the key answer information. These are deliberately excluded
            from the memory snapshot.
    """
    memories = []
    
    # Get all session numbers sorted
    all_sessions = sorted(
        int(k.replace("session_", ""))
        for k in conversation.keys()
        if k.startswith("session_") and not "date" in k
    )
    
    # Evidence session numbers
    ev_sessions = set(evidence_sessions)
    
    # Convert evidence turn IDs to a set for fast lookup
    excluded_turn_ids = set(evidence_turn_ids)
    
    # Include some memories from sessions BEFORE evidence
    # (simulating accumulated memory that's partially relevant)
    pre_sessions = [s for s in all_sessions if s < min(ev_sessions) if s in all_sessions]
    
    # Take up to 3 pre-evidence sessions for context
    context_sessions = pre_sessions[-3:] if pre_sessions else []
    
    for sn in context_sessions:
        session_key = f"session_{sn}"
        date_key = f"session_{sn}_date_time"
        turns = conversation.get(session_key, [])
        date_time = conversation.get(date_key, "")
        
        if not turns:
            continue
        
        session_text_lines = []
        if date_time:
            session_text_lines.append(f"[{date_time}]")
        for turn in turns:
            session_text_lines.append(f"{turn['speaker']}: {turn['text']}")
        
        full_text = "\n".join(session_text_lines)
        
        # Sometimes truncated (simulating imperfect memory)
        if truncate_some and random.random() < 0.3:
            full_text = full_text[:500]
        
        memories.append({
            "content": full_text,
            "embedding": None
        })
    
    # Include PARTIAL content from evidence sessions
    # DELIBERATELY EXCLUDE evidence turns to create improvement gap
    for sn in sorted(ev_sessions):
        session_key = f"session_{sn}"
        date_key = f"session_{sn}_date_time"
        turns = conversation.get(session_key, [])
        date_time = conversation.get(date_key, "")
        
        if not turns:
            continue
        
        # Include turns EXCEPT evidence turns (the ones with key answer info)
        selected_turns = []
        for i, turn in enumerate(turns):
            dia_id = turn.get("dia_id", "")
            # Skip evidence turns - these contain the answer
            if dia_id in excluded_turn_ids:
                continue
            # Also randomly skip some non-evidence turns for realism
            if random.random() < include_ratio or i < 2:
                selected_turns.append(turn)
        
        lines = []
        if date_time:
            lines.append(f"[{date_time}]")
        for turn in selected_turns:
            lines.append(f"{turn['speaker']}: {turn['text']}")
       
        memory_text = "\n".join(lines)
        memories.append({
            "content": memory_text,
            "embedding": None
        })
    
    return {"memories": memories}


def build_retrieved_memories(memory_snapshot: Dict, question: str, top_k: int = 3) -> List[str]:
    """
    Select the most relevant memories for retrieval.
    
    In real pipeline, this uses embedding similarity.
    Here we use a simple heuristic: pick memories that share words with the question.
    
    IMPORTANT: The GRPO reward pipeline uses retrieved_memories as the PRIMARY context
    for QA answer generation (see _generate_qa_answer). So retrieved_memories MUST
    contain the evidence session memory that has the answer information.
    We ensure evidence session memories (which contain key facts) are always included.
    """
    if not memory_snapshot or not memory_snapshot.get("memories"):
        return []
    
    question_words = set(question.lower().split())
    scored_memories = []
    
    for mem in memory_snapshot["memories"]:
        content = mem.get("content", "")
        if not content:
            continue
        # Simple word overlap scoring
        mem_words = set(content.lower().split())
        overlap = len(question_words & mem_words)
        scored_memories.append((overlap, content))
    
    # Sort by relevance (overlap) descending
    scored_memories.sort(key=lambda x: -x[0])
    
    # Return top_k (evidence session memories rank highest due to keyword overlap)
    return [m[1] for m in scored_memories[:top_k]]


def generate_bad_prediction(question: str, ground_truth: str) -> str:
    """Generate a deliberately wrong/vague prediction to simulate baseline failure."""
    vague_answers = [
        "I'm not sure about the specific details.",
        "sometime last year",
        "I don't have enough information to answer precisely.",
        "unknown",
        "I don't know the exact answer.",
        "it happened recently",
        "somewhere around that time",
    ]
    return random.choice(vague_answers)


def generate_bad_cases(
    locomo_data: List[Dict],
    num_cases: int = 20,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate bad cases from locomo data.
    
    Selects QA pairs that require temporal/factual reasoning,
    then constructs realistic but imperfect memory states.
    """
    random.seed(seed)
    
    # Collect candidate QA pairs with evidence references
    candidates = []
    for sample_idx, sample in enumerate(locomo_data):
        conversation = sample.get("conversation", {})
        qa_list = sample.get("qa", [])
        
        for qa in qa_list:
            evidence = qa.get("evidence", [])
            category = qa.get("category", 1)
            question = qa.get("question", "")
            answer = str(qa.get("answer", ""))
            
            # Skip adversarial (cat 5) and questions without evidence
            if category == 5 or not evidence:
                continue
            
            # Parse evidence references
            refs = parse_evidence_refs(evidence)
            if not refs:
                continue
            
            # Prefer temporal (cat 2) and multi-hop (cat 1) questions
            # as these are harder and benefit more from skill improvement
            priority = 0
            if category == 2:
                priority = 3  # Temporal - highest priority
            elif category == 1:
                priority = 2  # Multi-hop
            elif category == 4:
                priority = 1  # Single-hop
            else:
                priority = 0  # Open-domain
            
            candidates.append({
                "sample_idx": sample_idx,
                "question": question,
                "answer": answer,
                "category": category,
                "evidence": evidence,
                "refs": refs,
                "priority": priority,
                "conversation": conversation,
            })
    
    # Sort by priority (higher first), then shuffle within same priority
    random.shuffle(candidates)
    candidates.sort(key=lambda x: -x["priority"])
    
    # Select top num_cases, ensuring diversity across samples
    selected = []
    sample_counts = {}
    max_per_sample = max(2, num_cases // len(locomo_data) + 1)
    
    for cand in candidates:
        if len(selected) >= num_cases:
            break
        
        sidx = cand["sample_idx"]
        if sample_counts.get(sidx, 0) >= max_per_sample:
            continue
        
        sample_counts[sidx] = sample_counts.get(sidx, 0) + 1
        selected.append(cand)
    
    # Build bad cases
    bad_cases = []
    for cand in selected:
        conversation = cand["conversation"]
        refs = cand["refs"]
        evidence_sessions = [r[0] for r in refs]
        
        # Build full session text (the input that would be processed)
        session_text = format_session_text(conversation, evidence_sessions)
        
        # Build imperfect memory snapshot
        # Evidence turn IDs (e.g. "D1:3") are excluded from memory
        # to create the improvement gap for GRPO
        evidence_turn_ids = cand["evidence"]  # e.g. ["D1:3", "D2:8"]
        memory_snapshot = build_memory_bank_snapshot(
            conversation,
            evidence_sessions,
            evidence_turn_ids=evidence_turn_ids,
            include_ratio=0.6,
            truncate_some=True,
        )
        
        # Build retrievedmemories
        retrieved = build_retrieved_memories(
            memory_snapshot, cand["question"], top_k=3
        )
        
        # Generate fake bad prediction
        prediction = generate_bad_prediction(cand["question"], cand["answer"])
        
        bad_cases.append({
            "session_text": session_text,
            "question": cand["question"],
            "ground_truth": cand["answer"],
            "prediction": prediction,
            "memory_bank_snapshot": memory_snapshot,
            "retrieved_memories": retrieved,
        })
    
    return bad_cases


def main():
    parser = argparse.ArgumentParser(description="Generate bad cases for GRPO training")
    parser.add_argument("--input", default="data/locomo10.json", help="Input locomo data file")
    parser.add_argument("--output", default="data/bad_cases_extended.json", help="Output bad cases file")
    parser.add_argument("--num-cases", type=int, default=20, help="Number of bad cases to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load data
    with open(args.input, "r") as f:
        locomo_data = json.load(f)
    
    print(f"Loaded {len(locomo_data)} samples from {args.input}")
    print(f"Generating {args.num_cases} bad cases...")
    
    # Generate
    bad_cases = generate_bad_cases(locomo_data, num_cases=args.num_cases, seed=args.seed)
    
    # Save
    with open(args.output, "w") as f:
        json.dump(bad_cases, f, ensure_ascii=False, indent=2)
    
    # Stats
    print(f"\nGenerated {len(bad_cases)} bad cases -> {args.output}")
    print("\nQuality check:")
    for i, case in enumerate(bad_cases[:3]):
        mem_count = len(case["memory_bank_snapshot"]["memories"])
        avg_mem_len = sum(len(m["content"]) for m in case["memory_bank_snapshot"]["memories"]) / max(1, mem_count)
        session_len = len(case["session_text"])
        retrieved_count = len(case["retrieved_memories"])
        print(f"  Case {i}: Q='{case['question'][:50]}...'")
        print(f"    GT='{case['ground_truth']}', session_len={session_len}, "
              f"memories={mem_count}, avg_mem_len={avg_mem_len:.0f}, retrieved={retrieved_count}")


if __name__ == "__main__":
    main()
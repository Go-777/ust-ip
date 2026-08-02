"""
Prepare SFT training data from GRPO-collected offline samples.

Strategy: Reject Sampling SFT
- For each sample group, select the response with the highest reward
- Format as (prompt, completion) pairs for TRL SFTTrainer

Input:  grpo_data/grpo_training_data_filtered.jsonl
Output: grpo_data/sft_train.jsonl (for TRL SFTTrainer)
"""

import json
import argparse
from pathlib import Path


def load_grpo_data(input_path: str):
    """Load GRPO training samples from JSONL."""
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def reject_sampling_sft(samples, top_k: int = 1):
    """
    Reject Sampling: select top-k responses per group by reward.
    
    Args:
        samples: list of {prompt, responses, rewards}
        top_k: number of best responses to keep per group (default=1)
    
    Returns:
        list of {messages: [{role, content}, ...]} in chat format
    """
    sft_data = []
    
    for sample in samples:
        prompt = sample["prompt"]
        responses = sample["responses"]
        rewards = sample["rewards"]
        
        # Sort by reward descending, take top_k
        indexed = sorted(
            enumerate(rewards), key=lambda x: x[1], reverse=True
        )
        
        for idx, reward in indexed[:top_k]:
            response = responses[idx]
            # Format as chat messages for TRL SFTTrainer
            sft_item = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                "reward": reward  # metadata, not used in training
            }
            sft_data.append(sft_item)

    return sft_data


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from GRPO samples")
    parser.add_argument(
        "--input", type=str,
        default="grpo_data/grpo_training_data_filtered.jsonl",
        help="Input GRPO data file"
    )
    parser.add_argument(
        "--output", type=str,
        default="grpo_data/sft_train.jsonl",
        help="Output SFT training data file"
    )
    parser.add_argument(
        "--top-k", type=int, default=1,
        help="Number of best responses to keep per group (default: 1)"
    )
    parser.add_argument(
        "--min-reward", type=float, default=0.3,
        help="Minimum reward threshold for inclusion (default: 0.3)"
    )
    args = parser.parse_args()
    
    # Load data
    samples = load_grpo_data(args.input)
    print(f"Loaded {len(samples)} GRPO samples")
    
    # Filter by minimum reward
    filtered = []
    for s in samples:
        max_r = max(s["rewards"])
        if max_r >= args.min_reward:
            filtered.append(s)
    print(f"After min_reward={args.min_reward} filter: {len(filtered)} samples")
    
    # Reject sampling
    sft_data = reject_sampling_sft(filtered, top_k=args.top_k)
    print(f"Generated {len(sft_data)} SFT training examples")
    
    # Stats
    rewards = [item["reward"] for item in sft_data]
    print(f"  Reward stats: min={min(rewards):.3f}, max={max(rewards):.3f}, "
          f"avg={sum(rewards)/len(rewards):.3f}")
    
    # Estimate token lengths
    prompt_lens = [len(item["messages"][0]["content"]) for item in sft_data]
    resp_lens = [len(item["messages"][1]["content"]) for item in sft_data]
    total_lens = [p + r for p, r in zip(prompt_lens, resp_lens)]
    avg_tokens = sum(total_lens) / len(total_lens) / 4  # rough char/token ratio
    max_tokens = max(total_lens) / 4
    print(f"  Avg total length: ~{avg_tokens:.0f} tokens")
    print(f"  Max total length: ~{max_tokens:.0f} tokens")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
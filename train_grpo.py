"""
GRPO Training Entry Point for MemSkill Designer.

Usage:
    python train_grpo.py \
        --dataset locomo \
        --data-file ./data/locomo_data.json \
        --grpo-enabled \
        --dashscope-api-keys sk-xxx,sk-yyy \
        --grpo-export-dir ./grpo_data \
        --grpo-max-iterations 50

This script:
1. Loads dataset and prepares bad cases (from eval or provided file)
2. Initializes LLMClient, SkillSelector, Executor, Designer, OperationBank
3. Runs GRPOTrainingLoop iterations (Stage1 analysis → Stage2 sampling → Reward → Apply best)
4. Exports GRPO training data (JSONL) for OpenRLHF fine-tuning
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import random
import numpy as np
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False
from typing import List, Dict, Any, Optional

from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.llm_client import create_llm_client_from_args
from src.skill_selector import SkillSelector
from src.operation_bank import OperationBank
from src.grpo_trainer import (
    GRPOConfig,
    GRPOTrainingLoop,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_bad_cases(path: str) -> List[Dict[str, Any]]:
    """Load bad cases from a JSON/JSONL file.

    Each bad case should have at minimum:
      - session_text: str
      - question: str
      - ground_truth: str (or list of str)
      - prediction: str (model's wrong answer)
    """
    cases = []
    with open(path, 'r') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                cases = data
            elif isinstance(data, dict) and "bad_cases" in data:
                cases = data["bad_cases"]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


def build_grpo_config_from_args(args, config: AgenticMemoryConfig) -> GRPOConfig:
    """Build GRPOConfig from parsed args and config."""
    return GRPOConfig(
        group_size=getattr(config, "grpo_group_size", 8),
        clip_epsilon=getattr(config, "grpo_clip_epsilon", 0.2),
        kl_coef=getattr(config, "grpo_kl_coef", 0.05),
        learning_rate=getattr(config, "grpo_lr", 5e-6),
        temperature=getattr(config, "grpo_temperature", 0.7),
        max_designer_tokens=getattr(config, "grpo_max_designer_tokens", 4096),
        reward_metric=getattr(config, "reward_metric", "f1"),
        num_bad_cases=getattr(config, "grpo_num_bad_cases", 100),
        max_iterations=getattr(config, "grpo_max_iterations", 50),
        early_stop_patience=getattr(config, "grpo_early_stop_patience", 5),
        case_chunk_size=getattr(config, "grpo_case_chunk_size", 5),
        export_dir=getattr(config, "grpo_export_dir", "./grpo_data"),
    )


def main():
    # Parse arguments (reuse the same argparser with GRPO flags)
    args = get_agentic_memory_args()
    set_seed(args.seed)

    # Load config
    config = AgenticMemoryConfig()
    config.update_from_args(args)

    if not getattr(config, "grpo_enabled", False):
        print("[GRPO] ERROR: --grpo-enabled flag is required for GRPO training.")
        print("       Add --grpo-enabled to your command line arguments.")
        return

    # Create output directories
    export_dir = getattr(config, "grpo_export_dir", "./grpo_data")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(getattr(args, "save_dir", "./checkpoints"), exist_ok=True)

    # ========== 1. Initialize LLM Client ==========
    print("\n" + "=" * 80)
    print("Initializing LLM Client...")
    print("=" * 80)
    llm_client = create_llm_client_from_args(args)
    print(f"  Roles configured: {list(llm_client._role_configs.keys())}")

    # ========== 2. Initialize Components ==========
    print("\nInitializing OperationBank...")
    operation_bank = OperationBank(
        encoder=None,
        max_ops=getattr(args, 'max_ops', 20),
        skip_noop=getattr(args, 'skip_noop', False),
    )

    print("Initializing SkillSelector...")
    skill_selector = SkillSelector(
        llm_client=llm_client,
        operation_bank=operation_bank,
        max_skills=getattr(config, "action_top_k", 3),
    )

    # Note: Executor and Designer are not needed here since GRPOTrainingLoop
    # uses llm_client.call(role="executor"/"designer") directly.
    # They are initialized elsewhere for standard inference (main.py).

    # ========== 3. Load Bad Cases ==========
    print("\n" + "=" * 80)
    print("Loading Bad Cases...")
    print("=" * 80)

    bad_cases_path = getattr(args, "bad_cases_file", None)
    if bad_cases_path is None:
        # Try default path
        bad_cases_path = os.path.join("./data", "bad_cases.json")

    if not os.path.exists(bad_cases_path):
        print(f"[GRPO] ERROR: Bad cases file not found: {bad_cases_path}")
        print("       Please provide --bad-cases-file or place bad_cases.json in ./data/")
        print("       Format: [{\"session_text\": ..., \"question\": ..., \"ground_truth\": ..., \"prediction\": ...}, ...]")
        return

    bad_cases = load_bad_cases(bad_cases_path)
    print(f"  Loaded {len(bad_cases)} bad cases from {bad_cases_path}")

    if not bad_cases:
        print("[GRPO] ERROR: No bad cases found. Cannot proceed with GRPO training.")
        return

    # Limit to configured number
    grpo_config = build_grpo_config_from_args(args, config)
    if len(bad_cases) > grpo_config.num_bad_cases:
        bad_cases = random.sample(bad_cases, grpo_config.num_bad_cases)
        print(f"  Sampled {grpo_config.num_bad_cases} bad cases for training")

    # ========== 4. Build GRPO Training Loop ==========
    print("\n" + "=" * 80)
    print("Setting up GRPO Training Loop...")
    print("=" * 80)
    print(f"  Group size (G): {grpo_config.group_size}")
    print(f"  Temperature: {grpo_config.temperature}")
    print(f"  Max iterations: {grpo_config.max_iterations}")
    print(f"  Early stop patience: {grpo_config.early_stop_patience}")
    print(f"  Case chunk size: {grpo_config.case_chunk_size}")
    print(f"  Export dir: {grpo_config.export_dir}")

    training_loop = GRPOTrainingLoop(
        args=args,
        config=grpo_config,
        llm_client=llm_client,
        operation_bank=operation_bank,
        skill_selector=skill_selector,
    )

    # ========== 5. Prepare Prompt Templates ==========
    # Simplified templates for GRPO (bad_cases passed as formatted string)
    analysis_prompt_template = """You are an expert analyst for a memory-augmented QA system.
Analyze these failure cases and identify patterns in why the system failed.

## Failure Cases
{bad_cases}

## Instructions
1. Group failures by root cause: storage_failure / retrieval_failure / memory_quality_failure.
2. For each pattern, propose a concrete skill change.
3. Output JSON with "failure_patterns" and "recommendations".

Output ONLY valid JSON."""

    refinement_prompt_template = """Based on the following analysis of failure cases, propose a specific skill improvement.

## Analysis
{analysis}

## Current Skills
{current_skills}

## Instructions
Propose ONE concrete skill change. Output JSON with:
- "action": "add_new" or "refine"
- "target_skill": skill name to refine (or null for add_new)
- "name": skill name
- "description": what the skill does
- "instruction_template": the LLM instruction template for this skill
- "update_type": "insert" or "update" or "delete"

## CRITICAL CONSTRAINTS for instruction_template:
1. The skill's instruction_template MUST instruct the executor to output actions using ONLY these standard action types: INSERT, UPDATE, DELETE, or NOOP.
2. The executor output format MUST follow this structure:
   ACTION: INSERT/UPDATE/DELETE/NOOP
   MEMORY_ITEM: <content to store>  (for INSERT)
   MEMORY_INDEX: <index>  (for UPDATE/DELETE)
   UPDATED_MEMORY: <new content>  (for UPDATE)
   REASONING: <why this action>
3. Do NOT invent new action types (e.g., no "STRUCTURE_AND_ENRICH", no "PRE-PROCESSING").
4. Any preprocessing logic (temporal normalization, fact extraction, etc.) should be described as STEPS within the skill that produce a final INSERT/UPDATE/DELETE action — not as a separate action type.

Output ONLY valid JSON."""

    # ========== 6. Run GRPO Training Loop ==========
    print("\n" + "=" * 80)
    print("Starting GRPO Training Loop")
    print("=" * 80)

    iteration = 0
    all_samples = []
    while not training_loop.should_stop():
        iteration += 1
        print(f"\n{'─' * 60}")
        print(f"  Iteration {iteration}/{grpo_config.max_iterations}")
        print(f"{'─' * 60}")

        # Select chunk of bad cases for this iteration
        chunk_start = ((iteration - 1) * grpo_config.case_chunk_size) % len(bad_cases)
        # Shuffle bad_cases every time we complete a full pass
        if chunk_start == 0 and iteration > 1:
            random.shuffle(bad_cases)
        chunk_end = chunk_start + grpo_config.case_chunk_size
        if chunk_end > len(bad_cases):
            chunk = bad_cases[chunk_start:] + bad_cases[:chunk_end - len(bad_cases)]
        else:
            chunk = bad_cases[chunk_start:chunk_end]

        # Run one iteration
        result = training_loop.run(
            bad_cases=chunk,
            analysis_prompt_template=analysis_prompt_template,
            refinement_prompt_template=refinement_prompt_template,
            export_dir=grpo_config.export_dir,
        )

        if result and result.get("improved"):
            print(f"  Best reward: {result.get('best_reward', 0):.4f}")
            print(f"  Avg reward: {result.get('avg_reward', 0):.4f}")
            print(f"  Improved: True")
        else:
            avg_r = result.get("avg_reward", 0) if result else 0
            print(f"  Avg reward: {avg_r:.4f}, no improvement.")

        # Collect samples for final export
        if result and result.get("samples"):
            all_samples.extend(result["samples"])

    # ========== 7. Export & Summary ==========
    print("\n" + "=" * 80)
    print("GRPO Training Complete!")
    print("=" * 80)

    summary = training_loop.get_summary()
    print(f"  Total iterations: {summary.get('total_iterations', 0)}")
    print(f"  Best avg reward: {summary.get('best_avg_reward', 0):.4f}")
    print(f"  Reward history: {summary.get('reward_history', [])}")

    # Export all collected GRPO samples for OpenRLHF
    if all_samples:
        export_path = os.path.join(grpo_config.export_dir, "grpo_training_data_all.jsonl")
        training_loop.data_preparer.export_for_openrlhf(all_samples, export_path)
        print(f"\n  Exported {len(all_samples)} GRPO samples to: {export_path}")
        print(f"  Use with OpenRLHF for Qwen3.5-9B weight updates.")

    # Save final operation bank state
    op_bank_path = os.path.join(grpo_config.export_dir, "operation_bank_final.json")
    with open(op_bank_path, 'w') as f:
        json.dump(operation_bank.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"  Saved final operation bank to: {op_bank_path}")

    print("\n" + "=" * 80)
    print("Done! Next steps:")
    print("  1. Use grpo_training_data_all.jsonl with OpenRLHF for Qwen3.5-9B fine-tuning")
    print("  2. Deploy fine-tuned model as new Designer")
    print("  3. Re-evaluate on test set")
    print("=" * 80)


if __name__ == "__main__":
    main()
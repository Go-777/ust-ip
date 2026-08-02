"""
SFT Fine-tuning for MemSkill Designer using Reject Sampling data.

Uses TRL SFTTrainer with LoRA (QLoRA) to fine-tune Qwen2.5-7B-Instruct
on the best responses from GRPO offline collection.

Usage:
    python train_sft_designer.py \
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
        --data_path grpo_data/sft_train.jsonl \
        --output_dir checkpoints/designer_sft \
        --num_train_epochs 3
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig


def load_sft_dataset(data_path: str) -> Dataset:
    """Load SFT training data from JSONL (messages format)."""
    records = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                records.append({"messages": item["messages"]})
    
    dataset = Dataset.from_list(records)
    print(f"Loaded {len(dataset)} training examples from {data_path}")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, 
                        default="grpo_data/sft_train.jsonl")
    parser.add_argument("--output_dir", type=str, 
                        default="checkpoints/designer_sft")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("MemSkill Designer SFT Fine-tuning (Reject Sampling)")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"QLoRA (4-bit): {args.use_4bit}")
    print("=" * 60)

    # Load dataset
    dataset = load_sft_dataset(args.data_path)

    # Quantization config for QLoRA
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        # Dataset config for chat template
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    # Initialize trainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Print trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    metrics["trainable_params"] = trainable_params
    metrics["total_params"] = total_params
    metrics["num_examples"] = len(dataset)

    metrics_path = Path(args.output_dir) / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {args.output_dir}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
"""
LLM Controller: Drop-in replacement for PPOController.

Uses a strong LLM to select top-K operations from the operation bank,
keeping the same interface as PPOController so the rest of the pipeline
(Retriever, Designer, Executor) works unchanged.

Key design:
- __call__ signature matches PPOController.forward():
    (state_embedding, op_embeddings, deterministic, new_op_mask) -> (action_idx, log_prob, value)
- Internally uses text context (session_text + retrieved_memories + op descriptions)
  set via set_context() before each call.
- Returns dummy log_prob=0.0 and value=0.0 (no training needed).
"""
import json
import re
import logging
from typing import List, Optional, Tuple, Union

import torch
import numpy as np

from src.llm_client import LLMClient
from src.operation_bank import OperationBank, Operation

logger = logging.getLogger("AgenticMemory")


CONTROLLER_PROMPT = """\
You are a memory management controller. Given the current conversation text and \
retrieved memories, select the most appropriate operation(s) to apply.

## Current Text Chunk
{session_text}

## Retrieved Memories (existing memory bank entries)
{retrieved_memories}

## Available Operations
{operations_list}

## Instructions
- Analyze the input text and existing memories to decide which memory operation(s) are needed.
- Select exactly {top_k} operation(s) from the list above (by index number).
- Reasoning guidelines:
  * If the text contains new factual information not in existing memories → INSERT
  * If the text updates/extends an existing memory → UPDATE
  * If the text contradicts an existing memory → DELETE the old one (and INSERT new)
  * If no memory action is needed (trivial/redundant text) → NOOP
- Consider ALL available operations, including any specialized ones created by the Designer.

## Output Format
Return a JSON object:
```json
{{"selected_indices": [0, 2], "reasoning": "brief explanation"}}
```
The indices are 0-based, corresponding to the operation list above.
Select exactly {top_k} index(es). If fewer operations are warranted, include noop to fill.\
"""


class LLMController:
    """
    LLM-based Controller that replaces PPOController for inference.

    Maintains the same __call__ interface so trainer/main code works unchanged.
    No training is needed — the LLM acts as a zero-shot policy.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        operation_bank: OperationBank,
        action_top_k: int = 1,
        role: str = "selector",
        temperature: float = 0.0,
        **kwargs,  # Accept and ignore PPOController-specific kwargs
    ):
        """
        Args:
            llm_client: Unified LLM client for API calls
            operation_bank: Operation bank with available skills
            action_top_k: Number of operations to select per step
            role: LLM client role (determines which model/endpoint to use)
            temperature: LLM sampling temperature (0.0 = deterministic)
        """
        self.llm_client = llm_client
        self.operation_bank = operation_bank
        self.action_top_k = action_top_k
        self.role = role
        self.temperature = temperature

        # Context set before each call
        self._session_text: str = ""
        self._retrieved_memories: List[str] = []

        # Track training mode (compatibility with nn.Module interface)
        self._training = False

    def set_context(self, session_text: str, retrieved_memories: List[str]):
        """
        Set the text context for the next __call__.
        Must be called before each controller invocation.

        Args:
            session_text: Current conversation chunk being processed
            retrieved_memories: List of retrieved memory strings
        """
        self._session_text = session_text
        self._retrieved_memories = retrieved_memories

    def eval(self):
        """Compatibility with nn.Module.eval() — no-op for LLM controller."""
        self._training = False
        return self

    def train(self, mode: bool = True):
        """Compatibility with nn.Module.train() — no-op for LLM controller."""
        self._training = mode
        return self

    def parameters(self):
        """Compatibility — return empty iterator (no trainable params)."""
        return iter([])

    def state_dict(self):
        """Compatibility — return empty dict."""
        return {}

    def load_state_dict(self, state_dict, strict=True):
        """Compatibility — no-op."""
        pass

    def to(self, device):
        """Compatibility — no-op."""
        return self

    def set_new_action_bias_scale(self, bias_scale: float):
        """Compatibility — no-op for LLM controller (no RL exploration bias)."""
        pass

    def __call__(
        self,
        state_embedding: torch.Tensor,
        op_embeddings: torch.Tensor,
        deterministic: bool = False,
        new_op_mask=None,
    ) -> Tuple[Union[int, List[int]], float, float]:
        """
        Select operation(s) using LLM.

        Same signature as PPOController.forward():
            state_embedding: [state_dim] tensor (ignored, we use text context)
            op_embeddings: [num_ops, op_dim] tensor (ignored, we use text descriptions)
            deterministic: bool (controls LLM temperature)
            new_op_mask: optional mask (ignored, LLM sees all ops equally)

        Returns:
            action_idx:int (if top_k=1) or List[int] (if top_k>1)
            log_prob: 0.0 (dummy, no training)
            value: 0.0 (dummy, no training)
        """
        candidate_ops = self.operation_bank.get_candidate_operations()
        num_ops = len(candidate_ops)

        if num_ops == 0:
            logger.warning("[LLMController] No candidate operations available")
            return 0, 0.0, 0.0

        k = min(self.action_top_k, num_ops)

        # Build operations list text
        ops_text = self._build_operations_list(candidate_ops)
        mem_text = self._format_memories(self._retrieved_memories)

        prompt = CONTROLLER_PROMPT.format(
            session_text=self._session_text,
            retrieved_memories=mem_text,
            operations_list=ops_text,
            top_k=k,
        )

        # Call LLM
        temp = 0.0 if deterministic else self.temperature
        try:
            response = self.llm_client.call(
               role=self.role,
                prompt=prompt,
                temperature=temp,
            )
            selected_indices = self._parse_response(response, num_ops, k)
        except Exception as e:
            logger.warning(f"[LLMController] LLM call failed: {e}, using fallback")
            selected_indices = self._fallback_selection(candidate_ops, k)

        # Return in the same format as PPOController
        if k == 1:
            return selected_indices[0], 0.0, 0.0
        else:
            return selected_indices, 0.0, 0.0

    def _build_operations_list(self, candidate_ops: List[Operation]) -> str:
        """Build numbered list of operations with descriptions."""
        lines = []
        for i, op in enumerate(candidate_ops):
            name = op.name
            desc = op.description
            update_type = op.update_type.upper()
            usage = op.meta_info.get("usage_count", 0)
            avg_reward = op.meta_info.get("avg_reward", 0.0)
            lines.append(f"{i}. [{name}] (Type: {update_type})")
            lines.append(f"   Description: {desc}")
            if usage > 0:
                lines.append(f"   Stats: used {usage} times, avg_reward={avg_reward:.3f}")
            lines.append("")
        return "\n".join(lines)

    def _format_memories(self, retrieved_memories: List[str]) -> str:
        """Format retrieved memories for prompt."""
        if not retrieved_memories:
            return "(No existing memories in bank)"
        return "\n".join(
            f"{i+1}. {mem}" for i, mem in enumerate(retrieved_memories)
        )

    def _parse_response(self, response: str, num_ops: int, k: int) -> List[int]:
        """Parse LLM response to extract selected operation indices."""
        # Try JSON extraction
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                indices = data.get("selected_indices", [])
                if isinstance(indices, list):
                    # Validate indices
                    valid = [int(i) for i in indices if 0 <= int(i) < num_ops]
                    if valid:
                        return valid[:k]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: try to find numbers in response
        numbers = re.findall(r'\b(\d+)\b', response)
        valid = [int(n) for n in numbers if 0 <= int(n) < num_ops]
        if valid:
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for n in valid:
                if n not in seen:
                    seen.add(n)
                    deduped.append(n)
            return deduped[:k]

        # Last resort: default to first non-noop operation
        return self._fallback_selection(
            self.operation_bank.get_candidate_operations(), k
        )

    def _fallback_selection(self, candidate_ops: List[Operation], k: int) -> List[int]:
        """Fallback: select first non-noop operation(s)."""
        non_noop = [
            i for i, op in enumerate(candidate_ops)
            if op.update_type.lower() != "noop"
        ]
        if non_noop:
            return non_noop[:k]
        return list(range(min(k, len(candidate_ops))))
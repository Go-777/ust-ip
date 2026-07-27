"""
GRPO Trainer: Group Relative Policy Optimization for Designer LLM.

Core algorithm:
1. Collect bad cases from evaluation
2. Designer Stage1: Analyze bad cases (greedy, 1 response)
3. Designer Stage2: Generate G candidate skill proposals (temp=0.7)
4. For each candidate: apply skill -> re-run bad cases -> compute reward
5. Compute GRPO loss using group-relative advantages
6. Update Designer model parameters

Training is done via OpenRLHF framework (external), this module handles:
- Reward computation pipeline (apply candidate -> evaluate -> reward)
- GRPO data preparation (prompts, responses, rewards for OpenRLHF)
- Integration with the MemSkill inference loop
"""
import os
import json
import logging
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm_client import LLMClient
from src.operation_bank import OperationBank, Operation
from src.skill_selector import SkillSelector
from src.memory_bank import MemoryBank

logger = logging.getLogger("AgenticMemory")


@dataclass
class GRPOConfig:
    """GRPO training hyperparameters."""
    group_size: int = 8  # G: number of candidate samples per prompt
    clip_epsilon: float = 0.2  # PPO-style clipping
    kl_coef: float = 0.05  # KL divergence penalty coefficient
    learning_rate: float = 5e-6
    temperature: float = 0.7  # Sampling temperature for Stage2
    max_designer_tokens: int = 4096
    reward_metric: str = "f1"  # "f1" or "llm_judge"
    num_bad_cases: int = 100  # Number of bad cases per GRPO iteration
    max_iterations: int = 50  # Maximum GRPO training iterations
    early_stop_patience: int = 5  # Stop if no improvement for N iterations
    case_chunk_size: int = 5  # Number of bad cases per analysis batch
    export_dir: str = "./grpo_data"  # Directory for exported GRPO data


@dataclass
class GRPOSample:
    """A single GRPO training sample (prompt + G responses + rewards)."""
    prompt: str  # Designer input prompt (bad case analysis)
    responses: List[str] = field(default_factory=list)  # G candidate responses
    rewards: List[float] = field(default_factory=list)  # Reward for each response
    analysis: str = ""  # Stage1 analysis result

    @property
    def advantages(self) -> List[float]:
        """Compute group-relative advantages (normalized within group)."""
        if not self.rewards:
            return []
        mean_r = np.mean(self.rewards)
        std_r = np.std(self.rewards)
        if std_r < 1e-8:
            return [0.0] * len(self.rewards)
        return [(r - mean_r) / std_r for r in self.rewards]

    def best_response_idx(self) -> int:
        """Get index of the best response (highest reward)."""
        if not self.rewards:
            return 0
        return int(np.argmax(self.rewards))

    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "responses": self.responses,
            "rewards": self.rewards,
            "analysis": self.analysis,
            "advantages": self.advantages,
        }


class GRPORewardComputer:
    """
    Computes rewards for GRPO candidates by:
    1. Temporarily applying each candidate skill to the operation bank
    2. Re-running bad cases through the full pipeline
    3. Measuring QA performance (F1 / Judge score)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        operation_bank: OperationBank,
        config: GRPOConfig,
        evaluator: Any = None,
        data_processor: Any = None,
    ):
        self.llm_client = llm_client
        self.operation_bank = operation_bank
        self.config = config
        self.evaluator = evaluator
        self.data_processor = data_processor

    def compute_reward_for_candidate(
        self,
        candidate_skill: Dict,
        bad_cases: List[Dict],
        base_operation_bank_dict: Dict,
    ) -> float:
        """
        Compute reward for a single candidate skill proposal.

        Args:
            candidate_skill: Parsed skill definition dict with keys:
                - name, description, instruction_template, update_type
                - action: "add_new" or "refine" (with target_skill for refine)
            bad_cases: List of bad case dicts to re-evaluate
            base_operation_bank_dict: Snapshot of operation bank before modification

        Returns:
            Average reward score across bad cases
        """
        # Create a temporary operation bank with the candidate applied
        temp_op_bank = OperationBank.from_dict(base_operation_bank_dict, encoder=None)

        # Apply candidate skill
        action = candidate_skill.get("action", "add_new")
        if action == "refine":
            target = candidate_skill.get("target_skill", "")
            if target and target in temp_op_bank.operations:
                op = temp_op_bank.operations[target]
                op.description = candidate_skill.get("description", op.description)
                op.instruction_template = candidate_skill.get(
                    "instruction_template", op.instruction_template
                )
            else:
                # Target not found, treat as add_new
                action = "add_new"

        if action == "add_new":
            new_op = Operation(
                name=candidate_skill.get("name", "new_skill"),
                description=candidate_skill.get("description", ""),
                instruction_template=candidate_skill.get("instruction_template", ""),
                update_type=candidate_skill.get("update_type", "insert"),
            )
            temp_op_bank.operations[new_op.name] = new_op

        # Re-run bad cases with the modified operation bank
        rewards = []
        for case in bad_cases:
            reward = self._evaluate_single_case(case, temp_op_bank)
            rewards.append(reward)

        if not rewards:
            return 0.0
        return float(np.mean(rewards))

    def _evaluate_single_case(
        self,
        case: Dict,
        temp_op_bank: OperationBank,
    ) -> float:
        """
        Re-evaluate a single bad case with a modified operation bank.

        Full pipeline simulation:
        1. Rebuild memory bank from case snapshot
        2. Skill selection (using LLM selector with modified bank)
        3. Execution (apply selected skill to session text)
        4. Apply executor results to memory bank
        5. QA retrieval + answer generation using updated memory
        6. Compute real F1 score against ground truth

        Args:
            case: Bad case dict with:
                - question: str
                - ground_truth: str
                - session_text: str (conversation chunk)
                - memory_bank_snapshot: dict (serialized memory bank state)
                - retrieved_memories: list[str] (pre-retrieval results)
            temp_op_bank: Temporary operation bank with candidate skill applied

        Returns:
            Reward score (0-1) based on real QA F1
        """
        question = case.get("question", "")
        ground_truth = case.get("ground_truth", "")
        session_text = case.get("session_text", "")
        retrieved_memories = case.get("retrieved_memories", [])

        if not question or not ground_truth:
            return 0.0

        # --- Step 1: Rebuild memory bank from snapshot ---
        memory_snapshot = case.get("memory_bank_snapshot", None)
        temp_memory_bank = None
        if memory_snapshot:
            try:
                temp_memory_bank = MemoryBank.from_dict(memory_snapshot)
            except Exception as e:
                logger.debug(f"[GRPOReward] Cannot rebuild memory bank: {e}")

        # --- Step 2: Skill selection ---
        candidate_ops = list(temp_op_bank.operations.values())
        if not candidate_ops:
            return 0.0

        try:
            temp_selector = SkillSelector(
                llm_client=self.llm_client,
                operation_bank=temp_op_bank,
                max_skills=1,
            )
            selected_ops = temp_selector.select_skills(
                session_text=session_text or question,
                retrieved_memories=retrieved_memories,
                candidate_ops=candidate_ops,
            )
        except Exception as e:
            logger.warning(f"[GRPOReward] Skill selection failed: {e}")
            return 0.0

        # --- Step 3: Execute skill ---
        try:
            exec_prompt = self._build_eval_prompt(
                selected_ops, case, retrieved_memories
            )
            executor_response = self.llm_client.call(
                role="executor",
                prompt=exec_prompt,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"[GRPOReward] Executor call failed: {e}")
            return 0.0

        # --- Step 4: Apply executor results to memory bank ---
        if temp_memory_bank is not None:
            try:
                exec_results = self._parse_executor_response(executor_response)
                self._apply_results_to_memory(
                    exec_results, temp_memory_bank, retrieved_memories
                )
            except Exception as e:
                logger.debug(f"[GRPOReward] Apply results failed: {e}")

        # --- Step 5+6: QA evaluation with updated memory ---
        if self.config.reward_metric == "llm_judge":
            reward = self._judge_reward(question, ground_truth, executor_response)
        else:
            reward = self._compute_qa_f1_reward(
                question, ground_truth, temp_memory_bank, retrieved_memories
            )

        return reward

    def _parse_executor_response(self, response: str) -> list:
        """Parse executor LLM response into ExecutionResult objects.

        Self-contained parser (no dependency on Executor instance) that handles:
        - ACTION: INSERT/UPDATE/DELETE/NOOP format
        - Line-only action markers
        - JSON format responses

        Args:
            response: Raw LLM response string

        Returns:
            List of ExecutionResult objects
        """
        import re
        from src.executor import ExecutionResult

        if not response or not response.strip():
            return []

        text = response.replace("\r\n", "\n").strip()
        # Strip markdown code fences
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                inner = parts[1]
                if "\n" in inner:
                    first_line, rest = inner.split("\n", 1)
                    if first_line.strip().lower() in ("json", "text", ""):
                        text = rest.strip()
                    else:
                        text = inner.strip()

        results = []

        # --- Try ACTION: TYPE format ---
        action_pattern = re.compile(
            r'(?<!\w)ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b',
            re.IGNORECASE
        )
        action_matches = list(action_pattern.finditer(text))

        # --- Fallback: line-only action markers ---
        if not action_matches:
            line_action_pattern = re.compile(
                r'(?im)^(?:[-*]\s*)?(INSERT|UPDATE|DELETE|NOOP)\s*(?::|=|-)?\s*$'
            )
            action_matches = list(line_action_pattern.finditer(text))

        if action_matches:
            for i, match in enumerate(action_matches):
                block_start = match.start()
                block_end = (
                    action_matches[i + 1].start()
                    if i + 1 < len(action_matches)
                    else len(text)
                )
                block = text[block_start:block_end].strip()
                action_type = match.group(1).upper()

                if action_type == "NOOP":
                    results.append(ExecutionResult(
                        action_type="NOOP", success=True,
                        reasoning=self._extract_field(block, "REASONING")
                    ))
                elif action_type == "INSERT":
                    content = self._extract_field(
                        block, "MEMORY_ITEM", "MEMORY_CONTENT", "CONTENT"
                    )
                    if content:
                        results.append(ExecutionResult(
                            action_type="INSERT", success=True,
                            memory_content=content,
                            reasoning=self._extract_field(block, "REASONING")
                        ))
                elif action_type == "UPDATE":
                    idx = self._extract_index(block)
                    content = self._extract_field(
                        block, "UPDATED_MEMORY", "MEMORY_CONTENT", "CONTENT"
                    )
                    if content and idx >= 0:
                        results.append(ExecutionResult(
                            action_type="UPDATE", success=True,
                            memory_index=idx, memory_content=content,
                            reasoning=self._extract_field(block, "REASONING")
                        ))
                elif action_type == "DELETE":
                    idx = self._extract_index(block)
                    if idx >= 0:
                        results.append(ExecutionResult(
                            action_type="DELETE", success=True,
                            memory_index=idx,
                            reasoning=self._extract_field(block, "REASONING")
                        ))
            return results

        # --- Fallback: JSON format ---
        try:
            from json_repair import repair_json
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = repair_json(text[json_start:json_end])
                data = json.loads(json_str)
                items = []
                if isinstance(data, dict):
                    items = data.get("actions", [data])
                elif isinstance(data, list):
                    items = data

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    action = str(
                        item.get("action", item.get("ACTION", ""))
                    ).strip().upper()
                    if action == "INSERT":
                        content = str(
                            item.get("memory_item", item.get("MEMORY_ITEM", ""))
                        ).strip()
                        if content:
                            results.append(ExecutionResult(
                                action_type="INSERT", success=True,
                                memory_content=content,
                                reasoning=str(item.get("reasoning", ""))
                            ))
                    elif action == "UPDATE":
                        idx = int(item.get("memory_index", item.get("MEMORY_INDEX", -1)))
                        content = str(
                            item.get("updated_memory", item.get("UPDATED_MEMORY", ""))
                        ).strip()
                        if idx >= 0 and content:
                            results.append(ExecutionResult(
                                action_type="UPDATE", success=True,
                                memory_index=idx, memory_content=content,
                                reasoning=str(item.get("reasoning", ""))
                            ))
                    elif action == "DELETE":
                        idx = int(item.get("memory_index", item.get("MEMORY_INDEX", -1)))
                        if idx >= 0:
                            results.append(ExecutionResult(
                                action_type="DELETE", success=True,
                                memory_index=idx,
                                reasoning=str(item.get("reasoning", ""))
                            ))
                    elif action == "NOOP":
                        results.append(ExecutionResult(
                            action_type="NOOP", success=True,
                            reasoning=str(item.get("reasoning", ""))
                        ))
        except Exception:
            pass

        return results

    @staticmethod
    def _extract_field(block: str, *field_names: str) -> str:
        """Extract a field value from an action block by trying multiple field names."""
        import re
        for name in field_names:
            pattern = re.compile(
                rf'{name}\s*(?::|=|-)\s*(.+?)(?:\n[A-Z_]+\s*(?::|=|-)|\Z)',
                re.IGNORECASE | re.DOTALL
            )
            m = pattern.search(block)
            if m:
                return m.group(1).strip()
        return ""

    @staticmethod
    def _extract_index(block: str) -> int:
        """Extract MEMORY_INDEX from an action block."""
        import re
        pattern = re.compile(
            r'MEMORY_INDEX\s*(?::|=|-)\s*(\d+)', re.IGNORECASE
        )
        m = pattern.search(block)
        if m:
            return int(m.group(1))
        return -1

    def _apply_results_to_memory(
        self,
        exec_results: list,
        memory_bank: 'MemoryBank',
        retrieved_memories: list,
 ):
        """Apply parsed executor results to a temporary memory bank.

        Simplified version (no embedding computation) for GRPO reward evaluation.
        Uses zero vectors as placeholder embeddings since we only need content
        for the subsequent QA prompt building.
        """
        import numpy as np
        from src.executor import ExecutionResult

        # Determine embedding dimension from existing memories
        emb_dim = 768
        if memory_bank.memories and memory_bank.memories[0].embedding is not None:
            emb_dim = len(memory_bank.memories[0].embedding)

        for result in exec_results:
            if not isinstance(result, ExecutionResult) or not result.success:
                continue

            if result.action_type == "INSERT" and result.memory_content:
                memory_bank.add_memory(
                    content=result.memory_content,
                    embedding=np.zeros(emb_dim, dtype=np.float32),
                )
            elif result.action_type == "UPDATE" and result.memory_content:
                idx = result.memory_index
                if 0 <= idx < len(retrieved_memories):
                    target_content = retrieved_memories[idx]
                    for mi, mem in enumerate(memory_bank.memories):
                        if mem.content == target_content:
                            mem.content = result.memory_content
                            break
            elif result.action_type == "DELETE":
                idx = result.memory_index
                if 0 <= idx < len(retrieved_memories):
                    target_content = retrieved_memories[idx]
                    for mi, mem in enumerate(memory_bank.memories):
                        if mem.content == target_content:
                            del memory_bank.memories[mi]
                            break

    def _compute_qa_f1_reward(
        self,
        question: str,
        ground_truth: str,
        memory_bank: 'MemoryBank',
        fallback_memories: list,
    ) -> float:
        """Compute QA F1 reward by generating an answer using updated memories.

        Pipeline:
        1. Get context memories from updated memory bank
        2. Build QA prompt with memories as context
        3. Call LLM to generate answer
        4. Compute token-level F1 against ground truth

        Args:
            question: The evaluation question
            ground_truth: Expected answer
            memory_bank: Updated memory bank (or None)
            fallback_memories: Original retrieved memories (used if no memory bank)

        Returns:
            F1 score (0.0 - 1.0)
        """
        # Get context memories
        if memory_bank is not None and len(memory_bank.memories) > 0:
            context_memories = [m.content for m in memory_bank.memories[:20]]
        else:
            context_memories = fallback_memories or []

        # Build QA prompt
        if context_memories:
            context_str = "\n".join(
                f"{i+1}. {mem}" for i, mem in enumerate(context_memories)
            )
        else:
            context_str = "(No memories available)"

        qa_prompt = (
            f"Answer the following question based on the provided memory context.\n\n"
            f"Memory Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely:"
        )

        # Generate answer
        try:
            prediction = self.llm_client.call(
                role="executor",
                prompt=qa_prompt,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"[GRPOReward] QA generation failed: {e}")
            return 0.0

        # Compute F1
        return self._token_f1(prediction.strip(), ground_truth)

    def _build_eval_prompt(
        self,
        selected_ops: List[Operation],
        case: Dict,
        retrieved_memories: List[str],
    ) -> str:
        """Build evaluation prompt for the executor."""
        session_text = case.get("session_text", case.get("question", ""))
        mem_text = "\n".join(
            f"{i}. {m}" for i, m in enumerate(retrieved_memories)
        ) if retrieved_memories else "(No memories)"

        skill_desc = ""
        for op in selected_ops:
            skill_desc += f"[{op.name}] ({op.update_type.upper()}): {op.description}\n"
            skill_desc += f"Instructions: {op.instruction_template}\n\n"

        return (
            f"Apply the following skill to manage memory:\n\n"
            f"Skill:\n{skill_desc}\n"
            f"Input Text:\n{session_text}\n\n"
            f"Current Memories:\n{mem_text}\n\n"
            f"Output memory actions (INSERT/UPDATE/DELETE/NOOP):"
        )

    def _f1_reward(self, case: Dict, executor_response: str) -> float:
        """Compute F1-based reward (legacy fallback when full pipeline unavailable).

        Uses the executor response quality as a proxy signal:
        - If executor produced meaningful structured actions, award moderate reward
        - Penalize empty/NOOP-only responses
        - This is a fallback; prefer _compute_qa_f1_reward for real evaluation
        """
        baseline_f1 = case.get("f1_score", 0.0)
        if not executor_response or not executor_response.strip():
            return max(0.0, baseline_f1 - 0.1)

        has_insert = "INSERT" in executor_response.upper()
        has_update = "UPDATE" in executor_response.upper()
        has_delete = "DELETE" in executor_response.upper()
        has_noop = "NOOP" in executor_response.upper()

        if has_insert or has_update:
            return min(1.0, baseline_f1 + 0.15)
        elif has_delete:
            return min(1.0, baseline_f1 + 0.05)
        elif has_noop:
            return baseline_f1
        return baseline_f1

    @staticmethod
    def _token_f1(prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score between prediction and ground truth.

        Handles both single string and comma-separated multi-answer ground truths.
        """
        import re

        if not prediction or not ground_truth:
            return 0.0

        def _normalize(text: str) -> List[str]:
            text = text.lower().strip()
            text = re.sub(r'[^\w\s]', ' ', text)
            return text.split()

        # Handle comma-separated ground truths (multi-hop questions)
        if ',' in ground_truth:
            gt_parts = [g.strip() for g in ground_truth.split(',') if g.strip()]
            if len(gt_parts) > 1:
                scores = []
                for gt_part in gt_parts:
                    scores.append(GRPORewardComputer._single_f1(
                        _normalize(prediction), _normalize(gt_part)
                    ))
                return max(scores) if scores else 0.0

        pred_tokens = _normalize(prediction)
        gt_tokens = _normalize(ground_truth)
        return GRPORewardComputer._single_f1(pred_tokens, gt_tokens)

    @staticmethod
    def _single_f1(pred_tokens: list, gt_tokens: list) -> float:
        """Compute F1 between two token lists."""
        if not pred_tokens or not gt_tokens:
            return 0.0

        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    def _judge_reward(
        self, question: str, ground_truth: str, prediction: str
    ) -> float:
        """Use LLM Judge to score the QA response quality.

        5-point scale with few-shot examples for consistent scoring.
        Evaluates whether the prediction correctly answers the question
        compared to the ground truth.

        Returns:
            Score normalized to 0.0 - 1.0 range
        """
        try:
            judge_prompt = (
                "You are an evaluation judge. Score how well the Model Answer "
                "answers the Question compared to the Ground Truth.\n\n"
                "Scoring criteria (1-5):\n"
                "  5: Perfectly matches ground truth in meaning and key details\n"
                "  4: Mostly correct with minor omissions or extra info\n"
                "  3: Partially correct — captures some key facts but misses others\n"
                "  2: Tangentially related but largely incorrect\n"
                "  1: Completely wrong or irrelevant\n\n"
                "Examples:\n"
                "Q: What is Alice's job?\n"
                "Ground Truth: software engineer\n"
                "Model Answer: She works as a software engineer at Google.\n"
                "Score: 5\n\n"
                "Q: When did Bob move to NYC?\n"
                "Ground Truth: March 2023\n"
                "Model Answer: Bob moved to a new city recently.\n"
                "Score: 2\n\n"
                "---\n"
                f"Question: {question}\n"
                f"Ground Truth: {ground_truth}\n"
                f"Model Answer: {prediction}\n\n"
                "Score (1-5):"
            )
            response = self.llm_client.call(
                role="judge",
                prompt=judge_prompt,
                temperature=0.0,
            )
            # Parse 1-5 score
            import re
            match = re.search(r'\b([1-5])\b', response)
            if match:
                score = int(match.group(1))
                return (score - 1) / 4.0  #Normalize to 0.0-1.0
            # Fallback: check for decimal scores
            for val in ["1.0", "0.75", "0.5", "0.25", "0.0"]:
                if val in response:
                    return float(val)
            return 0.0
        except Exception:
            return 0.0


class GRPODataPreparer:
    """
    Prepares GRPO training data for OpenRLHF.

    Generates:
    - Prompts: Designer analysis prompts from bad cases
    - Responses: G candidate skill proposals (sampled from Designer)
    - Rewards: Computed by GRPORewardComputer

    Output format compatible with OpenRLHF's GRPO trainer.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        reward_computer: GRPORewardComputer,
        config: GRPOConfig,
    ):
        self.llm_client = llm_client
        self.reward_computer = reward_computer
        self.config = config

    def prepare_grpo_batch(
        self,
        bad_cases: List[Dict],
        operation_bank: OperationBank,
        analysis_prompt_template: str,
        refinement_prompt_template: str,
    ) -> List[GRPOSample]:
        """
        Prepare a batch of GRPO training samples.

        Process:
        1. Stage1: Analyze bad cases (greedy) -> analysis
        2. Stage2: Sample G candidate refinements (temp=0.7) -> responses
        3. Compute reward for each candidate -> rewards

        Args:
            bad_cases: Collected bad cases
            operation_bank: Current operation bank state
            analysis_prompt_template: Template for Stage1 analysis
            refinement_prompt_template: Template for Stage2 refinement

        Returns:
            List of GRPOSample ready for training
        """
        base_op_dict = operation_bank.to_dict()
        samples = []

        for case_batch in self._chunk_cases(bad_cases):
            # Stage1: Analysis (greedy)
            analysis_prompt = analysis_prompt_template.format(
                bad_cases=json.dumps(case_batch, ensure_ascii=False, indent=2)
            )
            try:
                analysis = self.llm_client.call(
                    role="designer",
                    prompt=analysis_prompt,
                    temperature=0.0,  # Greedy
                )
            except Exception as e:
                logger.warning(f"[GRPO] Stage1 analysis failed: {e}")
                continue

            # Stage2: Sample G candidates (temp=0.7)
            refinement_prompt = refinement_prompt_template.format(
                analysis=analysis,
                current_skills=json.dumps(
                    [op.to_dict() for op in operation_bank.operations.values()],
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            try:
                candidates = self.llm_client.call(
                    role="designer",
                    prompt=refinement_prompt,
                    temperature=self.config.temperature,
                    n=self.config.group_size,
                )
                if isinstance(candidates, str):
                    candidates = [candidates]
            except Exception as e:
                logger.warning(f"[GRPO] Stage2 sampling failed: {e}")
                continue

            # Compute rewards for each candidate
            rewards = []
            for candidate_text in candidates:
                parsed = self._parse_candidate(candidate_text)
                if parsed:
                    reward = self.reward_computer.compute_reward_for_candidate(
                        candidate_skill=parsed,
                        bad_cases=case_batch,
                        base_operation_bank_dict=base_op_dict,
                    )
                else:
                    reward = 0.0
                rewards.append(reward)

            sample = GRPOSample(
                prompt=refinement_prompt,
                responses=candidates,
                rewards=rewards,
                analysis=analysis,
            )
            samples.append(sample)

        return samples

    def _chunk_cases(self, cases: List[Dict], chunk_size: int = 5) -> List[List[Dict]]:
        """Split cases into chunks for batch processing."""
        return [cases[i:i + chunk_size] for i in range(0, len(cases), chunk_size)]

    def _parse_candidate(self, text: str) -> Optional[Dict]:
        """Parse a candidate skill proposal from LLM response.

        Supports nested JSON objects (e.g., instruction_template containing {placeholder}).
        Uses greedy bracket matching from the first '{' to find valid JSON.
        """
        import re

        # Strategy 1: Find outermost JSON object using bracket counting
        start_idx = text.find('{')
        if start_idx != -1:
            # Try progressively shorter substrings from the last '}' backwards
            last_brace = text.rfind('}')
            while last_brace >= start_idx:
                candidate_str = text[start_idx:last_brace + 1]
                try:
                    parsed = json.loads(candidate_str)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
                last_brace = text.rfind('}', start_idx, last_brace)

        # Strategy 2: Try json_repair if available
        try:
            from json_repair import repair_json
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, dict) and repaired:
                return repaired
        except (ImportError, Exception):
            pass

        # Strategy 3: Fallback regex extraction for key fields
        result = {}
        for key in ["name", "description", "instruction_template", "update_type", "action"]:
            # Match both simple strings and multi-line values
            pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)?"'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                result[key] = match.group(1).replace('\\"', '"').replace('\\n', '\n')

        return result if result else None

    def export_for_openrlhf(
        self, samples: List[GRPOSample], output_path: str
    ):
        """
        Export GRPO data in OpenRLHF-compatible format.

        Format: JSONL with fields:
        - prompt: str
        - responses: List[str]
        - rewards: List[float]
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "prompt": sample.prompt,
                    "responses": sample.responses,
                    "rewards": sample.rewards,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"[GRPO] Exported {len(samples)} samples to {output_path}")


class GRPOTrainingLoop:
    """
    Complete GRPO training orchestrator.

    Integrates:
    - CaseCollector (from Designer) for bad case collection
    - GRPODataPreparer for Stage1 analysis + Stage2 sampling
    - GRPORewardComputer for reward computation
    - OpenRLHF export for actual gradient updates

    Usage:
        loop = GRPOTrainingLoop(args, config, llm_client, operation_bank, ...)
        loop.run(bad_cases)
    """

    def __init__(
        self,
        args,
        config,  # GRPOConfig
        llm_client: LLMClient,
        operation_bank: OperationBank,
        skill_selector: Optional['SkillSelector'] = None,
        evaluator: Any = None,
        snapshot_manager: Any = None,
    ):
        self.args = args
        self.config = config
        self.llm_client = llm_client
        self.operation_bank = operation_bank
        self.skill_selector = skill_selector
        self.evaluator = evaluator
        self.snapshot_manager = snapshot_manager
        self.logger = logging.getLogger("AgenticMemory")

        # Initialize reward computer
        self.reward_computer = GRPORewardComputer(
            llm_client=llm_client,
            operation_bank=operation_bank,
            config=config,
            evaluator=evaluator,
        )

        # Initialize data preparer
        self.data_preparer = GRPODataPreparer(
            llm_client=llm_client,
            reward_computer=self.reward_computer,
            config=config,
        )

        # Training history
        self.iteration_history: List[Dict] = []
        self.best_avg_reward: float = 0.0
        self.no_improvement_count: int = 0

    def run(
        self,
        bad_cases: List[Dict],
        analysis_prompt_template: str,
        refinement_prompt_template: str,
        export_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run a single GRPO iteration.

        Args:
            bad_cases: Collected bad cases for this iteration
            analysis_prompt_template: Template for Stage1 analysis
            refinement_prompt_template: Template for Stage2 refinement
            export_dir: Optional directory to export OpenRLHF data

        Returns:
            Dict with iteration results:
            - samples: List[GRPOSample]
            - best_candidate: Dict (the best candidate from this iteration)
            - avg_reward: float
            - improved: bool
        """
        self.logger.info(
            f"[GRPOLoop] Starting iteration with {len(bad_cases)} bad cases"
        )

        # Prepare GRPO batch (Stage1 + Stage2 + Rewards)
        samples = self.data_preparer.prepare_grpo_batch(
            bad_cases=bad_cases,
            operation_bank=self.operation_bank,
            analysis_prompt_template=analysis_prompt_template,
            refinement_prompt_template=refinement_prompt_template,
        )

        if not samples:
            self.logger.warning("[GRPOLoop] No valid samples produced")
            return {
                "samples": [],
                "best_candidate": None,
                "avg_reward": 0.0,
                "improved": False,
            }

        # Find best candidate across all samples
        best_candidate_text = None
        best_reward = -float("inf")
        all_rewards = []

        for sample in samples:
            for i, reward in enumerate(sample.rewards):
                all_rewards.append(reward)
                if reward > best_reward:
                    best_reward = reward
                    best_candidate_text = sample.responses[i] if i < len(sample.responses) else None

        avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        # Check if improved
        improved = avg_reward > self.best_avg_reward
        if improved:
            self.best_avg_reward = avg_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Apply best candidate to operation bank if improved
        best_candidate_parsed = None
        if best_candidate_text and improved:
            best_candidate_parsed = self.data_preparer._parse_candidate(best_candidate_text)
            if best_candidate_parsed:
                self._apply_best_candidate(best_candidate_parsed)

        # Export for OpenRLHF if requested
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            iteration_num = len(self.iteration_history)
            export_path = os.path.join(export_dir, f"grpo_iter_{iteration_num:04d}.jsonl")
            self.data_preparer.export_for_openrlhf(samples, export_path)

        # Record iteration
        result = {
            "samples": samples,
            "best_candidate": best_candidate_parsed,
            "avg_reward": avg_reward,
            "best_reward": best_reward,
            "improved": improved,
            "num_samples": len(samples),
            "total_candidates": sum(len(s.responses) for s in samples),
        }
        self.iteration_history.append(result)

        self.logger.info(
            f"[GRPOLoop] Iteration complete: avg_reward={avg_reward:.4f}, "
            f"best_reward={best_reward:.4f}, improved={improved}"
        )

        return result

    def _apply_best_candidate(self, candidate: Dict):
        """Apply the best candidate skill to the operation bank."""
        action = candidate.get("action", "add_new")

        if action == "refine":
            target = candidate.get("target_skill", "")
            if target and target in self.operation_bank.operations:
                op = self.operation_bank.operations[target]
                if "description" in candidate:
                    op.description = candidate["description"]
                if "instruction_template" in candidate:
                    op.instruction_template = candidate["instruction_template"]
                self.logger.info(f"[GRPOLoop] Refined skill: {target}")
                return

        # Default: add new
        new_op = Operation(
            name=candidate.get("name", f"grpo_skill_{len(self.operation_bank.operations)}"),
            description=candidate.get("description", ""),
            instruction_template=candidate.get("instruction_template", ""),
            update_type=candidate.get("update_type", "insert"),
        )
        self.operation_bank.operations[new_op.name] = new_op
        self.logger.info(f"[GRPOLoop] Added new skill: {new_op.name}")

    def should_stop(self) -> bool:
        """Check if training should stop."""
        if len(self.iteration_history) >= self.config.max_iterations:
            self.logger.info("[GRPOLoop] Reached max iterations")
            return True
        if self.no_improvement_count >= self.config.early_stop_patience:
            self.logger.info(
                f"[GRPOLoop] Early stopping: no improvement for "
                f"{self.config.early_stop_patience} iterations"
            )
            return True
        return False

    def get_summary(self) -> Dict:
        """Get training summary."""
        return {
            "total_iterations": len(self.iteration_history),
            "best_avg_reward": self.best_avg_reward,
            "final_no_improvement_count": self.no_improvement_count,
            "reward_history": [h["avg_reward"] for h in self.iteration_history],
        }
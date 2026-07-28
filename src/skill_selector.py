"""
LLM Skill Selector: Replaces PPO Controller for skill selection.

Uses LLM (Qwen3.6-Plus) to select the most appropriate memory management
skill(s) from the operation bank, given the current session text and
retrieved memories context.
"""
import json
import re
import logging
from typing import List, Optional, Dict, Any

from src.llm_client import LLMClient
from src.operation_bank import OperationBank, Operation

logger = logging.getLogger("AgenticMemory")


SKILL_SELECTION_PROMPT = """\
You are a memory management skill selector. Given the input text chunk and retrieved memories, \
select the most appropriate skill(s) from the available skills list.

## Input Text Chunk
{session_text}

## Retrieved Memories (current memory bank state)
{retrieved_memories}

## Available Skills
{skills_list}

## Instructions
- Analyze the input text and existing memories to determine what memory operations are needed.
- Select 1-{max_skills} skill(s) that should be applied to this input.
- Consider: Does new information need to be stored? Do existing memories need updating? \
Is any stored information now outdated or contradicted?
- If no memory operation is needed (input is trivial/redundant),select "noop".

## Output Format
Return a JSON object with the selected skill names:
```json
{{"selected_skills": ["skill_name_1", "skill_name_2"]}}
```

Only use skill names from the Available Skills list above."""


class SkillSelector:
    """
    LLM-based skill selectorthat replaces the PPO Controller.

    Instead of learning a policy over operation embeddings,
    this module uses an LLM to directly select skills based on
    natural language understanding of the context.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        operation_bank: OperationBank,
        max_skills: int = 1,
        role: str = "selector",
    ):
        """
        Args:
            llm_client: Unified LLM client
            operation_bank: Operation bank containing available skills
            max_skills: Maximum number of skills to select per step
            role: LLM client role to use for API calls
        """
        self.llm_client = llm_client
        self.operation_bank = operation_bank
        self.max_skills = max_skills
        self.role = role

    def _build_skills_list(self, candidate_ops: List[Operation]) -> str:
        """Build a formatted skills list string for the prompt."""
        lines = []
        for i, op in enumerate(candidate_ops, 1):
            name = op.name
            desc = op.description
            update_type = op.update_type.upper()
            lines.append(f"{i}. [{name}] (Action: {update_type})")
            lines.append(f"   Description: {desc}")
            lines.append("")
        return "\n".join(lines)

    def _format_retrieved_memories(self, retrieved_memories: List[str]) -> str:
        """Format retrieved memories for the prompt."""
        if not retrieved_memories:
            return "(No existing memories retrieved)"
        return "\n".join(
            f"{i}. {mem}" for i, mem in enumerate(retrieved_memories)
        )

    def select_skills(
        self,
        session_text: str,
        retrieved_memories: List[str],
        candidate_ops: Optional[List[Operation]] = None,
    ) -> List[Operation]:
        """
        Select skill(s) for the given context.

        Args:
            session_text: Current text chunk being processed
            retrieved_memories: List of retrieved memory contents
            candidate_ops: Override candidate operations (default: from operation bank)

        Returns:
            List of selected Operation objects
        """
        if candidate_ops is None:
            candidate_ops = self.operation_bank.get_candidate_operations()

        if not candidate_ops:
            logger.warning("[SkillSelector] No candidate operations available")
            return []

        # Build prompt
        skills_list = self._build_skills_list(candidate_ops)
        mem_text = self._format_retrieved_memories(retrieved_memories)

        prompt = SKILL_SELECTION_PROMPT.format(
            session_text=session_text,
            retrieved_memories=mem_text,
            skills_list=skills_list,
            max_skills=self.max_skills,
        )

        # Call LLM
        try:
            response = self.llm_client.call(
                role=self.role,
                prompt=prompt,
                temperature=0.0,  # Deterministic selection
            )
        except Exception as e:
            logger.warning(f"[SkillSelector] LLM call failed: {e}")
            # Fallback: return first non-noop operation
            for op in candidate_ops:
                if op.update_type.lower() != "noop":
                    return [op]
            return candidate_ops[:1]

        # Parse response
        selected_names = self._parse_response(response)

        # Map names to operations
        name_to_op = {op.name.lower(): op for op in candidate_ops}
        selected_ops = []
        for name in selected_names:
            op = name_to_op.get(name.lower())
            if op is not None:
                selected_ops.append(op)

        # Fallback if parsing failed or no valid ops found
        if not selected_ops:
            logger.warning(
                f"[SkillSelector] Could not parse valid skills from response: {response[:200]}"
            )
            # Default to first operation
            selected_ops = candidate_ops[:1]

        return selected_ops[:self.max_skills]

    def _parse_response(self, response: str) -> List[str]:
        """Parse LLM response to extract selected skill names."""
        # Try JSON parsing first
        try:
            # Extract JSON from response (may be wrapped in markdown code block)
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                skills = data.get("selected_skills", [])
                if isinstance(skills, list):
                    return [str(s).strip("[]") for s in skills]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to extract skill names from text
        # Look for quoted strings or names after "select" keywords
        names = re.findall(r'"([^"]+)"', response)
        if names:
            return names

        # Last resort: look for lines that match operation names
        lines = response.strip().split("\n")
        results = []
        for line in lines:
            line = line.strip().strip("-").strip("*").strip()
            if line:
                results.append(line)
        return results[:self.max_skills]
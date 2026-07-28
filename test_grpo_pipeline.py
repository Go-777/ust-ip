#!/usr/bin/env python
"""
Dry-run test for GRPO reward pipeline logic.
Mocks LLM calls to verify code paths without real API usage.
"""
import sys
import json
import numpy as np
from unittest.mock import MagicMock, patch

# Ensure src is importable
sys.path.insert(0, '.')


def test_parse_executor_response():
    """Test _parse_executor_response with various formats."""
    from src.grpo_trainer import GRPORewardComputer
    from src.executor import ExecutionResult

    # Create a minimal GRPORewardComputer with mocked dependencies
    mock_client = MagicMock()
    mock_config = MagicMock()
    mock_config.reward_metric = "f1"

    computer = GRPORewardComputer.__new__(GRPORewardComputer)
    computer.llm_client = mock_client
    computer.config = mock_config

    # Test 1: ACTION: INSERT format
    response1 = """ACTION: INSERT
MEMORY_ITEM: Alice loves hiking in the mountains
REASONING: New fact about Alice's hobby"""

    results = computer._parse_executor_response(response1)
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0].action_type == "INSERT"
    assert "Alice loves hiking" in results[0].memory_content
    print("  [PASS] ACTION: INSERT format")

    # Test 2: ACTION: UPDATE format
    response2 = """ACTION: UPDATE
MEMORY_INDEX: 2
UPDATED_MEMORY: Bob now works at Google (previously at Meta)
REASONING: Updated employment info"""

    results = computer._parse_executor_response(response2)
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0].action_type == "UPDATE"
    assert results[0].memory_index == 2
    assert "Google" in results[0].memory_content
    print("  [PASS] ACTION: UPDATE format")

    # Test 3: ACTION: DELETE format
    response3 = """ACTION: DELETE
MEMORY_INDEX: 0
REASONING: This memory is outdated"""

    results = computer._parse_executor_response(response3)
    assert len(results) == 1
    assert results[0].action_type == "DELETE"
    assert results[0].memory_index == 0
    print("  [PASS] ACTION: DELETE format")

    # Test 4: Multiple actions
    response4 = """ACTION: INSERT
MEMORY_ITEM: Carol is a vegetarian
REASONING: Diet preference

ACTION: DELETE
MEMORY_INDEX: 1
REASONING: Duplicate memory"""

    results = computer._parse_executor_response(response4)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0].action_type == "INSERT"
    assert results[1].action_type == "DELETE"
    print("  [PASS] Multiple actions")

    # Test 5: JSON format
    response5 = json.dumps({
        "actions": [
            {"action": "INSERT", "memory_item": "Dave likes coffee", "reasoning": "preference"},
            {"action": "UPDATE", "memory_index": 0, "updated_memory": "Updated content", "reasoning": "fix"}
        ]
    })

    results = computer._parse_executor_response(response5)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0].action_type == "INSERT"
    assert results[1].action_type == "UPDATE"
    print("  [PASS] JSON format")

    # Test 6: Empty/invalid response
    results = computer._parse_executor_response("")
    assert len(results) == 0
    results = computer._parse_executor_response("I don't know what to do")
    assert len(results) == 0
    print("  [PASS] Empty/invalid response")

    # Test 7: NOOP
    response7 = "ACTION: NOOP\nREASONING: No action needed"
    results = computer._parse_executor_response(response7)
    assert len(results) == 1
    assert results[0].action_type == "NOOP"
    print("  [PASS] NOOP")


def test_apply_results_to_memory():
    """Test _apply_results_to_memory logic."""
    from src.grpo_trainer import GRPORewardComputer
    from src.memory_bank import MemoryBank, MemoryItem
    from src.executor import ExecutionResult

    computer = GRPORewardComputer.__new__(GRPORewardComputer)

    # Build a minimal memory bank
    bank = MemoryBank.__new__(MemoryBank)
    bank.memories = [
        MemoryItem(content="Alice is 30 years old", embedding=np.zeros(768)),
        MemoryItem(content="Bob works at Meta", embedding=np.zeros(768)),
        MemoryItem(content="Carol lives in NYC", embedding=np.zeros(768)),
    ]
    bank.timestep = 1
    bank.state_encoder = None

    retrieved_memories = ["Alice is 30 years old", "Bob works at Meta", "Carol lives in NYC"]

    # Test INSERT
    exec_results = [
        ExecutionResult(action_type="INSERT", success=True,
                       memory_content="Dave likes hiking", reasoning="new")
    ]
    computer._apply_results_to_memory(exec_results, bank, retrieved_memories)
    assert len(bank.memories) == 4
    assert bank.memories[3].content == "Dave likes hiking"
    print("  [PASS] INSERT adds memory")

    # Test UPDATE
    exec_results = [
        ExecutionResult(action_type="UPDATE", success=True,
                       memory_index=1, memory_content="Bob works at Google", reasoning="update")
    ]
    computer._apply_results_to_memory(exec_results, bank, retrieved_memories)
    # Find the updated memory
    found = any(m.content == "Bob works at Google" for m in bank.memories)
    assert found, "UPDATE didn't change memory content"
    print("  [PASS] UPDATE modifies memory")

    # Test DELETE
    exec_results = [
        ExecutionResult(action_type="DELETE", success=True,
                       memory_index=2, reasoning="outdated")
    ]
    computer._apply_results_to_memory(exec_results, bank, retrieved_memories)
    contents = [m.content for m in bank.memories]
    assert "Carol lives in NYC" not in contents, "DELETE didn't remove memory"
    print("  [PASS] DELETE removes memory")

    # Test failed result (should be skipped)
    prev_count = len(bank.memories)
    exec_results = [
        ExecutionResult(action_type="INSERT", success=False,
                       memory_content="Should not be added", reasoning="fail")
    ]
    computer._apply_results_to_memory(exec_results, bank, retrieved_memories)
    assert len(bank.memories) == prev_count
    print("  [PASS] Failed results skipped")


def test_token_f1():
    """Test _token_f1 static method."""
    from src.grpo_trainer import GRPORewardComputer

    #Exact match
    f1 = GRPORewardComputer._token_f1("Alice is 30", "Alice is 30")
    assert f1 == 1.0, f"Expected 1.0, got {f1}"
    print("  [PASS] Exact match F1=1.0")

    # Partial match
    f1 = GRPORewardComputer._token_f1("Alice is 30 years old", "Alice is 30")
    assert 0.5 < f1 < 1.0, f"Expected partial F1, got {f1}"
    print(f"  [PASS] Partial match F1={f1:.3f}")

    # No match
    f1 = GRPORewardComputer._token_f1("xyz abc", "hello world")
    assert f1 == 0.0, f"Expected 0.0, got {f1}"
    print("  [PASS] No match F1=0.0")

    # Multi-hop (comma separated ground truth)
    f1 = GRPORewardComputer._token_f1("Alice and Bob", "Alice, Bob")
    assert f1 >= 0.4, f"Expected >=0.4 for multi-hop, got {f1}"
    print(f"  [PASS] Multi-hop F1={f1:.3f}")

    # Empty strings
    f1 = GRPORewardComputer._token_f1("", "something")
    assert f1 == 0.0
    f1 = GRPORewardComputer._token_f1("something", "")
    assert f1 == 0.0
    print("  [PASS] Empty string F1=0.0")


def test_compute_qa_f1_reward():
    """Test _compute_qa_f1_reward with mocked LLM."""
    from src.grpo_trainer import GRPORewardComputer
    from src.memory_bank import MemoryBank, MemoryItem

    mock_client = MagicMock()
    # Mock LLM to return a QA answer
    mock_client.call.return_value = "Alice is 30 years old"

    computer = GRPORewardComputer.__new__(GRPORewardComputer)
    computer.llm_client = mock_client

    # Build memory bank
    bank = MemoryBank.__new__(MemoryBank)
    bank.memories = [
        MemoryItem(content="Alice is 30 years old", embedding=np.zeros(768)),
        MemoryItem(content="Bob works at Google", embedding=np.zeros(768)),
    ]
    bank.state_encoder = None

    reward = computer._compute_qa_f1_reward(
        question="How old is Alice?",
        ground_truth="30 years old",
        memory_bank=bank,
        fallback_memories=["Alice is 30 years old"]
    )
    assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"
    assert reward > 0.5, f"Expected high F1 reward, got {reward}"
    print(f"  [PASS] QA F1 reward={reward:.3f}")

    # Test with None memory bank (uses fallback)
    reward2 = computer._compute_qa_f1_reward(
        question="How old is Alice?",
        ground_truth="30 years old",
        memory_bank=None,
        fallback_memories=["Alice is 30 years old"]
    )
    assert 0.0 <= reward2 <= 1.0
    print(f"  [PASS] Fallback memories reward={reward2:.3f}")


def test_evaluate_single_case():
    """Test _evaluate_single_case end-to-end with mocked LLM."""
    from src.grpo_trainer import GRPORewardComputer
    from src.operation_bank import OperationBank, Operation
    from src.memory_bank import MemoryBank, MemoryItem

    mock_client = MagicMock()
    # Selector returns skill choice, Executor returns action, QA returns answer
    mock_client.call.side_effect = [
        "I select skill 1: extract_preferences",  # selector
        "ACTION: INSERT\nMEMORY_ITEM: Alice is 30 years old\nREASONING: age fact",  # executor
        "Alice is 30 years old",  # QA response
    ]

    mock_config = MagicMock()
    mock_config.reward_metric = "f1"

    computer = GRPORewardComputer.__new__(GRPORewardComputer)
    computer.llm_client = mock_client
    computer.config = mock_config

    # Create operation bank with one skill
    op_bank = OperationBank(encoder=None, max_ops=20, skip_noop=False)
    test_op = Operation(
        name="extract_preferences",
        description="Extract user preferences from conversation",
        instruction_template="Extract preferences from: {text}",
        update_type="INSERT",
    )
    op_bank.operations["extract_preferences"] = test_op

    # Build test case
    memory_data = {
        "retriever_name": "contriever",
        "top_k": 5,
        "timestep": 1,
        "memories": [
            {
                "content": "Some old memory",
                "embedding": [0.0] * 768,
                "metadata": {},
                "access_count": 0,
                "last_accessed": 0,
                "created_at": 0,
                "content_history": [],
                "operation_history": [],
            }
        ]
    }

    case = {
        "question": "How old is Alice?",
        "ground_truth": "30 years old",
        "session_text": "User: I'm Alice, I turned 30 last week.",
        "memory_bank_snapshot": memory_data,
        "retrieved_memories": ["Some old memory"],
    }

    reward = computer._evaluate_single_case(case, op_bank)
    assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"
    print(f"  [PASS] Full pipeline reward={reward:.3f}")
    # Verify LLM was called 3 times (selector, executor, QA)
    assert mock_client.call.call_count == 3, \
        f"Expected 3 LLM calls, got {mock_client.call.call_count}"
    print(f"  [PASS] LLM called 3 times as expected")


def test_judge_reward():
    """Test _judge_reward with mocked LLM."""
    from src.grpo_trainer import GRPORewardComputer

    mock_client = MagicMock()
    mock_client.call.return_value = "4"  # Score of 4 out of 5

    mock_config = MagicMock()
    mock_config.reward_metric = "llm_judge"

    computer = GRPORewardComputer.__new__(GRPORewardComputer)
    computer.llm_client = mock_client
    computer.config = mock_config

    reward = computer._judge_reward(
        question="How old is Alice?",
        ground_truth="30 years old",
        prediction="Alice is about 30 years old"
    )
    # Score 4 → normalized to (4-1)/4 = 0.75
    assert 0.7 <= reward <= 0.8, f"Expected ~0.75, got {reward}"
    print(f"  [PASS] Judge reward={reward:.3f} (score=4 → 0.75)")

    # Test invalid response
    mock_client.call.return_value = "This answer is pretty good"
    reward = computer._judge_reward(
        question="How old is Alice?",
        ground_truth="30 years old",
        prediction="She is 30"
    )
    assert 0.0 <= reward <= 1.0
    print(f"  [PASS] Invalid judge response handled, reward={reward:.3f}")


def test_imports():
    """Verify all critical imports work."""
    from src.grpo_trainer import GRPORewardComputer, GRPODataPreparer
    from src.operation_bank import OperationBank, Operation
    from src.memory_bank import MemoryBank, MemoryItem
    from src.skill_selector import SkillSelector
    from src.executor import ExecutionResult
    from src.llm_client import LLMClient
    print("  [PASS] All imports successful")


if __name__ == "__main__":
    print("\n=== GRPO Pipeline Dry-Run Tests ===\n")

    print("[1] Testing imports...")
    test_imports()

    print("\n[2] Testing _parse_executor_response...")
    test_parse_executor_response()

    print("\n[3] Testing _apply_results_to_memory...")
    test_apply_results_to_memory()

    print("\n[4] Testing _token_f1...")
    test_token_f1()

    print("\n[5] Testing _compute_qa_f1_reward...")
    test_compute_qa_f1_reward()

    print("\n[6] Testing _evaluate_single_case (full pipeline)...")
    test_evaluate_single_case()

    print("\n[7] Testing _judge_reward...")
    test_judge_reward()

    print("\n=== ALL TESTS PASSED ===\n")
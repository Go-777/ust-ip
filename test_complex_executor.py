import sys, re, os
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/tiktoken_cache"
sys.path.insert(0, "/root/MemSkill")
from llm_utils import get_llm_response_via_api

prompt = (
    "You are a memory management executor. Apply the selected skills to the input text\n"
    "chunk and retrieved memories, then output memory actions.\n\n"
    "Input Text Chunk:\n"
    "User: Remember my brother from Chicago? He left yesterday after 3 weeks. We visited Golden Gate Park and Napa Valley. Also I got promoted to Senior Data Scientist with a 30 percent raise!\n"
    "Assistant: Congrats on the promotion! How are you feeling?\n"
    "User: Excited! Looking at apartments in Pacific Heights. Also Max has been hiding under the bed, thinking of taking him to the vet.\n\n"
    "Retrieved Memories (0-based index):\n"
    "0. User has a dog named Max who enjoys parks\n"
    "1. User lives in Mission District, small apartment\n"
    "2. User works as a Data Scientist\n"
    "3. User has a brother in Chicago\n"
    "4. User enjoys outdoor activities\n"
    "5. User favorite restaurant is Kin Khao\n"
    "6. User goes running in Dolores Park\n"
    "7. User is learning guitar\n"
    "8. User grew up in Boston\n"
    "9. User went to MIT\n"
    "10. User enjoys cooking Italian food\n"
    "11. User is reading ML ethics book\n"
    "12. User has Saturday brunch with friends\n"
    "13. User planning Japan trip next spring\n"
    "14. User drives Tesla Model 3\n"
    "15. User watches Arsenal\n"
    "16. User allergic to shellfish\n"
    "17. User birthday March 15\n"
    "18. User moved from East Coast recently\n"
    "19. User considering getting a cat\n\n"
    "Selected Skills:\n"
    "[Skill 1] delete\n"
    "Description: Delete outdated or superseded memory items.\n"
    "Allowed action: DELETE\n\n"
    "[Skill 2] insert\n"
    "Description: Insert new memory items from conversation.\n"
    "Allowed action: INSERT\n\n"
    "[Skill 3] update\n"
    "Description: Update existing memory with new info.\n"
    "Allowed action: UPDATE\n\n"
    "Guidelines:\n"
    "- Apply any skill as needed; a skill may be used multiple times.\n"
    "- Only use action types supported by the selected skills.\n"
    "- MEMORY_INDEX is 0-based.\n"
    "- Output only action blocks in format below.\n"
    "- Do not include explanations.\n"
    "Output format:\n\n"
    "INSERT block:\n"
    "ACTION: INSERT\n"
    "MEMORY_ITEM: <content>\n\n"
    "UPDATE block:\n"
    "ACTION: UPDATE\n"
    "MEMORY_INDEX: <index>\n"
    "UPDATED_MEMORY: <content>\n\n"
    "DELETE block:\n"
    "ACTION: DELETE\n"
    "MEMORY_INDEX: <index>\n"
)

response, pt, ct = get_llm_response_via_api(
    prompt=prompt,
    LLM_MODEL="maas-token-latest",
    base_url="https://localhost:19443/tokenPlan/openai/v1",
    api_key="pk-8c8b5a47-95e1-4609-8272-35ccc50934f8",
    MAX_TOKENS=2048,
    TAU=0.0,
    MAX_TRIALS=3,
    TIME_GAP=3,
)
print(f"Tokens: prompt={pt}, completion={ct}")
print("=== RAW (repr first 1000) ===")
print(repr(response[:1000]))
print("\n=== FORMATTED ===")
print(response)
action_pattern = re.compile(r'(?<!\w)ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b', re.IGNORECASE)
matches = list(action_pattern.finditer(response))
print(f"\n=== ACTION matches: {len(matches)} ===")
for m in matches:
    print(f"  {m.group()} at pos {m.start()}")
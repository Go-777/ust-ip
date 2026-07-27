"""
Unified LLM Client: Multi-model, multi-role API calling layer.

Supports:
- Multiple model roles (selector, executor, judge, designer)
- DashScope (Alibaba Cloud Bailian) with OpenAI-compatible API
- Local model inference via vLLM/OpenAI-compatible server
- Round-robin API key rotation
- Retry with exponential backoff
"""
import os
import time
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

import openai
import httpx

logger = logging.getLogger("AgenticMemory")


@dataclass
class ModelConfig:
    """Configuration for a single model role."""
    model_name: str  # API model identifier (e.g., "qwen3.6-plus")
    base_url: str = ""  # API base URL
    api_keys: List[str] = field(default_factory=list)
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    max_retries: int = 3
    timeout: int = 120
    retry_delay: float = 3.0

    @classmethod
    def from_env(cls, role: str) -> "ModelConfig":
        """Create ModelConfig from environment variables.

        Env vars follow the pattern: MEMSKILL_{ROLE}_{PARAM}
        e.g., MEMSKILL_SELECTOR_MODEL, MEMSKILL_SELECTOR_API_KEY
        """
        prefix = f"MEMSKILL_{role.upper()}"
        api_key_str = os.environ.get(f"{prefix}_API_KEY", "")
        api_keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
        return cls(
            model_name=os.environ.get(f"{prefix}_MODEL", ""),
            base_url=os.environ.get(f"{prefix}_BASE_URL", ""),
            api_keys=api_keys,
            max_tokens=int(os.environ.get(f"{prefix}_MAX_TOKENS", "2048")),
            temperature=float(os.environ.get(f"{prefix}_TEMPERATURE", "0.0")),
        )


# Default configurations for each role based on confirmed selection
# JD Cloud tokenPlan service (port 8443 for SSH tunnel on server, standard 443 locally)
DEFAULT_API_BASE_URL = os.environ.get(
    "TOKENPLAN_API_BASE",
    "https://modelservice.jdcloud.com:8443/tokenPlan/openai/v1"
)
# Legacy DashScope URL (requires public internet - not available on JD servers)
LEGACY_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

ROLE_DEFAULTS = {
    "selector": ModelConfig(
        model_name="maas-token-latest",
        base_url=DEFAULT_API_BASE_URL,
        max_tokens=2048,
        temperature=0.0,
    ),
    "executor": ModelConfig(
        model_name="maas-token-latest",
        base_url=DEFAULT_API_BASE_URL,
        max_tokens=2048,
        temperature=0.0,
    ),
    "judge": ModelConfig(
        model_name="maas-token-latest",
        base_url=DEFAULT_API_BASE_URL,
        max_tokens=1024,
        temperature=0.0,
    ),
    "designer": ModelConfig(
        model_name="maas-token-latest",
        base_url=DEFAULT_API_BASE_URL,
        max_tokens=4096,
        temperature=0.0,
    ),
}


class LLMClient:
    """
    Unified LLM client supporting multiple model roles.

    Usage:
        client = LLMClient(api_keys=["sk-xxx"])
        # Use role-based calling
        response = client.call("selector", prompt="Select a skill...")
        response = client.call("executor", prompt="Execute...")
        response = client.call("judge", messages=[...])

        # Override temperature for GRPO sampling
        responses = client.call_batch("designer", prompts=[...], temperature=0.7, n=8)
    """

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        role_configs: Optional[Dict[str, ModelConfig]] = None,
    ):
        """
        Args:
            api_keys: Default API keys (used if role config doesn't specify keys)
            role_configs: Dict mapping role name -> ModelConfig. Missing roles use defaults.
        """
        self._default_api_keys = api_keys or []
        self._role_configs: Dict[str, ModelConfig] = {}

        # Initialize with defaults
        for role, default_cfg in ROLE_DEFAULTS.items():
            self._role_configs[role] = ModelConfig(
                model_name=default_cfg.model_name,
                base_url=default_cfg.base_url,
                api_keys=default_cfg.api_keys.copy() if default_cfg.api_keys else [],
                max_tokens=default_cfg.max_tokens,
                temperature=default_cfg.temperature,
                top_p=default_cfg.top_p,
                max_retries=default_cfg.max_retries,
                timeout=default_cfg.timeout,
                retry_delay=default_cfg.retry_delay,
            )

        # Override with user-provided configs
        if role_configs:
            for role, cfg in role_configs.items():
                self._role_configs[role] = cfg

        # Client cache (keyed by (base_url, api_key))
        self._client_cache: Dict[str, openai.OpenAI] = {}
        self._key_indices: Dict[str, int] = {}
        self._lock = threading.Lock()

    def get_config(self, role: str) -> ModelConfig:
        """Get the config for a role."""
        if role not in self._role_configs:
            raise ValueError(
                f"Unknown role '{role}'. Available: {list(self._role_configs.keys())}"
            )
        return self._role_configs[role]

    def set_config(self, role: str, config: ModelConfig):
        """Set/update config for a role."""
        self._role_configs[role] = config

    def _get_api_keys(self, role: str) -> List[str]:
        """Get API keys for a role, falling back to default keys."""
        cfg = self._role_configs.get(role)
        if cfg and cfg.api_keys:
            return cfg.api_keys
        return self._default_api_keys

    def _get_client(self, role: str) -> openai.OpenAI:
        """Get or create an OpenAI client with round-robin key rotation."""
        cfg = self._role_configs[role]
        api_keys = self._get_api_keys(role)

        if not api_keys:
            raise ValueError(
                f"No API keys available for role '{role}'. "
                f"Set via LLMClient(api_keys=[...]) or role_configs."
            )

        with self._lock:
            # Round-robin key selection
            idx = self._key_indices.get(role, 0)
            key = api_keys[idx % len(api_keys)]
            self._key_indices[role] = idx + 1

            # Cache key
            cache_key = f"{cfg.base_url}|{key}"
            if cache_key not in self._client_cache:
                # When using SSH tunnel (localhost), inject Host header for proper routing
                http_headers = {}
                if "localhost" in cfg.base_url or "127.0.0.1" in cfg.base_url:
                    http_headers["Host"] = "modelservice.jdcloud.com"
                self._client_cache[cache_key] = openai.OpenAI(
                    base_url=cfg.base_url,
                    api_key=key,
                    max_retries=1,  # We handle retries ourselves
                    timeout=cfg.timeout,
                    http_client=httpx.Client(
                        verify=False,
                        headers=http_headers if http_headers else None,
                    ),
                )
            return self._client_cache[cache_key]

    def call(
        self,
        role: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict] = None,
        n: int = 1,
    ) -> Union[str, List[str]]:
        """
        Call LLM for a specific role.

        Args:
            role: Model role (selector, executor, judge, designer)
            prompt: Simple string prompt (converted to user message)
            messages: Full message list (overrides prompt)
            temperature: Override role's default temperature
            max_tokens: Override role's default max_tokens
            top_p: Override role's default top_p
            response_format: Optional response format constraint
            n: Number of completions to generate

        Returns:
            Single response string if n=1, list of strings if n>1
        """
        cfg = self.get_config(role)

        # Build messages
        if messages is None:
            if prompt is None:
                raise ValueError("Either 'prompt' or 'messages' must be provided")
            messages = [{"role": "user", "content": prompt}]

        # Resolve parameters
        temp = temperature if temperature is not None else cfg.temperature
        max_tok = max_tokens if max_tokens is not None else cfg.max_tokens
        tp = top_p if top_p is not None else cfg.top_p

        # Retry loop
        last_error = None
        for attempt in range(cfg.max_retries):
            client = self._get_client(role)
            try:
                api_params = {
                    "model": cfg.model_name,
                    "messages": messages,
                    "temperature": temp,
                    "top_p": tp,
                    "max_tokens": max_tok,
                }
                if response_format is not None:
                    api_params["response_format"] = response_format

                if n == 1:
                    # Single completion
                    completion = client.chat.completions.create(**api_params)
                    return completion.choices[0].message.content
                else:
                    # Multiple completions: loop n times (DashScope doesn't support n>1 for most models)
                    results = []
                    for _i in range(n):
                        completion = client.chat.completions.create(**api_params)
                        results.append(completion.choices[0].message.content)
                    return results

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "timeout" in error_str:
                    logger.warning(
                        f"[LLMClient] {role} timeout (attempt {attempt + 1})"
                    )
                    break  # Don't retry timeouts
                logger.warning(
                    f"[LLMClient] {role} error (attempt {attempt + 1}/{cfg.max_retries}): {e}"
                )
                if attempt < cfg.max_retries - 1:
                    time.sleep(cfg.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"[LLMClient] {role} failed after {cfg.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def call_with_usage(
        self,
        role: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict] = None,
    ) -> tuple:
        """
        Call LLM and return (response, prompt_tokens, completion_tokens).
        Compatible with existing get_llm_response_via_api interface.
        """
        cfg = self.get_config(role)
        if messages is None:
            if prompt is None:
                raise ValueError("Either 'prompt' or 'messages' must be provided")
            messages = [{"role": "user", "content": prompt}]

        temp = temperature if temperature is not None else cfg.temperature
        max_tok = max_tokens if max_tokens is not None else cfg.max_tokens
        tp = top_p if top_p is not None else cfg.top_p

        last_error = None
        for attempt in range(cfg.max_retries):
            client = self._get_client(role)
            try:
                api_params = {
                    "model": cfg.model_name,
                    "messages": messages,
                    "temperature": temp,
                    "top_p": tp,
                    "max_tokens": max_tok,
                }
                if response_format is not None:
                    api_params["response_format"] = response_format

                completion = client.chat.completions.create(**api_params)
                content = completion.choices[0].message.content
                usage = completion.usage
                return (
                    content,
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )

            except Exception as e:
                last_error = e
                if "timeout" in str(e).lower():
                    break
                logger.warning(
                    f"[LLMClient] {role} error (attempt {attempt + 1}): {e}"
                )
                if attempt < cfg.max_retries - 1:
                    time.sleep(cfg.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"[LLMClient] {role} call_with_usage failed. Last error: {last_error}"
        )

    def call_batch(
        self,
        role: str,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Call LLM for multiple prompts sequentially.
        For GRPO sampling, use call() with n>1 on a single prompt instead.

        Args:
            role: Model role
            prompts: List of prompts
            temperature: Override temperature
            max_tokens: Override max_tokens

        Returns:
            List of response strings
        """
        results = []
        for prompt in prompts:
            try:
                resp = self.call(
                    role=role,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                results.append(resp)
            except Exception as e:
                logger.warning(f"[LLMClient] batch call failed for prompt: {e}")
                results.append("")
        return results


def create_llm_client_from_args(args) -> LLMClient:
    """
    Create LLMClient from command-line args.
    Maps args fields to role-based configs.
    """
    # Get API keys from args
    api_keys = getattr(args, "api_key", [])
    if isinstance(api_keys, str):
        api_keys = [api_keys]

    # Base URL
    base_url = getattr(args, "api_base", DEFAULT_API_BASE_URL)
    if not base_url or base_url == "[YOUR_API_BASE]":
        base_url = DEFAULT_API_BASE_URL

    # Build role configs from args
    # Default model: use --model arg as fallback for all roles
    default_model = getattr(args, "model", "maas-token-latest")

    def _resolve_model(role_attr: str) -> str:
        """Get role-specific model or fallback to --model."""
        val = getattr(args, role_attr, None)
        # If role-specific model is the hardcoded default and user provided --model, use --model
        if not val or val == "maas-token-latest":
            return default_model
        return val

    role_configs = {}

    # Selector config
    selector_model = _resolve_model("selector_model")
    selector_base_url = getattr(args, "selector_api_base", None) or base_url
    role_configs["selector"] = ModelConfig(
        model_name=selector_model,
        base_url=selector_base_url,
        api_keys=api_keys,
        max_tokens=getattr(args, "max_new_tokens", 2048),
        temperature=0.0,
    )

    # Executor config
    executor_model = _resolve_model("executor_model")
    executor_base_url = getattr(args, "executor_api_base", None) or base_url
    role_configs["executor"] = ModelConfig(
        model_name=executor_model,
        base_url=executor_base_url,
        api_keys=api_keys,
        max_tokens=getattr(args, "max_new_tokens", 2048),
        temperature=getattr(args, "temperature", 0.0),
    )

    # Judge config: prefer --judge-model over legacy --llm-judge-model
    judge_model_val = getattr(args, "judge_model", None)
    if not judge_model_val or judge_model_val == "maas-token-latest":
        judge_model_val = _resolve_model("llm_judge_model")
    else:
        judge_model_val = judge_model_val  # explicit --judge-model takes priority
    judge_base_url = getattr(args, "judge_api_base", None) or base_url
    role_configs["judge"] = ModelConfig(
        model_name=judge_model_val,
        base_url=judge_base_url,
        api_keys=api_keys,
        max_tokens=1024,
        temperature=0.0,
    )

    # Designer config (also uses gateway API now)
    designer_model = _resolve_model("designer_model")
    designer_base_url = getattr(args, "designer_api_base", None) or base_url
    designer_api_keys = getattr(args, "designer_api_key", None)
    if designer_api_keys is None:
        designer_api_keys = api_keys if api_keys else ["EMPTY"]
    elif isinstance(designer_api_keys, str):
        designer_api_keys = [designer_api_keys]
    role_configs["designer"] = ModelConfig(
        model_name=designer_model,
        base_url=designer_base_url,
        api_keys=designer_api_keys,
        max_tokens=4096,
        temperature=0.0,  # Greedy for Stage1; override for Stage2 sampling
    )

    return LLMClient(api_keys=api_keys, role_configs=role_configs)
"""Local LLM server client (Ollama, vLLM, LM Studio, llama-server).

All these backends expose an OpenAI-compatible REST API, so a single
provider implementation covers all of them.

Backend       Default URL                   Notes
-----------   ---------------------------   ------------------------------
Ollama        http://localhost:11434/v1     ollama serve
vLLM          http://localhost:8000/v1      vllm serve <model_id>
LM Studio     http://localhost:1234/v1      Start server from the GUI
llama-server  http://localhost:8080/v1      llama-server -m model.gguf
"""

import logging
import time
from dataclasses import dataclass

import httpx
from openai import OpenAI

from eleusis.llm.base import BaseLLMClient, LLMCallMetrics

logger = logging.getLogger(__name__)


@dataclass
class LocalMessage:
    content: str
    reasoning: str | None = None


@dataclass
class LocalChoice:
    message: LocalMessage
    finish_reason: str


class LocalClient(BaseLLMClient):
    """Client for local OpenAI-compatible LLM servers.

    Supports two reasoning formats for models that produce chain-of-thought:

    - ``"think_tags"``  – reasoning is embedded inline as ``<think>…</think>``
                          (e.g. DeepSeek-R1 fine-tunes, Qwen3-Thinking variants)
    - ``"none"``        – no reasoning extraction; answer tokens only

    Configuration example in ``models.yaml``::

        my-local-model:
          provider: local
          model_id: llama3.2:3b
          base_url: http://localhost:11434/v1
          reasoning_format: none
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "local",
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        seed: int | None = None,
        reasoning_format: str = "none",
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )
        self.base_url = base_url.rstrip("/")
        self.reasoning_format = reasoning_format

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(3600.0),
        )

    @property
    def provider_name(self) -> str:
        return "local"

    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[LocalChoice, LLMCallMetrics]:
        """Make a single API call with retry logic."""
        logger.debug(
            f"Calling local API ({self.base_url}) model={self.model_name}, "
            f"max_tokens={self.max_tokens}"
        )

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                api_kwargs: dict = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                if self.seed is not None:
                    api_kwargs["seed"] = self.seed

                completion = self.client.chat.completions.create(**api_kwargs)
                end_time = time.time()

                choice = completion.choices[0]
                raw_content = choice.message.content or ""
                reasoning_content: str | None = None
                answer_content: str = raw_content

                # Extract <think>…</think> reasoning when requested
                if self.reasoning_format == "think_tags" and not disable_thinking:
                    if "</think>" in raw_content:
                        parts = raw_content.split("</think>", 1)
                        reasoning_content = parts[0].replace("<think>", "").strip()
                        answer_content = parts[1].strip()

                wrapped_choice = LocalChoice(
                    message=LocalMessage(
                        content=answer_content,
                        reasoning=reasoning_content,
                    ),
                    finish_reason=choice.finish_reason or "stop",
                )

                metrics = self._extract_metrics(
                    completion, wrapped_choice, start_time, end_time,
                    is_continuation, continuation_depth,
                )

                logger.debug(f"Local LLM response (first 300 chars):\n{answer_content[:300]}")
                return wrapped_choice, metrics

            except Exception as e:
                logger.warning(
                    f"[local] {self.model_name} attempt {attempt + 1}/{self.max_retries} "
                    f"failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def _extract_metrics(
        self,
        completion,
        choice: LocalChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract and normalise token metrics from the local server response.

        Token invariant: output_tokens = reasoning_tokens + answer_tokens.

        Local servers often report only ``completion_tokens`` (total output).
        Reasoning tokens are estimated from word count of the extracted
        ``<think>`` block when the API does not expose them separately.
        """
        duration = end_time - start_time

        api_prompt_tokens = 0
        api_completion_tokens = 0
        if hasattr(completion, "usage") and completion.usage:
            usage = completion.usage
            api_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            api_completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        reasoning_text = choice.message.reasoning
        reasoning_tokens = 0
        has_reasoning = bool(reasoning_text)

        if reasoning_text:
            word_count = len(reasoning_text.split())
            reasoning_tokens = int(word_count * 1.3)
            logger.debug(
                f"[local] Estimated reasoning_tokens={reasoning_tokens} "
                f"({word_count} words × 1.3)"
            )

        output_tokens = api_completion_tokens
        answer_tokens = max(0, output_tokens - reasoning_tokens)

        finish_reason = choice.finish_reason
        if finish_reason in ("max_tokens", "length"):
            finish_reason = "length"

        return LLMCallMetrics(
            model_name=self.model_name,
            role=self.role,
            prompt_tokens=api_prompt_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            duration_seconds=duration,
            throughput_tokens_per_sec=output_tokens / duration if duration > 0 else 0.0,
            finish_reason=finish_reason,
            has_reasoning=has_reasoning,
            timestamp=start_time,
            is_continuation=is_continuation,
            continuation_depth=continuation_depth,
            provider=self.provider_name,
        )

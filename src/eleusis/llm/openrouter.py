"""OpenRouter API client implementation."""

import logging
import os
import time
from dataclasses import dataclass

import httpx
from openai import OpenAI

from eleusis.llm.base import BaseLLMClient, LLMCallMetrics

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterMessage:
    """Message wrapper for OpenRouter responses."""

    content: str
    reasoning: str | None = None


@dataclass
class OpenRouterChoice:
    """Choice wrapper for OpenRouter responses."""

    message: OpenRouterMessage
    finish_reason: str


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API (OpenAI-compatible).

    OpenRouter provides a unified API to access hundreds of models
    from different providers via an OpenAI-compatible interface.

    Supports optional reasoning effort control for models that expose
    extended thinking (e.g., deepseek/deepseek-r1, qwen/qwq-32b).
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        seed: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialize OpenRouter client.

        Args:
            model_name: OpenRouter model ID (e.g., "deepseek/deepseek-r1")
            api_key: OpenRouter API key (falls back to OPENROUTER_API_KEY env var)
            temperature: Sampling temperature
            max_retries: Maximum number of API call retries
            max_tokens: Maximum tokens to generate
            role: Role identifier for metrics ("scientist" or "compiler")
            seed: Random seed for reproducibility
            reasoning_effort: Reasoning effort level ("low"|"medium"|"high") for
                              models that support the reasoning parameter
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )
        self.reasoning_effort = reasoning_effort

        self.client = OpenAI(
            base_url=self.OPENROUTER_BASE_URL,
            api_key=self.api_key,
            timeout=httpx.Timeout(3600.0),
            default_headers={
                "HTTP-Referer": "https://github.com/yannsadowski/eleusis-slm-benchmark",
                "X-Title": "Eleusis SLM Benchmark",
            },
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[OpenRouterChoice, LLMCallMetrics]:
        """Make a single API call with retry logic."""
        logger.debug(
            f"Calling OpenRouter API with {self.max_tokens} tokens, messages:\n{messages}"
        )

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                api_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }

                if self.seed is not None:
                    api_kwargs["seed"] = self.seed

                # Add reasoning effort for models that support it (skip on force-answer)
                if self.reasoning_effort and not disable_thinking:
                    api_kwargs["reasoning"] = {"effort": self.reasoning_effort}

                completion = self.client.chat.completions.create(**api_kwargs)

                end_time = time.time()
                choice = completion.choices[0]

                text_content = choice.message.content or ""
                reasoning_content = None

                # OpenRouter exposes reasoning in `reasoning` or `reasoning_content` field
                if hasattr(choice.message, "reasoning") and choice.message.reasoning:
                    reasoning_content = choice.message.reasoning
                elif hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                    reasoning_content = choice.message.reasoning_content

                wrapped_choice = OpenRouterChoice(
                    message=OpenRouterMessage(
                        content=text_content,
                        reasoning=reasoning_content,
                    ),
                    finish_reason=choice.finish_reason or "stop",
                )

                metrics = self._extract_metrics(
                    completion, wrapped_choice, start_time, end_time,
                    is_continuation, continuation_depth,
                )

                logger.debug(f"LLM response:\n{wrapped_choice}")
                return wrapped_choice, metrics

            except Exception as e:
                logger.warning(
                    f"{self.model_name} Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def _extract_metrics(
        self,
        completion,
        choice: OpenRouterChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract metrics from API response with normalized token fields.

        OpenRouter follows the OpenAI pattern for most models:
        - completion_tokens = total output (reasoning + answer)
        - reasoning_tokens available in completion_tokens_details for thinking models

        Token invariant: output_tokens = reasoning_tokens + answer_tokens
        """
        duration = end_time - start_time

        # --- RAW API VALUES ---
        api_prompt_tokens = 0
        api_completion_tokens = 0
        api_reasoning_tokens = None

        if hasattr(completion, "usage") and completion.usage:
            usage = completion.usage
            api_prompt_tokens = usage.prompt_tokens or 0
            api_completion_tokens = usage.completion_tokens or 0
            logger.debug(
                f"[OpenRouter] RAW API usage: prompt_tokens={api_prompt_tokens}, "
                f"completion_tokens={api_completion_tokens}"
            )

            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                logger.debug(f"[OpenRouter] completion_tokens_details: {details}")
                if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                    api_reasoning_tokens = details.reasoning_tokens
                    logger.debug(f"[OpenRouter] RAW API reasoning_tokens={api_reasoning_tokens}")
        else:
            logger.debug("[OpenRouter] No usage data in completion")

        # --- REASONING CONTENT ---
        reasoning_text = choice.message.reasoning
        reasoning_word_count = len(reasoning_text.split()) if reasoning_text else 0
        logger.debug(
            f"[OpenRouter] Reasoning content present: {reasoning_text is not None}, "
            f"word_count={reasoning_word_count}"
        )
        if reasoning_text:
            logger.debug(f"[OpenRouter] Reasoning preview: {reasoning_text[:200]}...")

        # --- COMPUTED VALUES ---
        # OpenAI pattern: completion_tokens includes both reasoning and answer
        prompt_tokens = api_prompt_tokens
        output_tokens = api_completion_tokens
        reasoning_tokens = api_reasoning_tokens or 0
        has_reasoning = reasoning_tokens > 0

        # Estimate from content if API doesn't provide reasoning token count
        if not has_reasoning and reasoning_text:
            has_reasoning = True
            reasoning_tokens = int(reasoning_word_count * 1.3)
            logger.debug(
                f"[OpenRouter] ESTIMATED reasoning_tokens: "
                f"{reasoning_word_count} words × 1.3 = {reasoning_tokens}"
            )
        elif api_reasoning_tokens:
            logger.debug(f"[OpenRouter] Using NATIVE reasoning_tokens from API: {api_reasoning_tokens}")

        answer_tokens = max(0, output_tokens - reasoning_tokens)

        logger.debug(
            f"[OpenRouter] FINAL token counts: prompt={prompt_tokens}, "
            f"output={output_tokens} (answer={answer_tokens} + reasoning={reasoning_tokens})"
        )

        finish_reason = choice.finish_reason
        if finish_reason == "max_tokens":
            finish_reason = "length"

        metrics = LLMCallMetrics(
            model_name=self.model_name,
            role=self.role,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            duration_seconds=duration,
            throughput_tokens_per_sec=output_tokens / duration if duration > 0 else 0,
            finish_reason=finish_reason,
            has_reasoning=has_reasoning,
            timestamp=start_time,
            is_continuation=is_continuation,
            continuation_depth=continuation_depth,
            provider=self.provider_name,
        )

        logger.debug(
            f"[OpenRouter] Metrics summary: {output_tokens} output tokens in {duration:.2f}s "
            f"({metrics.throughput_tokens_per_sec:.2f} tok/s)"
        )

        return metrics

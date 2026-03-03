"""HuggingFace Inference Providers client implementation."""

import logging
import os
import time
from dataclasses import dataclass

from huggingface_hub import InferenceClient

from eleusis.llm.base import BaseLLMClient, LLMCallMetrics


@dataclass
class StreamedMessage:
    """Message wrapper for streaming responses."""
    content: str
    reasoning: str | None = None


@dataclass
class StreamedChoice:
    """Choice wrapper for streaming responses."""
    message: StreamedMessage
    finish_reason: str

logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseLLMClient):
    """Client for Hugging Face Inference Providers API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        seed: int | None = None,
        stream: bool = True,
        hf_provider: str | None = None,
        reasoning_format: str = "separate_field",
        timeout: int = 300,
    ) -> None:
        """Initialize HuggingFace client using Inference Providers.

        Args:
            hf_provider: Inference provider to use (e.g., "together", "novita").
                        If None, uses HuggingFace's default routing.
            reasoning_format: How reasoning is provided by the model:
                            "think_tags" - reasoning in <think>...</think> tags in content
                            "separate_field" - reasoning in separate API field
            timeout: Request timeout in seconds (default 300s / 5 minutes).
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("HF_TOKEN"),
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )

        self.stream = stream
        self.hf_provider = hf_provider
        self.reasoning_format = reasoning_format
        self.timeout = timeout

        # Initialize client with provider if specified
        client_kwargs = {"timeout": self.timeout}
        if self.api_key:
            client_kwargs["token"] = self.api_key
        if self.hf_provider:
            client_kwargs["provider"] = self.hf_provider
        self.client = InferenceClient(**client_kwargs)

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[object, LLMCallMetrics]:
        """Make a single API call with retry logic."""
        if self.stream:
            return self._call_api_streaming(
                messages, is_continuation, continuation_depth, disable_thinking
            )
        return self._call_api_non_streaming(
            messages, is_continuation, continuation_depth, disable_thinking
        )

    def _call_api_streaming(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[StreamedChoice, LLMCallMetrics]:
        """Make a streaming API call."""
        logger.debug(
            f"Calling HF API (streaming) with {self.max_tokens} tokens, messages:\n{messages}"
        )

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                api_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }

                if self.seed is not None:
                    api_kwargs["seed"] = self.seed

                # HuggingFace Inference API doesn't support disable_thinking
                if disable_thinking:
                    logger.debug("disable_thinking not supported by HF API")

                stream = self.client.chat.completions.create(**api_kwargs)

                content = ""
                reasoning = ""
                finish_reason = "stop"
                usage = None
                chars_since_dot = 0
                dots_printed = 0
                for chunk in stream:
                    # Capture usage from final chunk
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = chunk.usage
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            content += delta.content
                            chars_since_dot += len(delta.content)
                            # Print a dot every ~100 tokens (~400 chars)
                            if chars_since_dot >= 400:
                                print(".", end="", flush=True)
                                dots_printed += 1
                                chars_since_dot = 0
                        # Capture reasoning field if present (GPT-OSS, Kimi, etc.)
                        if hasattr(delta, 'reasoning') and delta.reasoning:
                            reasoning += delta.reasoning
                        # Also check reasoning_content (GLM 4.7 via Z.AI)
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            reasoning += delta.reasoning_content
                        if chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason
                if dots_printed > 0:
                    print()  # Newline after dots

                end_time = time.time()

                choice = StreamedChoice(
                    message=StreamedMessage(
                        content=content,
                        reasoning=reasoning if reasoning else None
                    ),
                    finish_reason=finish_reason,
                )

                # Log reasoning content in DEBUG mode
                if reasoning:
                    logger.debug(f"[HF stream] Reasoning content captured ({len(reasoning)} chars):")
                    logger.debug(f"[HF stream] Reasoning preview: {reasoning[:500]}...")

                metrics = self._extract_metrics_streaming(
                    content, choice, start_time, end_time,
                    is_continuation, continuation_depth, usage
                )

                logger.debug(f"LLM response:\n{choice}")
                return choice, metrics

            except Exception as e:
                logger.warning(f"{self.model_name} Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def _call_api_non_streaming(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[object, LLMCallMetrics]:
        """Make a non-streaming API call."""
        logger.debug(
            f"Calling HF API with {self.max_tokens} tokens, messages:\n{messages}"
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

                # HuggingFace Inference API doesn't support disable_thinking
                if disable_thinking:
                    logger.debug("disable_thinking not supported by HF API")

                completion = self.client.chat.completions.create(**api_kwargs)

                end_time = time.time()
                choice = completion.choices[0]

                metrics = self._extract_metrics(
                    completion, choice, start_time, end_time,
                    is_continuation, continuation_depth
                )

                logger.debug(f"LLM response:\n{choice}")
                return choice, metrics

            except Exception as e:
                logger.warning(f"{self.model_name} Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def _extract_metrics(
        self,
        completion,
        choice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract metrics from API response with normalized token fields."""
        from eleusis.llm.base import estimate_reasoning_tokens

        duration = end_time - start_time

        # --- RAW API VALUES ---
        prompt_tokens = 0
        api_completion_tokens = 0

        if hasattr(completion, 'usage') and completion.usage:
            usage = completion.usage
            prompt_tokens = usage.prompt_tokens or 0
            api_completion_tokens = usage.completion_tokens or 0
            logger.debug(f"[HF non-stream] RAW API usage: prompt_tokens={prompt_tokens}, completion_tokens={api_completion_tokens}")
        else:
            logger.debug("[HF non-stream] No usage data in completion")

        logger.debug(f"[HF non-stream] reasoning_format={self.reasoning_format}")

        # --- REASONING CONTENT ---
        has_reasoning = False
        reasoning_tokens = 0
        reasoning_text = None
        reasoning_word_count = 0

        if self.reasoning_format == "separate_field":
            # Reasoning in separate API field (Kimi, GPT-OSS)
            # completion_tokens = total output (includes reasoning)
            output_tokens = api_completion_tokens

            # Check if reasoning field is present (GPT-OSS, Kimi use 'reasoning')
            if hasattr(choice.message, 'reasoning') and choice.message.reasoning:
                has_reasoning = True
                reasoning_text = choice.message.reasoning
                reasoning_word_count = len(reasoning_text.split())
                logger.debug(f"[HF non-stream] Reasoning field present: {reasoning_word_count} words")
                logger.debug(f"[HF non-stream] Reasoning preview: {reasoning_text[:200]}...")
            # Also check reasoning_content (GLM 4.7 via Z.AI uses 'reasoning_content')
            elif hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
                has_reasoning = True
                reasoning_text = choice.message.reasoning_content
                reasoning_word_count = len(reasoning_text.split())
                logger.debug(f"[HF non-stream] reasoning_content field present: {reasoning_word_count} words")
                logger.debug(f"[HF non-stream] Reasoning preview: {reasoning_text[:200]}...")
            else:
                logger.debug("[HF non-stream] No reasoning field in response")

            # Estimate answer tokens from visible content
            content_text = choice.message.content or ""
            content_word_count = len(content_text.split())
            answer_tokens = int(content_word_count * 1.3)
            logger.debug(f"[HF non-stream] Content: {content_word_count} words, ESTIMATED answer_tokens: {answer_tokens}")

            # Reasoning = total - answer
            reasoning_tokens = max(0, output_tokens - answer_tokens)
            logger.debug(f"[HF non-stream] separate_field: output_tokens={output_tokens}, answer_tokens={answer_tokens}, reasoning_tokens={reasoning_tokens}")

        elif self.reasoning_format == "think_tags":
            # Reasoning inline in <think> tags (DeepSeek R1)
            # completion_tokens = full content including reasoning
            content = choice.message.content or ""
            has_think_tags = "<think>" in content or "</think>" in content
            logger.debug(f"[HF non-stream] Content has <think> tags: {has_think_tags}")
            if has_think_tags:
                has_reasoning = True
                reasoning_tokens = estimate_reasoning_tokens(content) or 0
                # Estimate word count from think content
                if "</think>" in content:
                    think_content = content.split("</think>", 1)[0]
                    if "<think>" in think_content:
                        think_content = think_content.split("<think>", 1)[1]
                    reasoning_word_count = len(think_content.split())
                logger.debug(f"[HF non-stream] Think content: ~{reasoning_word_count} words")
                logger.debug(f"[HF non-stream] ESTIMATED reasoning_tokens from think tags: {reasoning_tokens}")
            output_tokens = api_completion_tokens
            answer_tokens = max(0, output_tokens - reasoning_tokens)
            logger.debug(f"[HF non-stream] think_tags: output_tokens=API completion_tokens={output_tokens}, answer_tokens=output-reasoning={answer_tokens}")

        else:
            # No reasoning
            output_tokens = api_completion_tokens
            answer_tokens = api_completion_tokens
            reasoning_tokens = 0
            logger.debug(f"[HF non-stream] No reasoning format: output_tokens=answer_tokens={output_tokens}")

        logger.debug(f"[HF non-stream] FINAL token counts: prompt={prompt_tokens}, "
                    f"output={output_tokens} (answer={answer_tokens} + reasoning={reasoning_tokens})")

        metrics = LLMCallMetrics(
            model_name=self.model_name,
            role=self.role,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            duration_seconds=duration,
            throughput_tokens_per_sec=output_tokens / duration if duration > 0 else 0,
            finish_reason=choice.finish_reason,
            has_reasoning=has_reasoning,
            timestamp=start_time,
            is_continuation=is_continuation,
            continuation_depth=continuation_depth,
            provider=self.provider_name,
        )

        logger.debug(
            f"[HF non-stream] Metrics summary: {output_tokens} output tokens in {duration:.2f}s "
            f"({metrics.throughput_tokens_per_sec:.2f} tok/s)"
        )

        return metrics

    def _extract_metrics_streaming(
        self,
        content: str,
        choice: StreamedChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
        usage: object | None = None,
    ) -> LLMCallMetrics:
        """Extract metrics from streaming response with normalized token fields.

        Uses actual token counts from usage if available (via stream_options),
        otherwise estimates from text content.
        """
        from eleusis.llm.base import estimate_reasoning_tokens

        duration = end_time - start_time

        # --- RAW API VALUES ---
        if usage:
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
            api_completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
            estimated = False
            logger.debug(f"[HF stream] RAW API usage (from stream_options): prompt_tokens={prompt_tokens}, completion_tokens={api_completion_tokens}")
        else:
            prompt_tokens = 0
            content_word_count = len(content.split())
            api_completion_tokens = int(content_word_count * 1.3)
            estimated = True
            logger.debug(f"[HF stream] No usage in stream - ESTIMATING from content: {content_word_count} words × 1.3 = {api_completion_tokens}")

        logger.debug(f"[HF stream] reasoning_format={self.reasoning_format}")

        # --- REASONING CONTENT ---
        has_reasoning = False
        reasoning_tokens = 0
        reasoning_word_count = 0

        if self.reasoning_format == "separate_field":
            # Reasoning in separate API field (Kimi, GPT-OSS)
            # completion_tokens = total output (includes reasoning)
            output_tokens = api_completion_tokens

            # Check if reasoning field is present
            if choice.message.reasoning:
                has_reasoning = True
                reasoning_text = choice.message.reasoning
                reasoning_word_count = len(reasoning_text.split())
                logger.debug(f"[HF stream] Reasoning field present: {reasoning_word_count} words")
                logger.debug(f"[HF stream] Reasoning preview: {reasoning_text[:200]}...")
            else:
                logger.debug("[HF stream] No reasoning field in response")

            # Estimate answer tokens from visible content
            content_word_count = len(content.split())
            answer_tokens = int(content_word_count * 1.3)
            logger.debug(f"[HF stream] Content: {content_word_count} words, ESTIMATED answer_tokens: {answer_tokens}")

            # Reasoning = total - answer
            reasoning_tokens = max(0, output_tokens - answer_tokens)
            logger.debug(f"[HF stream] separate_field: output_tokens={output_tokens}, answer_tokens={answer_tokens}, reasoning_tokens={reasoning_tokens}")

        elif self.reasoning_format == "think_tags":
            # Reasoning inline in <think> tags (DeepSeek R1)
            # completion_tokens = full content including reasoning
            has_think_tags = "<think>" in content or "</think>" in content
            logger.debug(f"[HF stream] Content has <think> tags: {has_think_tags}")
            if has_think_tags:
                has_reasoning = True
                reasoning_tokens = estimate_reasoning_tokens(content) or 0
                # Estimate word count from think content
                if "</think>" in content:
                    think_content = content.split("</think>", 1)[0]
                    if "<think>" in think_content:
                        think_content = think_content.split("<think>", 1)[1]
                    reasoning_word_count = len(think_content.split())
                logger.debug(f"[HF stream] Think content: ~{reasoning_word_count} words")
                logger.debug(f"[HF stream] ESTIMATED reasoning_tokens from think tags: {reasoning_tokens}")
            output_tokens = api_completion_tokens
            answer_tokens = max(0, output_tokens - reasoning_tokens)
            logger.debug(f"[HF stream] think_tags: output_tokens=API completion_tokens={output_tokens}, answer_tokens=output-reasoning={answer_tokens}")

        else:
            # No reasoning
            output_tokens = api_completion_tokens
            answer_tokens = api_completion_tokens
            reasoning_tokens = 0
            logger.debug(f"[HF stream] No reasoning format: output_tokens=answer_tokens={output_tokens}")

        logger.debug(f"[HF stream] FINAL token counts: prompt={prompt_tokens}, "
                    f"output={output_tokens} (answer={answer_tokens} + reasoning={reasoning_tokens})")

        metrics = LLMCallMetrics(
            model_name=self.model_name,
            role=self.role,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            duration_seconds=duration,
            throughput_tokens_per_sec=output_tokens / duration if duration > 0 else 0,
            finish_reason=choice.finish_reason,
            has_reasoning=has_reasoning,
            timestamp=start_time,
            is_continuation=is_continuation,
            continuation_depth=continuation_depth,
            provider=self.provider_name,
        )

        est_label = " (ESTIMATED)" if estimated else ""
        logger.debug(
            f"[HF stream] Metrics summary{est_label}: {output_tokens} output tokens in {duration:.2f}s "
            f"({metrics.throughput_tokens_per_sec:.2f} tok/s)"
        )

        return metrics

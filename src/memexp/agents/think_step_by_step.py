from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from memexp.agents.base import AnswerRecord, MemoryReader
from memexp.core.dataset import DatasetQuestion


FINAL_ANSWER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[*_]{1,3}\s*)?(?:#{1,6}\s*)?"
    r"(?:[*_]{0,3}\s*)?final\s+answer(?:\s*[*_]{0,3})?"
    r"\s*[:：]?\s*(?:[*_]{0,3})?\s*",
)
ANSWER_MESSAGE_MARKERS = ("<|message|>",)

THINK_STEP_BY_STEP_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear
5. PRESERVE relative or anchored time expressions as written. Do not convert "the week before 9 June 2023" into an absolute date range unless the question explicitly asks for a calendar-date calculation.

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory 1: [timestamp] - [content]
- Memory 2: [timestamp] - [content]
...

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown -> Memory 2 mentions A's hometown is LA -> Therefore A moved from LA"]
- Inferred facts: [list any facts that require combining information from multiple memories]

## STEP 4: TIME REFERENCE HANDLING
[If applicable, identify relative or anchored time references without replacing them]
- Original reference: [e.g., "the week before 9 June 2023", "yesterday", "last year"]
- Anchor time: [e.g., "9 June 2023" or the memory/question timestamp, if needed]
- Answer wording to preserve: [the exact relative/anchored expression to use in the final answer]

## STEP 5: CONTRADICTION CHECK
[If multiple memories contain different information]
- Conflicting information: [describe]
- Resolution: [explain which is most recent/reliable]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included: [list them]
- [ ] All locations included: [list them]
- [ ] All numbers exact: [list them]
- [ ] All frequencies specific: [list them]
- [ ] All dates/times precise: [list them]
- [ ] All proper nouns preserved: [list them]

## STEP 7: ANSWER FORMULATION
[Explain how you're combining the information to answer the question]

## FINAL ANSWER:
[Provide the concise answer with ALL specific details preserved]

---

Memories:
{{ memories }}

{% if include_question_time %}
Question time: {{ question_time }}

{% endif %}
Question: {{ question }}

Now, follow the Chain-of-Thought process above to answer the question:
""".strip()


@dataclass(frozen=True)
class ThinkStepByStepAgentConfig:
    answer_policy: str = "think_step_by_step_v1"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.0
    top_k: int | None = None
    context_budget_tokens: int | None = None
    include_question_time: bool = True
    empty_memories: str = "No relevant memories retrieved."


class ThinkStepByStepAgent:
    """LLM QA agent that answers from rendered memories using a fixed template."""

    name = "think_step_by_step"

    def __init__(self, config: ThinkStepByStepAgentConfig | None = None) -> None:
        self.config = config or ThinkStepByStepAgentConfig()
        if self.config.answer_policy != "think_step_by_step_v1":
            raise ValueError(
                f"Unsupported think-step-by-step answer policy: {self.config.answer_policy}"
            )

    def answer(
        self,
        question: DatasetQuestion,
        memory_runtime: MemoryReader,
        *,
        item_id: str,
        top_k: int | None = None,
        context_budget_tokens: int | None = None,
    ) -> AnswerRecord:
        request = question.to_read_request(
            top_k=top_k if top_k is not None else self.config.top_k,
            context_budget_tokens=(
                context_budget_tokens
                if context_budget_tokens is not None
                else self.config.context_budget_tokens
            ),
        )
        read_result = memory_runtime.read(request)
        prompt = render_think_step_by_step_prompt(
            memories=read_result.context.text or self.config.empty_memories,
            question=_question_text(question.query),
            question_time=question.query_time,
            include_question_time=(
                self.config.include_question_time and bool(question.query_time)
            ),
        )
        response, usage = self._complete(prompt)
        answer = extract_final_answer(response)
        return AnswerRecord(
            item_id=item_id,
            question_id=question.question_id,
            query=question.query,
            query_time=question.query_time,
            answer=answer,
            agent_name=self.name,
            memory_artifact_id=read_result.stats.get("artifact_id"),
            memory_reads=(read_result,),
            stats={
                "answer_policy": self.config.answer_policy,
                "model": self.config.model,
                "memory_read_count": 1,
                "context_tokens": read_result.context.token_count,
                "context_blocks": read_result.context.block_count,
                "qa_prompt_chars": len(prompt),
                "qa_response_chars": len(response),
                "qa_answer_chars": len(answer),
                "qa_generation_tokens": usage,
            },
            metadata={
                "prompt_name": "think_step_by_step_v1",
                "prompt": prompt,
                "raw_response": response,
            },
        )

    def _complete(self, prompt: str) -> tuple[str, dict[str, int]]:
        if not self.config.model or not self.config.api_key:
            raise RuntimeError(
                "ThinkStepByStepAgent requires model and api_key in config."
            )
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "ThinkStepByStepAgent requires the openai package."
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        response = OpenAI(**client_kwargs).chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[{
                "role": "user",
                "content": prompt,
            }],
        )
        usage = getattr(response, "usage", None)
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens")
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        return str(response.choices[0].message.content or ""), {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        }

    def cache_spec(self) -> dict[str, Any]:
        return {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "config": {
                "answer_policy": self.config.answer_policy,
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_k": self.config.top_k,
                "context_budget_tokens": self.config.context_budget_tokens,
                "include_question_time": self.config.include_question_time,
                "empty_memories": self.config.empty_memories,
            },
            "prompt_template": THINK_STEP_BY_STEP_PROMPT,
        }


def render_think_step_by_step_prompt(
    *,
    memories: str,
    question: str,
    question_time: str | None,
    include_question_time: bool,
) -> str:
    prompt = THINK_STEP_BY_STEP_PROMPT.replace("{{ memories }}", memories)
    prompt = prompt.replace("{{ question }}", question)
    if include_question_time:
        prompt = prompt.replace("{{ question_time }}", str(question_time or ""))
        prompt = prompt.replace("{% if include_question_time %}\n", "")
        prompt = prompt.replace("\n{% endif %}", "")
        return prompt

    start = prompt.index("{% if include_question_time %}")
    end = prompt.index("{% endif %}") + len("{% endif %}")
    return prompt[:start] + prompt[end:].lstrip("\n")


def extract_final_answer(response: str) -> str:
    text = _text_after_last_message_marker(str(response or "").strip())
    matches = list(FINAL_ANSWER_PATTERN.finditer(text.lower()))
    if not matches:
        return text
    answer = text[matches[-1].end():].strip()
    return answer or text


def _text_after_last_message_marker(text: str) -> str:
    lowered = text.lower()
    last_index = -1
    last_marker_length = 0
    for marker in ANSWER_MESSAGE_MARKERS:
        marker_index = lowered.rfind(marker.lower())
        if marker_index > last_index:
            last_index = marker_index
            last_marker_length = len(marker)
    if last_index < 0:
        return text
    return text[last_index + last_marker_length:].strip()


def _question_text(query: str | dict[str, Any]) -> str:
    if isinstance(query, str):
        return query
    value = query.get("question") or query.get("query") or query.get("text")
    if value is not None:
        return str(value)
    return json.dumps(query, ensure_ascii=False, sort_keys=True)


def _usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(name, 0)
    else:
        value = getattr(usage, name, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0

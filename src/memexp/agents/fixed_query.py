from __future__ import annotations

from dataclasses import dataclass

from memexp.agents.base import AnswerRecord, MemoryReader
from memexp.core.dataset import DatasetQuestion


@dataclass(frozen=True)
class FixedQueryAgentConfig:
    answer_policy: str = "context_only_v1"
    top_k: int | None = None
    context_budget_tokens: int | None = None
    empty_answer: str = ""


class FixedQueryAgent:
    """Deterministic QA agent: one question, one memory read, one answer."""

    name = "fixed_query"

    def __init__(self, config: FixedQueryAgentConfig | None = None) -> None:
        self.config = config or FixedQueryAgentConfig()
        if self.config.answer_policy != "context_only_v1":
            raise ValueError(
                f"Unsupported fixed-query answer policy: {self.config.answer_policy}"
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
        answer = read_result.context.text or self.config.empty_answer
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
                "memory_read_count": 1,
                "context_tokens": read_result.context.token_count,
                "context_blocks": read_result.context.block_count,
            },
        )

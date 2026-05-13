from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Protocol

from memexp.agents.base import AnswerRecord
from memexp.agents.think_step_by_step import sanitize_response_for_judge
from memexp.core.dataset import Dataset, DatasetItem, DatasetQuestion
from memexp.evaluators.base import EvaluationRecord

LOCOMO_PROMPT_NAME = "locomo_llm_judge_v1"
LONGMEMEVAL_PROMPT_NAME = "longmemeval_official_eval_qa_v1"
MBENCH_PROMPT_NAME = "mbench_llm_judge_v1"

LOCOMO_ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

RELATION_TYPE_GUIDANCE = {
    "complementary": (
        "The memory items are jointly valid. Judge whether the answer correctly "
        "integrates compatible evidence."
    ),
    "nuanced": (
        "The memory items are valid only when temporal or contextual conditions "
        "are preserved. Judge whether the answer selects the condition relevant "
        "to the question."
    ),
    "contradictory": (
        "The memory items remain inconsistent. Judge whether the answer respects "
        "the unresolved inconsistency instead of merging incompatible memories "
        "into one confident state."
    ),
    "default": "Judge only against the provided references and metadata.",
}

RELATION_SUBTYPE_GUIDANCE = {
    "K=1": (
        "One memory item is decisive for the target; compatible background alone "
        "is not enough."
    ),
    "K>1": (
        "Multiple compatible memory items must be combined; all required target "
        "facts or constraints should be present."
    ),
    "any_one": (
        "Any one of multiple compatible memory items is sufficient; do not require "
        "all valid paths."
    ),
    "Temporal": "Time determines which memory applies.",
    "Context": "Context determines which memory applies.",
    "contradictory": (
        "The memories remain irreconcilable under supported conditions."
    ),
    "non_persona_contradiction": (
        "This is a factual or non-persona contradiction; do not infer persona "
        "context from the subtype label."
    ),
    "default": "No additional relation-subtype guidance is available.",
}

RELATION_SUBTYPE_ALIASES = {
    "a_user_vs_user": "non_persona_contradiction",
    "b_user_vs_non_user": "non_persona_contradiction",
    "c_non_user_vs_non_user": "non_persona_contradiction",
}

SOURCE_GUIDANCE = {
    "user-related": (
        "This is a user-related memory question. Judge user preferences, status, "
        "habits, identity, or contextual state only from the provided references."
    ),
    "user-unrelated": (
        "This is not a persona/user-related grading case. Judge only by the "
        "provided references and relation semantics."
    ),
    "default": "Judge neutrally using the provided references and metadata.",
}

MBENCH_JUDGE_PROMPT = """
You are a benchmark answer judge for open-ended memory evaluation questions.

Label the generated answer as CORRECT or WRONG.

Grading rules:
- Mark CORRECT if the generated answer matches the meaning of any accepted correct answer.
- Accept paraphrases, summaries, longer explanations, and equivalent wording.
- Mark WRONG if the generated answer clearly agrees with any known incorrect answer.
- Mark WRONG if the generated answer contradicts the accepted correct answers, misses the key point, or makes an unsupported over-confident choice.
- Use the case description, facts, relation type, and relation subtype as supporting context, not as replacements for the accepted correct answers.
- If the case is contradictory or nuanced, do not reward answers that collapse unresolved tension into an unjustified single conclusion.
- Ignore style or tone differences; grade factual and decision correctness.

Question:
{question}

Accepted correct answers:
{accepted_correct_answers}

Known incorrect answers:
{known_incorrect_answers}

Generated answer:
{generated_answer}

Facts:
{facts}

Case description:
{case}

Relation type:
{relation_type}

Relation subtype:
{relation_subtype}

Topic:
{topic}

Source:
{source}

Additional judging guidance:
{relation_guidance}
{source_guidance}

Return JSON only with exactly two keys: label and reason.
The label must be CORRECT or WRONG.
The reason must be one brief sentence explaining the key grading decision.
""".strip()


class JudgeBackend(Protocol):

    def complete(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class DatasetPromptJudgeConfig:
    dataset_family: str = "auto"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 8192


class OpenAICompatibleJudgeBackend:

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 8192,
    ) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "OpenAI-compatible judge requires the openai package."
            ) from exc

        self.model = model
        self.response_format = response_format
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def complete(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "max_tokens": self.max_tokens,
        }
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
        response = self.client.chat.completions.create(**kwargs)
        return str(response.choices[0].message.content or "")


class DatasetPromptJudgeEvaluator:
    name = "dataset_prompt_judge"

    def __init__(
        self,
        config: DatasetPromptJudgeConfig | None = None,
        *,
        backend: JudgeBackend | None = None,
    ) -> None:
        self.config = config or DatasetPromptJudgeConfig()
        self.backend = backend

    def evaluate(
        self,
        answer: AnswerRecord,
        question: DatasetQuestion,
        *,
        dataset: Dataset | None = None,
        item: DatasetItem | None = None,
    ) -> EvaluationRecord:
        dataset_family = self._dataset_family(dataset)
        reference_answer = _question_reference_answer(question)
        if not _has_nonempty_gold_answer(reference_answer):
            return _skipped_record(
                answer,
                question,
                dataset_family=dataset_family,
                prompt_name=_prompt_name(dataset_family),
                reference_answer=reference_answer,
                skip_reason="empty_gold_answer",
            )

        if dataset_family == "locomo":
            return self._evaluate_locomo(answer, question, reference_answer)
        if dataset_family == "longmemeval":
            return self._evaluate_longmemeval(answer, question,
                                              reference_answer)
        if dataset_family == "mbench":
            return self._evaluate_mbench(answer, question, reference_answer)
        raise ValueError(
            f"Unsupported evaluation dataset family: {dataset_family}")

    def _evaluate_locomo(
        self,
        answer: AnswerRecord,
        question: DatasetQuestion,
        reference_answer: Any,
    ) -> EvaluationRecord:
        category = str(question.metadata.get("question_type", "unknown"))
        prompt_name = LOCOMO_PROMPT_NAME
        if category == "5":
            return _skipped_record(
                answer,
                question,
                dataset_family="locomo",
                prompt_name=prompt_name,
                reference_answer=reference_answer,
                skip_reason="category_5",
                extra_metrics={"category": category},
            )

        prompt = LOCOMO_ACCURACY_PROMPT.format(
            question=str(question.query),
            gold_answer=reference_answer,
            generated_answer=_generated_answer_for_judge(answer),
        )
        raw_judge = self._backend_for("locomo").complete(prompt)
        label = _parse_locomo_label(raw_judge) or "WRONG"
        passed = label == "CORRECT"
        return EvaluationRecord(
            item_id=answer.item_id,
            question_id=answer.question_id,
            evaluator_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reference=reference_answer,
            metrics={
                "evaluated": True,
                "dataset_family": "locomo",
                "prompt_name": prompt_name,
                "category": category,
                "judge_label": label,
                "judge_raw": raw_judge,
            },
        )

    def _evaluate_longmemeval(
        self,
        answer: AnswerRecord,
        question: DatasetQuestion,
        reference_answer: Any,
    ) -> EvaluationRecord:
        question_type = str(question.metadata.get("question_type", "unknown"))
        abstention = "_abs" in question.question_id
        prompt_name = LONGMEMEVAL_PROMPT_NAME
        prompt = longmemeval_prompt(
            question_type,
            str(question.query),
            str(reference_answer),
            _generated_answer_for_judge(answer),
            abstention=abstention,
        )
        raw_judge = self._backend_for("longmemeval").complete(prompt)
        passed = "yes" in raw_judge.lower()
        return EvaluationRecord(
            item_id=answer.item_id,
            question_id=answer.question_id,
            evaluator_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reference=reference_answer,
            metrics={
                "evaluated": True,
                "dataset_family": "longmemeval",
                "prompt_name": prompt_name,
                "question_type": question_type,
                "abstention": abstention,
                "judge_raw": raw_judge,
            },
        )

    def _evaluate_mbench(
        self,
        answer: AnswerRecord,
        question: DatasetQuestion,
        reference_answer: Any,
    ) -> EvaluationRecord:
        metadata = _mbench_metadata(question)
        correct_answers = _reference_answers(reference_answer)
        incorrect_answers = _reference_answers(metadata.get("incorrect_answers"))
        generated_answer = _generated_answer_for_judge(answer)
        prompt_name = MBENCH_PROMPT_NAME
        verdict = _deterministic_reference_verdict(
            generated_answer=generated_answer,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers,
        )
        if verdict is None:
            prompt = mbench_judge_prompt(
                question=str(question.query),
                generated_answer=generated_answer,
                correct_answers=correct_answers,
                incorrect_answers=incorrect_answers,
                metadata=metadata,
            )
            raw_judge = self._backend_for("mbench").complete(prompt)
            label, reason = _parse_mbench_judge(raw_judge)
            judge_source = "llm_judge"
        else:
            passed, reason = verdict
            label = "CORRECT" if passed else "WRONG"
            raw_judge = json.dumps(
                {
                    "label": label,
                    "reason": reason,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            judge_source = "deterministic_reference_bypass"

        passed = label == "CORRECT"
        return EvaluationRecord(
            item_id=answer.item_id,
            question_id=answer.question_id,
            evaluator_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reference=reference_answer,
            metrics={
                "evaluated": True,
                "dataset_family": "mbench",
                "prompt_name": prompt_name,
                "relation_type": _metadata_text(metadata, "relation_type"),
                "relation_subtype": _metadata_text(metadata, "relation_subtype"),
                "topic": _metadata_text(metadata, "topic"),
                "source": _metadata_text(metadata, "source"),
                "judge_label": label,
                "judge_reason": reason,
                "judge_raw": raw_judge,
                "judge_source": judge_source,
            },
        )

    def _dataset_family(self, dataset: Dataset | None) -> str:
        if self.config.dataset_family != "auto":
            return self.config.dataset_family
        dataset_name = (dataset.name if dataset is not None else "").lower()
        if "locomo" in dataset_name:
            return "locomo"
        if "longmemeval" in dataset_name:
            return "longmemeval"
        if "mbench" in dataset_name or "subtlememory" in dataset_name:
            return "mbench"
        raise ValueError(
            "Cannot infer evaluation dataset family. Set DatasetPromptJudgeConfig.dataset_family."
        )

    def _backend_for(self, dataset_family: str) -> JudgeBackend:
        if self.backend is not None:
            return self.backend
        model = self.config.model or os.getenv("OPENAI_MODEL")
        if not model:
            raise RuntimeError(
                "Judge model is required for dataset prompt evaluation.")
        return OpenAICompatibleJudgeBackend(
            model=model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            response_format={"type": "json_object"}
            if dataset_family in {"locomo", "mbench"} else None,
            max_tokens=self.config.max_tokens,
        )


def longmemeval_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool,
) -> str:
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            f"Question: {question}\n\n"
            f"Explanation: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )

    if task in {
            "single-session-user", "single-session-assistant", "multi-session"
    }:
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. If the response only contains a subset "
            "of the information required by the answer, answer no.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only.")
    if task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. If the response only contains a subset "
            "of the information required by the answer, answer no. In addition, do not penalize off-by-one "
            "errors for the number of days. If the question asks for the number of days/weeks/months, etc., "
            "and the model makes off-by-one errors, the model's response is still correct.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only.")
    if task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered correct as long as the updated answer is the required answer.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only.")
    if task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response from a model. "
            "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. The response is correct as long as "
            "it recalls and utilizes the user's personal information correctly.\n\n"
            f"Question: {question}\n\n"
            f"Rubric: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only.")
    raise NotImplementedError(f"Unsupported LongMemEval question type: {task}")


def mbench_judge_prompt(
    *,
    question: str,
    generated_answer: str,
    correct_answers: list[str],
    incorrect_answers: list[str],
    metadata: dict[str, Any],
) -> str:
    return MBENCH_JUDGE_PROMPT.format(
        question=question,
        accepted_correct_answers=_format_reference_block(correct_answers),
        known_incorrect_answers=_format_reference_block(
            incorrect_answers,
            empty_placeholder="(no explicit incorrect references provided)",
        ),
        generated_answer=generated_answer,
        facts=_format_reference_block(
            _reference_answers(metadata.get("facts")),
            empty_placeholder="(no facts provided)",
        ),
        case=_optional_text(metadata.get("case")),
        relation_type=_optional_text(metadata.get("relation_type")),
        relation_subtype=_optional_text(metadata.get("relation_subtype")),
        topic=_optional_text(metadata.get("topic")),
        source=_optional_text(metadata.get("source")),
        relation_guidance=_relation_guidance(metadata),
        source_guidance=_source_guidance(metadata),
    )


def _skipped_record(
    answer: AnswerRecord,
    question: DatasetQuestion,
    *,
    dataset_family: str,
    prompt_name: str,
    reference_answer: Any,
    skip_reason: str,
    extra_metrics: dict[str, Any] | None = None,
) -> EvaluationRecord:
    metrics = {
        "evaluated": False,
        "dataset_family": dataset_family,
        "prompt_name": prompt_name,
        "skip_reason": skip_reason,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return EvaluationRecord(
        item_id=answer.item_id,
        question_id=answer.question_id,
        evaluator_name=DatasetPromptJudgeEvaluator.name,
        score=None,
        passed=None,
        reference=reference_answer,
        metrics=metrics,
    )


def _prompt_name(dataset_family: str) -> str:
    if dataset_family == "locomo":
        return LOCOMO_PROMPT_NAME
    if dataset_family == "longmemeval":
        return LONGMEMEVAL_PROMPT_NAME
    if dataset_family == "mbench":
        return MBENCH_PROMPT_NAME
    return "unknown"


def _question_reference_answer(question: DatasetQuestion) -> Any:
    if question.label is not None:
        return question.label.reference_answer
    return None


def _has_nonempty_gold_answer(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set | frozenset):
        return bool(_reference_answers(value))
    return True


def _generated_answer_for_judge(answer: AnswerRecord) -> str:
    reasoning = answer.metadata.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    raw_response = answer.metadata.get("raw_response")
    if isinstance(raw_response, str) and raw_response.strip():
        return sanitize_response_for_judge(raw_response)
    return answer.answer


def _parse_locomo_label(raw_judge: str) -> str | None:
    text = (raw_judge or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        label = str(parsed.get("label", "")).strip().upper()
        if "CORRECT" in label:
            return "CORRECT"
        if "WRONG" in label:
            return "WRONG"
    except json.JSONDecodeError:
        pass

    normalized = text.upper()
    if "CORRECT" in normalized:
        return "CORRECT"
    if "WRONG" in normalized:
        return "WRONG"
    return None


def _mbench_metadata(question: DatasetQuestion) -> dict[str, Any]:
    metadata = dict(question.metadata)
    if question.label is not None:
        metadata.update(question.label.metadata)
    return metadata


def _reference_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set | frozenset):
        return [
            text for item in value
            if (text := _clean_text(item))
        ]
    text = _clean_text(value)
    return [text] if text else []


def _deterministic_reference_verdict(
    *,
    generated_answer: str,
    correct_answers: list[str],
    incorrect_answers: list[str],
) -> tuple[bool, str] | None:
    normalized_generated = _normalize_reference_text(generated_answer)
    if not normalized_generated:
        return None

    correct = {
        _normalize_reference_text(answer)
        for answer in correct_answers
        if answer
    }
    if normalized_generated in correct:
        return True, "Generated answer exactly matches an accepted correct reference."

    incorrect = {
        _normalize_reference_text(answer)
        for answer in incorrect_answers
        if answer
    }
    if normalized_generated in incorrect:
        return False, "Generated answer exactly matches a known incorrect reference."

    return None


def _parse_mbench_judge(raw_judge: str) -> tuple[str, str]:
    payload = _extract_json_object(raw_judge)
    if payload is not None:
        label = str(payload.get("label", "")).strip().upper()
        reason = _clean_text(payload.get("reason"))
        if label == "CORRECT":
            return "CORRECT", reason
        if label == "WRONG":
            return "WRONG", reason

    label = _parse_mbench_text_label(raw_judge)
    return label, ""


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    for candidate in (
        text,
        _regex_group(r"```(?:json)?\s*(\{.*?\})\s*```", text),
        _regex_group(r"(\{[^{}]*\"label\"\s*:[^{}]*\})", text),
    ):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_mbench_text_label(raw_judge: str) -> str:
    normalized = str(raw_judge or "").strip().upper()
    if not normalized:
        return "WRONG"
    if re.search(r"\bWRONG\b", normalized) or re.search(
        r"\bINCORRECT\b", normalized
    ):
        return "WRONG"
    if re.search(r"\bCORRECT\b", normalized):
        return "CORRECT"
    return "WRONG"


def _regex_group(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _format_reference_block(
    answers: list[str],
    *,
    empty_placeholder: str = "(none)",
) -> str:
    if not answers:
        return empty_placeholder
    return "\n".join(f"- {answer}" for answer in answers)


def _relation_guidance(metadata: dict[str, Any]) -> str:
    relation_type = _relation_type_key(metadata.get("relation_type"))
    relation_subtype = _relation_subtype_key(metadata.get("relation_subtype"))
    if relation_type == "default" and relation_subtype == "default":
        return ""
    return "\n".join((
        "Relation semantics guidance:",
        (
            f"- Relation type guidance ({relation_type}): "
            f"{RELATION_TYPE_GUIDANCE[relation_type]}"
        ),
        (
            f"- Relation subtype guidance ({relation_subtype}): "
            f"{RELATION_SUBTYPE_GUIDANCE[relation_subtype]}"
        ),
    ))


def _source_guidance(metadata: dict[str, Any]) -> str:
    source = _source_key(metadata.get("source"))
    if source == "default" and not _metadata_text(metadata, "source"):
        return ""
    return f"Source guidance ({source}): {SOURCE_GUIDANCE[source]}"


def _relation_type_key(value: Any) -> str:
    key = _metadata_text({"value": value}, "value")
    return key if key in RELATION_TYPE_GUIDANCE else "default"


def _relation_subtype_key(value: Any) -> str:
    raw = _metadata_text({"value": value}, "value")
    key = RELATION_SUBTYPE_ALIASES.get(raw, raw)
    return key if key in RELATION_SUBTYPE_GUIDANCE else "default"


def _source_key(value: Any) -> str:
    key = _metadata_text({"value": value}, "value")
    return key if key in SOURCE_GUIDANCE else "default"


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    return _clean_text(metadata.get(key))


def _optional_text(value: Any, *, empty_placeholder: str = "(none)") -> str:
    text = _clean_text(value)
    return text or empty_placeholder


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_reference_text(value: Any) -> str:
    return _clean_text(value).lower()

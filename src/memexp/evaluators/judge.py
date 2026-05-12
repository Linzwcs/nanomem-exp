from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Protocol

from memexp.agents.base import AnswerRecord
from memexp.agents.think_step_by_step import sanitize_response_for_judge
from memexp.core.dataset import Dataset, DatasetItem, DatasetQuestion
from memexp.evaluators.base import EvaluationRecord

LOCOMO_PROMPT_NAME = "locomo_llm_judge_v1"
LONGMEMEVAL_PROMPT_NAME = "longmemeval_official_eval_qa_v1"

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

    def _dataset_family(self, dataset: Dataset | None) -> str:
        if self.config.dataset_family != "auto":
            return self.config.dataset_family
        dataset_name = (dataset.name if dataset is not None else "").lower()
        if "locomo" in dataset_name:
            return "locomo"
        if "longmemeval" in dataset_name:
            return "longmemeval"
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
            if dataset_family == "locomo" else None,
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

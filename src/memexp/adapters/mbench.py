from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from memexp.adapters.unified import SCHEMA_VERSION, clean_metadata, text, text_or_none


def mbench_records_to_unified(
    history_sessions: Iterable[dict[str, Any]],
    bench_instances: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
    source_dir: str | None = None,
) -> dict[str, Any]:
    sessions = list(history_sessions)
    instances = list(bench_instances)
    persona_id = _persona_id(sessions, instances)
    persona_name = _persona_name(instances)
    item_id = f"persona_{persona_id}" if persona_id else "persona"
    conversations = [
        _conversation_from_session(session, persona_name=persona_name)
        for session in sessions
    ]
    questions = [
        question
        for instance in instances
        for question in _questions_from_instance(instance)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "metadata": clean_metadata({
            "source_dataset": "mbench",
            "source_dir": str(Path(source_dir).name) if source_dir else None,
            "persona_id": persona_id,
            "session_count": len(sessions),
            "instance_count": len(instances),
            "item_count": 1,
            "question_count": len(questions),
        }),
        "items": [{
            "item_id": item_id,
            "subject_id": persona_id,
            "conversations": conversations,
            "questions": questions,
            "metadata": clean_metadata({
                "persona_id": persona_id,
                "persona_name": persona_name,
                "source_dataset": "mbench",
            }),
        }],
    }


def _conversation_from_session(
    session: dict[str, Any],
    *,
    persona_name: str | None,
) -> list[dict[str, Any]]:
    session_id = text(session.get("session_id"))
    timestamp = text_or_none(session.get("timestamp"))
    history = session.get("history") or []
    return [
        _message_from_turn(
            turn,
            session=session,
            session_id=session_id,
            timestamp=timestamp,
            turn_index=turn_index,
            persona_name=persona_name,
        ) for turn_index, turn in enumerate(history, start=1)
    ]


def _message_from_turn(
    turn: dict[str, Any],
    *,
    session: dict[str, Any],
    session_id: str,
    timestamp: str | None,
    turn_index: int,
    persona_name: str | None,
) -> dict[str, Any]:
    role = text(turn.get("role")) or "unknown"
    speaker = _speaker_for_role(role, persona_name=persona_name)
    return {
        "message_id": f"{session_id}:m{turn_index}",
        "role": role,
        "speaker": speaker,
        "content": text(turn.get("content")),
        "timestamp": timestamp,
        "metadata": clean_metadata({
            "session_id": session_id,
            "turn_index": turn_index,
            "case_id": text_or_none(session.get("case_id")),
            "source": text_or_none(session.get("source")),
            "conversation_type": text_or_none(session.get("conversation_type")),
            "conversation_flow": text_or_none(session.get("conversation_flow")),
            "persona_signal_level": text_or_none(
                session.get("persona_signal_level")
            ),
            "order": session.get("order"),
        }),
    }


def _questions_from_instance(instance: dict[str, Any]) -> list[dict[str, Any]]:
    qas = instance.get("qas") or []
    questions = []
    for question_index, qa in enumerate(qas, start=1):
        questions.append(_question_from_qa(instance, qa, question_index))
    return questions


def _question_from_qa(
    instance: dict[str, Any],
    qa: dict[str, Any],
    question_index: int,
) -> dict[str, Any]:
    instance_id = text(instance.get("instance_id"))
    relation_type = text_or_none(instance.get("relation_type"))
    relation_subtype = text_or_none(instance.get("relation_subtype"))
    return {
        "question_id": f"{instance_id}:q{question_index}",
        "query": text(qa.get("query")),
        "query_time": None,
        "label": {
            "reference_answer": _string_list(qa.get("correct_answers")),
            "evidence_ids": _string_list(instance.get("session_ids")),
            "metadata": clean_metadata({
                "incorrect_answers": _string_list(qa.get("incorrect_answers")),
                "facts": _string_list(instance.get("facts")),
                "case": text_or_none(instance.get("case")),
                "evidence_level": "conversation",
            }),
        },
        "metadata": clean_metadata({
            "instance_id": instance_id,
            "case_id": text_or_none(instance.get("case_id")),
            "persona_id": text_or_none(instance.get("persona_id")),
            "relation_type": relation_type,
            "relation_subtype": relation_subtype,
            "question_category": relation_type,
            "question_type": relation_subtype,
            "topic": text_or_none(instance.get("topic")),
            "source": text_or_none(instance.get("source")),
        }),
    }


def _speaker_for_role(role: str, *, persona_name: str | None) -> str:
    normalized = role.strip().lower()
    if normalized == "user":
        return persona_name or "user"
    if normalized == "assistant":
        return "assistant"
    return normalized or "unknown"


def _persona_id(
    sessions: list[dict[str, Any]],
    instances: list[dict[str, Any]],
) -> str | None:
    for item in (*instances, *sessions):
        value = text_or_none(item.get("persona_id"))
        if value:
            return value
    return None


def _persona_name(instances: list[dict[str, Any]]) -> str | None:
    for instance in instances:
        raw = instance.get("persona_str")
        if not raw:
            continue
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        name = text_or_none(parsed.get("name"))
        if name:
            return name
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text(item) for item in value if text(item)]

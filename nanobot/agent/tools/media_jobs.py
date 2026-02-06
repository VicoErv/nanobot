"""Shared job tracking for long-running media generation."""

from __future__ import annotations

import asyncio
from typing import Dict, Optional


_ACTIVE_JOBS: Dict[str, Dict[str, object]] = {}


def _job_key(channel: str, chat_id: str) -> str:
    return f"{channel}:{chat_id}"


def is_job_active(channel: str, chat_id: str) -> bool:
    job = _ACTIVE_JOBS.get(_job_key(channel, chat_id))
    if not job:
        return False
    task = job.get("task")
    return isinstance(task, asyncio.Task) and not task.done()


def get_job_status(channel: str, chat_id: str) -> Optional[str]:
    job = _ACTIVE_JOBS.get(_job_key(channel, chat_id))
    if not job:
        return None
    started_at = job.get("started_at")
    tool = job.get("tool")
    prompt = job.get("prompt")
    return f"Media generation in progress ({tool}). Started at {started_at}. Prompt: {prompt}"


def register_job(
    channel: str,
    chat_id: str,
    task: asyncio.Task,
    tool: str,
    prompt: str,
    started_at: str,
) -> None:
    _ACTIVE_JOBS[_job_key(channel, chat_id)] = {
        "task": task,
        "tool": tool,
        "prompt": prompt[:200],
        "started_at": started_at,
    }


def clear_job(channel: str, chat_id: str) -> None:
    _ACTIVE_JOBS.pop(_job_key(channel, chat_id), None)

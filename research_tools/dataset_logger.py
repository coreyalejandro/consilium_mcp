"""
Lightweight dataset logger for creating SFT-ready JSONL records.

Usage: call `log_training_example(...)` with the prompt, tool results,
and final output. Controlled by env var USE_DATASET_LOGGING.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

def _ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def log_training_example(
    *,
    output_path: str,
    session_id: Optional[str],
    calling_model: str,
    user_prompt: str,
    tool_messages: List[Dict[str, Any]],
    final_output: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a single JSONL training example to `output_path`.

    The JSON schema is intentionally simple and general:
    {
      "ts": ISO-8601 string,
      "session_id": str|None,
      "calling_model": str,
      "prompt": str,
      "tools": [ {"name": str|None, "content": str} ... ],
      "final": str,
      "meta": {...}
    }
    """
    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "session_id": session_id,
        "calling_model": calling_model,
        "prompt": user_prompt,
        "tools": [
            {
                "name": msg.get("tool_name") or msg.get("tool_call_id") or None,
                "content": msg.get("content", ""),
            }
            for msg in tool_messages
            if isinstance(msg, dict) and msg.get("role") == "tool"
        ],
        "final": final_output,
        "meta": metadata or {},
    }

    _ensure_parent_dir(output_path)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



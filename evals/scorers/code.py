"""Code scorer: extract code, append test_code, execute, score by exit code."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import tempfile

from evals.scorers.base import EvalCase, EvalResult

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> str:
    text = _THINK_RE.sub("", text)
    matches = _FENCE_RE.findall(text)
    if matches:
        return max(matches, key=len)
    return text


class CodeExecScorer:
    """Score = 1.0 if the candidate code + appended test_code exits 0,
    0.0 otherwise. Uses a subprocess with timeout. NOT a sandbox — only
    use on trusted prompts."""

    name = "code_exec"

    def __init__(self, timeout: int = 5):
        self._timeout = timeout

    async def score(self, case: EvalCase, answer: str) -> EvalResult:
        code = _extract_code(answer).strip()
        if not code:
            return EvalResult(case=case, answer=answer, score=0.0, notes="no code extracted")

        full = code + "\n\n" + (case.test_code or "")
        path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(full)
                path = f.name
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return EvalResult(case=case, answer=answer, score=0.0, notes="timeout")
            if proc.returncode == 0:
                return EvalResult(case=case, answer=answer, score=1.0, notes="passed")
            err = stderr.decode("utf-8", errors="replace")[:300]
            return EvalResult(case=case, answer=answer, score=0.0, notes=f"failed: {err}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("scorer scaffolding failed: %s", exc)
            return EvalResult(case=case, answer=answer, score=0.0, notes=f"scaffold error: {exc}")
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

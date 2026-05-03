"""Code verifier — AST validation + (optional) sandboxed execution.

Execution is OPT-IN because running arbitrary LLM-generated code is a real
RCE vector. Even with `execute=True`, we statically reject obvious dangerous
patterns (subprocess, os.system, network, file I/O) before running. This is
NOT a sandbox — for production use, wrap in firejail/bubblewrap/Docker.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import sys
import tempfile

from fleet.verifiers.base import Candidate, VerificationResult, Verifier

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# Modules / names blocklisted from execution. Conservative — false positives
# are fine here, false negatives leak RCE.
_DANGEROUS_IMPORTS = {
    "subprocess", "os", "sys", "socket", "urllib", "requests", "httpx",
    "shutil", "ctypes", "multiprocessing", "asyncio", "pathlib",
    "importlib", "tempfile", "pickle", "marshal", "pty", "fcntl",
}
_DANGEROUS_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input",
}


def _strip(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _extract_code(text: str) -> str:
    matches = _FENCE_RE.findall(text)
    if matches:
        return max(matches, key=len)
    return text


def _has_dangerous_pattern(tree: ast.AST) -> tuple[bool, str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in _DANGEROUS_IMPORTS:
                    return True, f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".", 1)[0]
            if top in _DANGEROUS_IMPORTS:
                return True, f"from {node.module} import"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _DANGEROUS_CALLS:
                return True, f"call to {f.id}"
            if isinstance(f, ast.Attribute) and f.attr in _DANGEROUS_CALLS:
                return True, f"call to .{f.attr}"
    return False, ""


class CodeVerifier:
    """Score code candidates by AST validity and (optional) execution."""

    tag = "code"

    def __init__(self, execute: bool = False, execute_timeout: int = 5):
        self._execute = execute
        self._timeout = execute_timeout

    async def aggregate(
        self,
        prompt: str,
        candidates: list[Candidate],
    ) -> VerificationResult:
        if not candidates:
            return VerificationResult(winner=None, all_scored=[], rationale="no candidates", abstain=True)

        scored: list[Candidate] = []
        for c in candidates:
            scored.append(await self._score_one(c))

        winner = max(scored, key=lambda c: (c.score, len(c.text)))
        # Code is mostly objective: if even one parses, return it. Only
        # abstain when nothing even compiles.
        abstain = winner.score == 0.0
        return VerificationResult(
            winner=winner if not abstain else None,
            all_scored=scored,
            rationale=winner.notes,
            abstain=abstain,
        )

    async def _score_one(self, candidate: Candidate) -> Candidate:
        code = _extract_code(_strip(candidate.text))
        if not code.strip():
            return candidate.with_score(0.0, "no code found")

        # Static checks — base score.
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError) as exc:
            return candidate.with_score(0.0, f"syntax error: {exc}")
        except (RecursionError, MemoryError):
            return candidate.with_score(0.0, "parse exhausted resources")

        score = 0.5  # parses
        notes_parts = ["parses"]

        defined = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        if defined:
            score += 0.15
            notes_parts.append(f"defines {defined} symbol(s)")

        has_doc = ast.get_docstring(tree) is not None or any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and ast.get_docstring(n) is not None
            for n in ast.walk(tree)
        )
        if has_doc:
            score += 0.05
            notes_parts.append("documented")

        dangerous, why = _has_dangerous_pattern(tree)
        if dangerous:
            notes_parts.append(f"unsafe ({why}); not executing")
            return candidate.with_score(min(score, 0.5), "; ".join(notes_parts))

        if not self._execute:
            return candidate.with_score(score, "; ".join(notes_parts))

        # Execution gate — runs in a subprocess with timeout.
        exec_score, exec_note = await self._try_execute(code)
        score = max(score, exec_score)
        notes_parts.append(exec_note)
        return candidate.with_score(min(score, 1.0), "; ".join(notes_parts))

    async def _try_execute(self, code: str) -> tuple[float, str]:
        path: str = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
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
                return 0.5, "execution timed out"
            if proc.returncode == 0:
                return 1.0, "executes cleanly"
            err = stderr.decode("utf-8", errors="replace")[:200]
            return 0.6, f"runtime error: {err}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("code exec scaffolding failed: %s", exc)
            return 0.5, f"exec scaffolding failed: {type(exc).__name__}"
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

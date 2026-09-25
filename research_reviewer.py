#!/usr/bin/env python3
"""Deep academic document reviewer for papers, theses, reports and proposals.

Independent application from the Deep Research Engine and Research Assistant.

Workflow
--------
1. Start a new review -> immediately provide a PDF path.
2. The reviewer parses the document, renders relevant pages, and runs specialist
   review agents for:
      * line/paragraph-level review
      * technical accuracy and logic
      * equations and mathematical notation
      * figures/plots/diagrams/tables
      * numerical/data consistency
      * citations and bibliography verification with web evidence
      * literature completeness / missing literature opportunities
      * grammar/readability/academic style
      * natural academic voice (flags machine-like prose signals)
      * cross-section consistency and claims
3. A senior arbiter produces the consolidated review and suggested rewrites.
4. The review is persisted so the user can quit and resume later.
5. Follow-up questions use the saved review + source excerpts and can request
   alternative rewrites or challenge a reviewer suggestion.

The reviewer does not claim to determine whether text was written by AI. It only
flags textual signals that may make prose read as generic or machine-like and
offers more specific, natural academic rewrites.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import textwrap
import time
import uuid
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

try:
    from ddgs import DDGS
except Exception:
    DDGS = None

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / os.getenv("REVIEWER_ENV_FILE", ".env.research_reviewer")
load_dotenv(ENV_PATH, override=True)

SESSIONS_DIR = ROOT / "review_sessions"
REVIEWER_PDF_DIR = ROOT / "reviewer PDFs"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
REVIEWER_PDF_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
REVIEW_GENERAL_MODEL = os.getenv("REVIEW_GENERAL_MODEL", "qwen3.5:35b-a3b")
REVIEW_CRITICAL_MODEL = os.getenv("REVIEW_CRITICAL_MODEL", "qwq:32b")
REVIEW_VISION_MODEL = os.getenv("REVIEW_VISION_MODEL", "qwen3.8:27b")
REVIEW_LLM_TIMEOUT = float(os.getenv("REVIEW_LLM_TIMEOUT_SECONDS", "1800"))
REVIEW_HEARTBEAT = float(os.getenv("REVIEW_HEARTBEAT_SECONDS", "15"))
REVIEW_GENERAL_CTX = int(os.getenv("REVIEW_GENERAL_CTX", "16384"))
REVIEW_GENERAL_TOKENS = int(os.getenv("REVIEW_GENERAL_TOKENS", "4500"))
REVIEW_CRITICAL_CTX = int(os.getenv("REVIEW_CRITICAL_CTX", "24576"))
REVIEW_CRITICAL_TOKENS = int(os.getenv("REVIEW_CRITICAL_TOKENS", "5000"))
REVIEW_VISION_CTX = int(os.getenv("REVIEW_VISION_CTX", "12288"))
REVIEW_VISION_TOKENS = int(os.getenv("REVIEW_VISION_TOKENS", "2500"))
REVIEW_VISION_TIMEOUT = float(os.getenv("REVIEW_VISION_TIMEOUT_SECONDS", "1200"))
REVIEW_CHUNK_CHARS = int(os.getenv("REVIEW_CHUNK_CHARS", "11000"))
REVIEW_CHUNK_OVERLAP = int(os.getenv("REVIEW_CHUNK_OVERLAP", "1200"))
REVIEW_SECTION_MAX_CHARS = int(os.getenv("REVIEW_SECTION_MAX_CHARS", "28000"))
REVIEW_MAX_CITATIONS = int(os.getenv("REVIEW_MAX_CITATIONS", "250"))
REVIEW_CITATION_RESULTS = int(os.getenv("REVIEW_CITATION_RESULTS", "3"))
REVIEW_LITERATURE_QUERIES = int(os.getenv("REVIEW_LITERATURE_QUERIES", "10"))
REVIEW_WEB_RESULTS_PER_QUERY = int(os.getenv("REVIEW_WEB_RESULTS_PER_QUERY", "6"))
REVIEW_VISUAL_REVIEW = os.getenv("REVIEW_VISUAL_REVIEW", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_VISUAL_DPI = int(os.getenv("REVIEW_VISUAL_DPI", "144"))
REVIEW_MAX_VISUAL_PAGES = int(os.getenv("REVIEW_MAX_VISUAL_PAGES", "0"))  # 0 = no limit
REVIEW_SAVE_PAGE_RENDER = os.getenv("REVIEW_SAVE_PAGE_RENDER", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_MAX_PDF_MB = int(os.getenv("REVIEW_MAX_PDF_MB", "300"))
REVIEW_WEB_TIMEOUT = int(os.getenv("REVIEW_WEB_TIMEOUT_SECONDS", "30"))
REVIEW_USER_AGENT = os.getenv(
    "REVIEW_USER_AGENT",
    "LocalResearchReviewer/1.0 (academic document review workflow)",
)
REVIEW_REFERENCE_MAX_WEB = int(os.getenv("REVIEW_REFERENCE_MAX_WEB", "120"))
REVIEW_REWRITE_EXAMPLES = int(os.getenv("REVIEW_REWRITE_EXAMPLES", "12"))
REVIEW_HOSTILE_MODEL = os.getenv("REVIEW_HOSTILE_MODEL", REVIEW_CRITICAL_MODEL)
REVIEW_HOSTILE_CTX = int(os.getenv("REVIEW_HOSTILE_CTX", "24576"))
REVIEW_HOSTILE_TOKENS = int(os.getenv("REVIEW_HOSTILE_TOKENS", "5000"))
REVIEW_HOSTILE_TIMEOUT = float(os.getenv("REVIEW_HOSTILE_TIMEOUT_SECONDS", "1800"))

# Coordinator + dedicated language review. The coordinator is deliberately separate
# from the final arbiter: it decides what needs another pass, while the arbiter
# integrates the final evidence.
REVIEW_COORDINATOR_MODEL = os.getenv("REVIEW_COORDINATOR_MODEL", REVIEW_CRITICAL_MODEL)
REVIEW_COORDINATOR_CTX = int(os.getenv("REVIEW_COORDINATOR_CTX", "24576"))
REVIEW_COORDINATOR_TOKENS = int(os.getenv("REVIEW_COORDINATOR_TOKENS", "4200"))
REVIEW_COORDINATOR_TIMEOUT = float(os.getenv("REVIEW_COORDINATOR_TIMEOUT_SECONDS", "1800"))
REVIEW_COORDINATOR_ENABLED = os.getenv("REVIEW_COORDINATOR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_COORDINATOR_MAX_ROUNDS = int(os.getenv("REVIEW_COORDINATOR_MAX_ROUNDS", "2"))
REVIEW_COORDINATOR_MAX_ACTIONS_PER_ROUND = int(os.getenv("REVIEW_COORDINATOR_MAX_ACTIONS_PER_ROUND", "5"))

REVIEW_LANGUAGE_MODEL = os.getenv("REVIEW_LANGUAGE_MODEL", REVIEW_GENERAL_MODEL)
REVIEW_LANGUAGE_CTX = int(os.getenv("REVIEW_LANGUAGE_CTX", "16384"))
REVIEW_LANGUAGE_TOKENS = int(os.getenv("REVIEW_LANGUAGE_TOKENS", "5000"))
REVIEW_LANGUAGE_TIMEOUT = float(os.getenv("REVIEW_LANGUAGE_TIMEOUT_SECONDS", "1800"))
REVIEW_LANGUAGE_ENABLED = os.getenv("REVIEW_LANGUAGE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_LANGUAGE_REVIEW_ALL_CHUNKS = os.getenv("REVIEW_LANGUAGE_REVIEW_ALL_CHUNKS", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_HUMANIZE_EXAMPLES = int(os.getenv("REVIEW_HUMANIZE_EXAMPLES", "16"))

# Dedicated AI-style signal analysis. This is intentionally framed as a stylistic
# audit rather than an authorship detector because text-only AI detection is not reliable.
REVIEW_AI_STYLE_MODEL = os.getenv("REVIEW_AI_STYLE_MODEL", REVIEW_GENERAL_MODEL)
REVIEW_AI_STYLE_CTX = int(os.getenv("REVIEW_AI_STYLE_CTX", "16384"))
REVIEW_AI_STYLE_TOKENS = int(os.getenv("REVIEW_AI_STYLE_TOKENS", "4200"))
REVIEW_AI_STYLE_TIMEOUT = float(os.getenv("REVIEW_AI_STYLE_TIMEOUT_SECONDS", "1800"))
REVIEW_AI_STYLE_ENABLED = os.getenv("REVIEW_AI_STYLE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

ANSI = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
}
USE_COLOR = bool(getattr(__import__("sys"), "stdout").isatty()) and not os.getenv("NO_COLOR")


def colour(text: str, name: str) -> str:
    if not USE_COLOR:
        return text
    return f"{ANSI.get(name, '')}{text}{ANSI['reset']}"


def log(tag: str, msg: str, color: str = "dim") -> None:
    print(colour(f"[{tag}]", color) + f" {msg}", flush=True)


@dataclass
class PageRecord:
    page: int
    text: str
    lines: list[str]
    has_images: bool
    has_drawings: bool
    has_math_signals: bool
    visual_needed: bool


@dataclass
class ReviewSession:
    session_id: str
    title: str
    pdf_path: str
    pdf_copy: str
    document_type: str
    created: str
    updated: str
    pages: int
    status: str = "created"
    source_filename: str = ""
    mode: str = "full"


# -----------------------------------------------------------------------------
# PDF/document ingestion
# -----------------------------------------------------------------------------
def _math_signal(text: str) -> bool:
    if not text:
        return False
    score = 0
    if re.search(r"(?:=|≤|≥|≈|≠|∂|∇|∫|∑|∏|√|∞|→|←|↔)", text):
        score += 1
    if re.search(r"[αβγδεζηθικλμνξοπρστυφχψω]", text.lower()):
        score += 1
    if re.search(r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*[^.]", text):
        score += 1
    if text.count("/") >= 2 and text.count("^") >= 1:
        score += 1
    return score >= 2


def extract_pages(pdf_path: Path, render_dir: Path) -> list[PageRecord]:
    doc = fitz.open(pdf_path)
    pages: list[PageRecord] = []
    render_dir.mkdir(parents=True, exist_ok=True)
    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        lines = text.splitlines()
        images = page.get_images(full=True)
        drawings = page.get_drawings()
        has_math = _math_signal(text)
        visual_needed = bool(images) or bool(drawings) or has_math or bool(
            re.search(r"\b(?:figure|fig\.?|plot|table|equation|eq\.?|algorithm|diagram)\b", text, re.I)
        )
        pages.append(
            PageRecord(
                page=idx,
                text=text,
                lines=lines,
                has_images=bool(images),
                has_drawings=bool(drawings),
                has_math_signals=has_math,
                visual_needed=visual_needed,
            )
        )
        if REVIEW_VISUAL_REVIEW and visual_needed:
            if REVIEW_MAX_VISUAL_PAGES and sum(1 for p in pages if p.visual_needed) > REVIEW_MAX_VISUAL_PAGES:
                continue
            scale = REVIEW_VISUAL_DPI / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pix.save(render_dir / f"page_{idx:04d}.png")
    doc.close()
    return pages


def detect_document_type(text: str) -> str:
    lower = text[:120000].lower()
    if "thesis" in lower or "dissertation" in lower:
        return "thesis/dissertation"
    if "abstract" in lower and ("keywords" in lower or "introduction" in lower):
        return "research paper/article"
    if "proposal" in lower or "research objectives" in lower and "methodology" in lower:
        return "research proposal"
    if "executive summary" in lower or "recommendations" in lower:
        return "technical/report document"
    return "academic/technical document"


def document_outline(pages: list[PageRecord]) -> str:
    lines: list[str] = []
    for p in pages:
        for raw in p.lines:
            s = raw.strip()
            if not s or len(s) > 120:
                continue
            if re.match(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Za-z0-9][A-Za-z0-9 ,:&/()\-]{3,110}$", s):
                if s.lower() not in {"abstract", "introduction", "references"} or len(lines) < 500:
                    lines.append(f"p.{p.page}: {s}")
    return "\n".join(lines[:500])


def chunk_pages(pages: list[PageRecord]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for p in pages:
        if not p.text.strip():
            continue
        lines = []
        for i, line in enumerate(p.lines, start=1):
            lines.append(f"[p.{p.page} L{i:03d}] {line}")
        page_text = "\n".join(lines)
        if len(page_text) <= REVIEW_CHUNK_CHARS:
            chunks.append({"id": len(chunks) + 1, "pages": [p.page], "text": page_text})
            continue
        start = 0
        while start < len(page_text):
            end = min(len(page_text), start + REVIEW_CHUNK_CHARS)
            piece = page_text[start:end]
            first_page = p.page
            chunks.append({"id": len(chunks) + 1, "pages": [first_page], "text": piece})
            if end >= len(page_text):
                break
            start = max(end - REVIEW_CHUNK_OVERLAP, start + 1)
    return chunks


def extract_reference_block(pages: list[PageRecord]) -> str:
    full = "\n".join(f"\n--- PAGE {p.page} ---\n{p.text}" for p in pages)
    m = re.search(r"\n\s*(?:references|bibliography|works cited)\s*\n", full, re.I)
    if not m:
        return ""
    block = full[m.start():]
    return block[:180000]


def split_reference_entries(block: str) -> list[str]:
    if not block:
        return []
    lines = [re.sub(r"\s+", " ", x).strip() for x in block.splitlines() if x.strip()]
    entries: list[str] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"^(?:\[?\d{1,4}\]?\.?|\([A-Z]?\d{4}[a-z]?\))\s+", line):
            if current:
                entries.append(" ".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append(" ".join(current))
    return entries[:REVIEW_MAX_CITATIONS]


# -----------------------------------------------------------------------------
# Ollama helpers
# -----------------------------------------------------------------------------
# Reasoning is enabled by default for every reviewer agent. Role-specific .env flags
# may deliberately disable it, while the centralized recovery layer may temporarily
# switch to think=False only after configured failure-recovery paths are exhausted.
# Quality-first default: every reviewer agent starts with thinking enabled.
# A user may deliberately disable a specific role with the corresponding .env flag.
REVIEW_GENERAL_THINK = os.getenv("REVIEW_GENERAL_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_MICRO_THINK = os.getenv("REVIEW_MICRO_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_SECTION_THINK = os.getenv("REVIEW_SECTION_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_DATA_THINK = os.getenv("REVIEW_DATA_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_CITATION_THINK = os.getenv("REVIEW_CITATION_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_LITERATURE_PLAN_THINK = os.getenv("REVIEW_LITERATURE_PLAN_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_LITERATURE_ANALYSIS_THINK = os.getenv("REVIEW_LITERATURE_ANALYSIS_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_FOLLOWUP_THINK = os.getenv("REVIEW_FOLLOWUP_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_REWRITE_THINK = os.getenv("REVIEW_REWRITE_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_CRITICAL_THINK = os.getenv("REVIEW_CRITICAL_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_COORDINATOR_THINK = os.getenv("REVIEW_COORDINATOR_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_LANGUAGE_THINK = os.getenv("REVIEW_LANGUAGE_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_AI_STYLE_THINK = os.getenv("REVIEW_AI_STYLE_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_VISION_THINK = os.getenv("REVIEW_VISION_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_HOSTILE_THINK = os.getenv("REVIEW_HOSTILE_THINK", "true").lower() in {"1", "true", "yes", "on"}

# Additional role controls for bounded late-stage specialist passes. These also default
# to thinking mode and may be explicitly disabled in .env.
REVIEW_REPRODUCIBILITY_THINK = os.getenv("REVIEW_REPRODUCIBILITY_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_EXPERIMENTAL_DESIGN_THINK = os.getenv("REVIEW_EXPERIMENTAL_DESIGN_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_NOVELTY_THINK = os.getenv("REVIEW_NOVELTY_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_STATISTICS_THINK = os.getenv("REVIEW_STATISTICS_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_SUITABILITY_THINK = os.getenv("REVIEW_SUITABILITY_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_TITLE_ABSTRACT_THINK = os.getenv("REVIEW_TITLE_ABSTRACT_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_NOMENCLATURE_THINK = os.getenv("REVIEW_NOMENCLATURE_THINK", "true").lower() in {"1", "true", "yes", "on"}

# Empty final answers should not permanently discard a specialist pass. Qwen-family
# reasoning models can spend the generation budget in `thinking` before emitting
# `content`, so retries deliberately increase the output budget before falling back
# to a non-thinking generation as a last recovery path.
# Retry policy for reasoning-capable models.  The reviewer keeps thinking enabled
# while increasing the generation budget until the configured ceiling is reached.
# Only after a failed attempt at that ceiling does it fall back to think=False.
# REVIEW_LLM_MAX_RETRIES is an optional safety cap: 0 means "no retry-count cap;
# continue until the token ceiling is reached".
REVIEW_LLM_RETRY_TOKEN_MULTIPLIER = max(1.1, float(os.getenv("REVIEW_LLM_RETRY_TOKEN_MULTIPLIER", "1.5")))
REVIEW_LLM_MAX_TOKENS = max(1000, int(os.getenv("REVIEW_LLM_MAX_TOKENS", "12000")))
REVIEW_LLM_MAX_RETRIES = max(0, int(os.getenv("REVIEW_LLM_MAX_RETRIES", "0")))
REVIEW_LLM_FALLBACK_NO_THINK = os.getenv("REVIEW_LLM_FALLBACK_NO_THINK", "true").lower() in {"1", "true", "yes", "on"}

# Timeout-specific recovery is deliberately separate from generation-limit recovery.
# Keep the full context so the model does not lose source/work context.
# Timeout #1: increase wall-clock timeout to 1.5x, keep thinking ON.
# Timeout #2: increase wall-clock timeout to 2.0x, keep thinking ON.
# Timeout #3: keep the 2.0x timeout and full context, then disable thinking.
REVIEW_TIMEOUT_FIRST_MULTIPLIER = max(1.0, float(os.getenv("REVIEW_TIMEOUT_FIRST_MULTIPLIER", "1.5")))
REVIEW_TIMEOUT_SECOND_MULTIPLIER = max(REVIEW_TIMEOUT_FIRST_MULTIPLIER, float(os.getenv("REVIEW_TIMEOUT_SECOND_MULTIPLIER", "2.0")))
REVIEW_TIMEOUT_MAX_RECOVERIES = max(0, int(os.getenv("REVIEW_TIMEOUT_MAX_RECOVERIES", "3")))
REVIEW_TIMEOUT_FALLBACK_NO_THINK = os.getenv("REVIEW_TIMEOUT_FALLBACK_NO_THINK", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_TIMEOUT_FALLBACK_TOKEN_MULTIPLIER = max(1.0, float(os.getenv("REVIEW_TIMEOUT_FALLBACK_TOKEN_MULTIPLIER", "1.5")))

# Additional Ollama failure recovery. These policies are deliberately separate from
# generation-limit and timeout recovery so a network/server fault never masquerades
# as a need for more generation tokens.
REVIEW_OLLAMA_TRANSIENT_RETRIES = max(0, int(os.getenv("REVIEW_OLLAMA_TRANSIENT_RETRIES", "3")))
REVIEW_OLLAMA_RESPONSE_RETRIES = max(0, int(os.getenv("REVIEW_OLLAMA_RESPONSE_RETRIES", "2")))
# Ollama token-repeat recovery. This is separate from generic HTTP 5xx recovery because
# repeating the exact same deterministic request is unlikely to escape a repetition loop.
REVIEW_REPEAT_MAX_RECOVERIES = max(0, int(os.getenv("REVIEW_REPEAT_MAX_RECOVERIES", "3")))
REVIEW_REPEAT_TEMPERATURE_1 = max(0.0, float(os.getenv("REVIEW_REPEAT_TEMPERATURE_1", "0.15")))
REVIEW_REPEAT_TEMPERATURE_2 = max(REVIEW_REPEAT_TEMPERATURE_1, float(os.getenv("REVIEW_REPEAT_TEMPERATURE_2", "0.25")))
REVIEW_REPEAT_FALLBACK_TEMPERATURE = max(0.0, float(os.getenv("REVIEW_REPEAT_FALLBACK_TEMPERATURE", "0.15")))
REVIEW_REPEAT_PENALTY_1 = max(1.0, float(os.getenv("REVIEW_REPEAT_PENALTY_1", "1.10")))
REVIEW_REPEAT_PENALTY_2 = max(REVIEW_REPEAT_PENALTY_1, float(os.getenv("REVIEW_REPEAT_PENALTY_2", "1.15")))
REVIEW_REPEAT_FALLBACK_PENALTY = max(1.0, float(os.getenv("REVIEW_REPEAT_FALLBACK_PENALTY", "1.10")))
REVIEW_REPEAT_LAST_N = max(0, int(os.getenv("REVIEW_REPEAT_LAST_N", "128")))
REVIEW_REPEAT_FALLBACK_TOKEN_MULTIPLIER = max(1.0, float(os.getenv("REVIEW_REPEAT_FALLBACK_TOKEN_MULTIPLIER", "1.5")))

# Micro-review stage resilience. After all per-request fallbacks are exhausted,
# the review continues when one or two micro chunks fail. Failed chunks are recorded
# explicitly and surfaced to downstream coordinator/final-arbiter stages.
REVIEW_MAX_MICRO_FAILURES_BEFORE_ABORT = max(0, int(os.getenv("REVIEW_MAX_MICRO_FAILURES_BEFORE_ABORT", "2")))
REVIEW_ABORT_ON_MICRO_FAILURES = os.getenv("REVIEW_ABORT_ON_MICRO_FAILURES", "true").lower() in {"1", "true", "yes", "on"}
# Independent specialist-stage resilience. After all per-request fallbacks are exhausted,
# an individual foundational specialist failure is recorded and the pipeline continues.
# The coordinator/final-arbiter stages remain fail-fast because they are aggregation stages.
REVIEW_MAX_SPECIALIST_FAILURES_BEFORE_ABORT = max(0, int(os.getenv("REVIEW_MAX_SPECIALIST_FAILURES_BEFORE_ABORT", "2")))
REVIEW_ABORT_ON_SPECIALIST_FAILURES = os.getenv("REVIEW_ABORT_ON_SPECIALIST_FAILURES", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_OLLAMA_BACKOFF_INITIAL_SECONDS = max(0.0, float(os.getenv("REVIEW_OLLAMA_BACKOFF_INITIAL_SECONDS", "5")))
REVIEW_OLLAMA_BACKOFF_MAX_SECONDS = max(REVIEW_OLLAMA_BACKOFF_INITIAL_SECONDS, float(os.getenv("REVIEW_OLLAMA_BACKOFF_MAX_SECONDS", "60")))
REVIEW_OLLAMA_HEALTH_TIMEOUT_SECONDS = max(1.0, float(os.getenv("REVIEW_OLLAMA_HEALTH_TIMEOUT_SECONDS", "10")))
REVIEW_CONTEXT_ERROR_REDUCTION = min(0.9, max(0.25, float(os.getenv("REVIEW_CONTEXT_ERROR_REDUCTION", "0.6666667"))))
REVIEW_CONTEXT_ERROR_MIN = max(4096, int(os.getenv("REVIEW_CONTEXT_ERROR_MIN", "8192")))
REVIEW_CONTEXT_ERROR_MAX_RECOVERIES = max(0, int(os.getenv("REVIEW_CONTEXT_ERROR_MAX_RECOVERIES", "2")))
REVIEW_OOM_CONTEXT_REDUCTION = min(0.9, max(0.25, float(os.getenv("REVIEW_OOM_CONTEXT_REDUCTION", "0.75"))))
REVIEW_OOM_CONTEXT_MIN = max(4096, int(os.getenv("REVIEW_OOM_CONTEXT_MIN", "8192")))
REVIEW_OOM_MAX_RECOVERIES = max(0, int(os.getenv("REVIEW_OOM_MAX_RECOVERIES", "2")))
# Context-constrained generation recovery. This handles a different failure from an
# Ollama context-overflow HTTP error: the request is accepted, but `done_reason=length`
# because the input already consumed most of the context window. In that case we shrink
# the source payload, not the model context.
REVIEW_STALLED_LENGTH_MAX_RECOVERIES = max(0, int(os.getenv("REVIEW_STALLED_LENGTH_MAX_RECOVERIES", "4")))
REVIEW_STALLED_LENGTH_MIN_PROMPT_CHARS = max(6000, int(os.getenv("REVIEW_STALLED_LENGTH_MIN_PROMPT_CHARS", "12000")))
REVIEW_STALLED_LENGTH_PROMPT_TARGET_FRACTION = min(0.85, max(0.40, float(os.getenv("REVIEW_STALLED_LENGTH_PROMPT_TARGET_FRACTION", "0.70"))))
REVIEW_STALLED_LENGTH_MIN_PROMPT_FRACTION = min(0.90, max(0.10, float(os.getenv("REVIEW_STALLED_LENGTH_MIN_PROMPT_FRACTION", "0.20"))))
REVIEW_DATA_INPUT_MAX_CHARS = max(12000, int(os.getenv("REVIEW_DATA_INPUT_MAX_CHARS", "36000")))
REVIEW_DATA_INPUT_MIN_CHARS = max(8000, int(os.getenv("REVIEW_DATA_INPUT_MIN_CHARS", "12000")))
REVIEW_MODEL_FALLBACK_ENABLED = os.getenv("REVIEW_MODEL_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_GENERAL_FALLBACK_MODEL = os.getenv("REVIEW_GENERAL_FALLBACK_MODEL", "")
REVIEW_CRITICAL_FALLBACK_MODEL = os.getenv("REVIEW_CRITICAL_FALLBACK_MODEL", REVIEW_GENERAL_MODEL)
REVIEW_COORDINATOR_FALLBACK_MODEL = os.getenv("REVIEW_COORDINATOR_FALLBACK_MODEL", REVIEW_GENERAL_MODEL)
REVIEW_LANGUAGE_FALLBACK_MODEL = os.getenv("REVIEW_LANGUAGE_FALLBACK_MODEL", REVIEW_GENERAL_MODEL)
REVIEW_AI_STYLE_FALLBACK_MODEL = os.getenv("REVIEW_AI_STYLE_FALLBACK_MODEL", REVIEW_GENERAL_MODEL)
REVIEW_HOSTILE_FALLBACK_MODEL = os.getenv("REVIEW_HOSTILE_FALLBACK_MODEL", REVIEW_GENERAL_MODEL)
REVIEW_VISION_FALLBACK_MODEL = os.getenv("REVIEW_VISION_FALLBACK_MODEL", "")

# Web/Crossref recovery. Search failures are retried with backoff and are recorded
# explicitly as failures rather than being mistaken for zero search results.
REVIEW_WEB_RETRIES = max(0, int(os.getenv("REVIEW_WEB_RETRIES", "3")))
REVIEW_WEB_BACKOFF_INITIAL_SECONDS = max(0.0, float(os.getenv("REVIEW_WEB_BACKOFF_INITIAL_SECONDS", "2")))
REVIEW_WEB_BACKOFF_MAX_SECONDS = max(REVIEW_WEB_BACKOFF_INITIAL_SECONDS, float(os.getenv("REVIEW_WEB_BACKOFF_MAX_SECONDS", "20")))
REVIEW_CROSSREF_RETRIES = max(0, int(os.getenv("REVIEW_CROSSREF_RETRIES", "3")))
REVIEW_CROSSREF_BACKOFF_INITIAL_SECONDS = max(0.0, float(os.getenv("REVIEW_CROSSREF_BACKOFF_INITIAL_SECONDS", "2")))
REVIEW_CROSSREF_BACKOFF_MAX_SECONDS = max(REVIEW_CROSSREF_BACKOFF_INITIAL_SECONDS, float(os.getenv("REVIEW_CROSSREF_BACKOFF_MAX_SECONDS", "20")))
REVIEW_WEB_CROSSREF_FALLBACK = os.getenv("REVIEW_WEB_CROSSREF_FALLBACK", "true").lower() in {"1", "true", "yes", "on"}

# -----------------------------------------------------------------------------
# Runtime modes + proactive context packing
# -----------------------------------------------------------------------------
ACTIVE_MODE = "full"
ACTIVE_CONTEXT_SUMMARY = ""
ACTIVE_CONTEXT_PACK_PATH: Path | None = None

REVIEW_MODE_ENV = os.getenv("REVIEW_MODE", "full").strip().lower()
REVIEW_CONTEXT_SUMMARY_MODEL = os.getenv("REVIEW_CONTEXT_SUMMARY_MODEL", REVIEW_GENERAL_MODEL)
REVIEW_CONTEXT_SUMMARY_CTX = max(4096, int(os.getenv("REVIEW_CONTEXT_SUMMARY_CTX", "8192")))
REVIEW_CONTEXT_SUMMARY_TOKENS = max(800, int(os.getenv("REVIEW_CONTEXT_SUMMARY_TOKENS", "2200")))
REVIEW_CONTEXT_SUMMARY_INPUT_CHARS = max(12000, int(os.getenv("REVIEW_CONTEXT_SUMMARY_INPUT_CHARS", "60000")))
REVIEW_CONTEXT_REUSE = os.getenv("REVIEW_CONTEXT_REUSE", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_CONTEXT_PACK_VERSION = "v2"
REVIEW_FULL_PROACTIVE_CONTEXT = os.getenv("REVIEW_FULL_PROACTIVE_CONTEXT", "true").lower() in {"1", "true", "yes", "on"}
REVIEW_FULL_SOURCE_TARGET_GENERAL = max(8000, int(os.getenv("REVIEW_FULL_SOURCE_TARGET_GENERAL", "28000")))
REVIEW_FULL_SOURCE_TARGET_CRITICAL = max(10000, int(os.getenv("REVIEW_FULL_SOURCE_TARGET_CRITICAL", "36000")))
REVIEW_FULL_SOURCE_TARGET_COORDINATOR = max(10000, int(os.getenv("REVIEW_FULL_SOURCE_TARGET_COORDINATOR", "32000")))
REVIEW_SHORT_SOURCE_TARGET_GENERAL = max(5000, int(os.getenv("REVIEW_SHORT_SOURCE_TARGET_GENERAL", "12000")))
REVIEW_SHORT_SOURCE_TARGET_CRITICAL = max(6000, int(os.getenv("REVIEW_SHORT_SOURCE_TARGET_CRITICAL", "10000")))
REVIEW_SHORT_SOURCE_TARGET_COORDINATOR = max(7000, int(os.getenv("REVIEW_SHORT_SOURCE_TARGET_COORDINATOR", "14000")))
REVIEW_SHORT_CONTEXT_SUMMARY_TOKENS = max(600, int(os.getenv("REVIEW_SHORT_CONTEXT_SUMMARY_TOKENS", "1600")))
REVIEW_SHORT_LLM_MAX_TOKENS = max(1200, int(os.getenv("REVIEW_SHORT_LLM_MAX_TOKENS", "3800")))
REVIEW_SHORT_GENERAL_CTX = max(4096, int(os.getenv("REVIEW_SHORT_GENERAL_CTX", "8192")))
REVIEW_SHORT_GENERAL_TOKENS = max(800, int(os.getenv("REVIEW_SHORT_GENERAL_TOKENS", "2200")))
REVIEW_SHORT_CRITICAL_CTX = max(8192, int(os.getenv("REVIEW_SHORT_CRITICAL_CTX", "12288")))
REVIEW_SHORT_CRITICAL_TOKENS = max(1000, int(os.getenv("REVIEW_SHORT_CRITICAL_TOKENS", "3600")))
REVIEW_SHORT_VISION_CTX = max(4096, int(os.getenv("REVIEW_SHORT_VISION_CTX", "8192")))
REVIEW_SHORT_VISION_TOKENS = max(600, int(os.getenv("REVIEW_SHORT_VISION_TOKENS", "1400")))
REVIEW_SHORT_COORDINATOR_CTX = max(8192, int(os.getenv("REVIEW_SHORT_COORDINATOR_CTX", "12288")))
REVIEW_SHORT_COORDINATOR_TOKENS = max(1000, int(os.getenv("REVIEW_SHORT_COORDINATOR_TOKENS", "2400")))
REVIEW_SHORT_LANGUAGE_CTX = max(4096, int(os.getenv("REVIEW_SHORT_LANGUAGE_CTX", "8192")))
REVIEW_SHORT_LANGUAGE_TOKENS = max(900, int(os.getenv("REVIEW_SHORT_LANGUAGE_TOKENS", "2500")))
REVIEW_SHORT_AI_STYLE_CTX = max(4096, int(os.getenv("REVIEW_SHORT_AI_STYLE_CTX", "8192")))
REVIEW_SHORT_AI_STYLE_TOKENS = max(900, int(os.getenv("REVIEW_SHORT_AI_STYLE_TOKENS", "2200")))
REVIEW_SHORT_HOSTILE_CTX = max(8192, int(os.getenv("REVIEW_SHORT_HOSTILE_CTX", "12288")))
REVIEW_SHORT_HOSTILE_TOKENS = max(1000, int(os.getenv("REVIEW_SHORT_HOSTILE_TOKENS", "3000")))
REVIEW_SHORT_TIMEOUT = max(180.0, float(os.getenv("REVIEW_SHORT_TIMEOUT_SECONDS", "900")))
REVIEW_SHORT_VISION_TIMEOUT = max(180.0, float(os.getenv("REVIEW_SHORT_VISION_TIMEOUT_SECONDS", "600")))
REVIEW_SHORT_CONTEXT_RECOVERIES = max(0, int(os.getenv("REVIEW_SHORT_CONTEXT_RECOVERIES", "1")))
REVIEW_SHORT_STALLED_RECOVERIES = max(0, int(os.getenv("REVIEW_SHORT_STALLED_RECOVERIES", "1")))
REVIEW_SHORT_REPEAT_RECOVERIES = max(0, int(os.getenv("REVIEW_SHORT_REPEAT_RECOVERIES", "1")))


def _set_runtime_mode(mode: str) -> None:
    """Apply runtime overrides without requiring separate short-mode env files."""
    global ACTIVE_MODE
    global REVIEW_GENERAL_CTX, REVIEW_GENERAL_TOKENS
    global REVIEW_CRITICAL_CTX, REVIEW_CRITICAL_TOKENS
    global REVIEW_VISION_CTX, REVIEW_VISION_TOKENS, REVIEW_VISION_TIMEOUT
    global REVIEW_COORDINATOR_CTX, REVIEW_COORDINATOR_TOKENS, REVIEW_COORDINATOR_TIMEOUT
    global REVIEW_LANGUAGE_CTX, REVIEW_LANGUAGE_TOKENS, REVIEW_LANGUAGE_TIMEOUT
    global REVIEW_AI_STYLE_CTX, REVIEW_AI_STYLE_TOKENS, REVIEW_AI_STYLE_TIMEOUT
    global REVIEW_HOSTILE_CTX, REVIEW_HOSTILE_TOKENS, REVIEW_HOSTILE_TIMEOUT
    global REVIEW_LLM_MAX_TOKENS, REVIEW_LLM_MAX_RETRIES
    global REVIEW_CONTEXT_ERROR_MAX_RECOVERIES, REVIEW_OOM_MAX_RECOVERIES
    global REVIEW_STALLED_LENGTH_MAX_RECOVERIES, REVIEW_REPEAT_MAX_RECOVERIES
    global REVIEW_TIMEOUT_MAX_RECOVERIES
    global REVIEW_CONTEXT_SUMMARY_TOKENS
    global REVIEW_MICRO_THINK, REVIEW_SECTION_THINK, REVIEW_DATA_THINK
    global REVIEW_CITATION_THINK, REVIEW_LITERATURE_PLAN_THINK, REVIEW_LITERATURE_ANALYSIS_THINK
    global REVIEW_FOLLOWUP_THINK, REVIEW_REWRITE_THINK, REVIEW_CRITICAL_THINK
    global REVIEW_COORDINATOR_THINK, REVIEW_LANGUAGE_THINK, REVIEW_AI_STYLE_THINK
    global REVIEW_VISION_THINK, REVIEW_HOSTILE_THINK, REVIEW_REPRODUCIBILITY_THINK
    global REVIEW_EXPERIMENTAL_DESIGN_THINK, REVIEW_NOVELTY_THINK, REVIEW_STATISTICS_THINK
    global REVIEW_SUITABILITY_THINK, REVIEW_TITLE_ABSTRACT_THINK, REVIEW_NOMENCLATURE_THINK
    global REVIEW_GENERAL_THINK
    global REVIEW_MAX_VISUAL_PAGES, REVIEW_LANGUAGE_REVIEW_ALL_CHUNKS

    mode = (mode or "full").strip().lower()
    if mode not in {"full", "short"}:
        raise ValueError(f"Unknown reviewer mode: {mode}")
    ACTIVE_MODE = mode
    if mode != "short":
        return

    # Short mode keeps EVERY stage and EVERY visual page; only depth per call is reduced.
    REVIEW_GENERAL_CTX = REVIEW_SHORT_GENERAL_CTX
    REVIEW_GENERAL_TOKENS = REVIEW_SHORT_GENERAL_TOKENS
    REVIEW_CRITICAL_CTX = REVIEW_SHORT_CRITICAL_CTX
    REVIEW_CRITICAL_TOKENS = REVIEW_SHORT_CRITICAL_TOKENS
    REVIEW_VISION_CTX = REVIEW_SHORT_VISION_CTX
    REVIEW_VISION_TOKENS = REVIEW_SHORT_VISION_TOKENS
    REVIEW_VISION_TIMEOUT = REVIEW_SHORT_VISION_TIMEOUT
    REVIEW_COORDINATOR_CTX = REVIEW_SHORT_COORDINATOR_CTX
    REVIEW_COORDINATOR_TOKENS = REVIEW_SHORT_COORDINATOR_TOKENS
    REVIEW_COORDINATOR_TIMEOUT = REVIEW_SHORT_TIMEOUT
    REVIEW_LANGUAGE_CTX = REVIEW_SHORT_LANGUAGE_CTX
    REVIEW_LANGUAGE_TOKENS = REVIEW_SHORT_LANGUAGE_TOKENS
    REVIEW_LANGUAGE_TIMEOUT = REVIEW_SHORT_TIMEOUT
    REVIEW_AI_STYLE_CTX = REVIEW_SHORT_AI_STYLE_CTX
    REVIEW_AI_STYLE_TOKENS = REVIEW_SHORT_AI_STYLE_TOKENS
    REVIEW_AI_STYLE_TIMEOUT = REVIEW_SHORT_TIMEOUT
    REVIEW_HOSTILE_CTX = REVIEW_SHORT_HOSTILE_CTX
    REVIEW_HOSTILE_TOKENS = REVIEW_SHORT_HOSTILE_TOKENS
    REVIEW_HOSTILE_TIMEOUT = REVIEW_SHORT_TIMEOUT
    REVIEW_LLM_MAX_TOKENS = REVIEW_SHORT_LLM_MAX_TOKENS
    REVIEW_LLM_MAX_RETRIES = min(REVIEW_LLM_MAX_RETRIES, 1)
    REVIEW_CONTEXT_ERROR_MAX_RECOVERIES = min(REVIEW_CONTEXT_ERROR_MAX_RECOVERIES, REVIEW_SHORT_CONTEXT_RECOVERIES)
    REVIEW_OOM_MAX_RECOVERIES = min(REVIEW_OOM_MAX_RECOVERIES, REVIEW_SHORT_CONTEXT_RECOVERIES)
    REVIEW_STALLED_LENGTH_MAX_RECOVERIES = min(REVIEW_STALLED_LENGTH_MAX_RECOVERIES, REVIEW_SHORT_STALLED_RECOVERIES)
    REVIEW_REPEAT_MAX_RECOVERIES = min(REVIEW_REPEAT_MAX_RECOVERIES, REVIEW_SHORT_REPEAT_RECOVERIES)
    REVIEW_TIMEOUT_MAX_RECOVERIES = min(REVIEW_TIMEOUT_MAX_RECOVERIES, 1)
    REVIEW_CONTEXT_SUMMARY_TOKENS = REVIEW_SHORT_CONTEXT_SUMMARY_TOKENS

    # Short mode intentionally disables reasoning for every agent. The review still runs
    # every stage, all pages, all configured agents and all coordinator/hostile/final passes.
    REVIEW_GENERAL_THINK = False
    REVIEW_MICRO_THINK = False
    REVIEW_SECTION_THINK = False
    REVIEW_DATA_THINK = False
    REVIEW_CITATION_THINK = False
    REVIEW_LITERATURE_PLAN_THINK = False
    REVIEW_LITERATURE_ANALYSIS_THINK = False
    REVIEW_FOLLOWUP_THINK = False
    REVIEW_REWRITE_THINK = False
    REVIEW_CRITICAL_THINK = False
    REVIEW_COORDINATOR_THINK = False
    REVIEW_LANGUAGE_THINK = False
    REVIEW_AI_STYLE_THINK = False
    REVIEW_VISION_THINK = False
    REVIEW_HOSTILE_THINK = False
    REVIEW_REPRODUCIBILITY_THINK = False
    REVIEW_EXPERIMENTAL_DESIGN_THINK = False
    REVIEW_NOVELTY_THINK = False
    REVIEW_STATISTICS_THINK = False
    REVIEW_SUITABILITY_THINK = False
    REVIEW_TITLE_ABSTRACT_THINK = False
    REVIEW_NOMENCLATURE_THINK = False
    REVIEW_MAX_VISUAL_PAGES = 0
    REVIEW_LANGUAGE_REVIEW_ALL_CHUNKS = True


class OllamaContextError(RuntimeError):
    pass


class OllamaOOMError(RuntimeError):
    pass


class OllamaModelUnavailableError(RuntimeError):
    pass


class OllamaRepeatError(RuntimeError):
    pass


def _sleep_backoff(base: float, retry_no: int, cap: float) -> float:
    """Sleep exponentially and return the actual delay used."""
    delay = min(cap, base * (2 ** max(0, retry_no - 1)))
    if delay > 0:
        time.sleep(delay)
    return delay


def _response_text(response: requests.Response) -> str:
    try:
        return (response.text or "")[:12000]
    except Exception:
        return ""


def _classify_ollama_http(response: requests.Response) -> tuple[str, str]:
    """Classify an Ollama HTTP error without relying on exact server wording."""
    status = response.status_code
    body = _response_text(response).lower()

    model_terms = (
        "model not found", "pull the model first", "unknown model", "model not found"
    )
    context_terms = (
        "context length", "context window", "maximum context", "too many tokens",
        "input length", "prompt is too long", "exceeds the context", "num_ctx",
    )
    oom_terms = (
        "out of memory", "out-of-memory", "cuda out of memory", "not enough memory",
        "failed to allocate", "memory allocation", "insufficient memory", "cuda error",
    )
    repeat_terms = (
        "token repeat limit reached",
        "repeat limit reached",
        "prediction aborted, token repeat",
    )

    if any(term in body for term in repeat_terms):
        return "repeat", body
    if status == 404 or any(term in body for term in model_terms):
        return "model_unavailable", body
    if any(term in body for term in context_terms):
        return "context", body
    if any(term in body for term in oom_terms):
        return "oom", body
    if status == 429:
        return "rate_limit", body
    if 500 <= status <= 599:
        return "server", body
    if status in {408}:
        return "transient", body
    if 400 <= status <= 499:
        return "client", body
    return "unknown", body


def _exception_is_transport(exc: Exception) -> bool:
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError))


def _fallback_model_for(model: str) -> str:
    if not REVIEW_MODEL_FALLBACK_ENABLED:
        return ""
    mapping = {
        REVIEW_GENERAL_MODEL: REVIEW_GENERAL_FALLBACK_MODEL,
        REVIEW_CRITICAL_MODEL: REVIEW_CRITICAL_FALLBACK_MODEL,
        REVIEW_COORDINATOR_MODEL: REVIEW_COORDINATOR_FALLBACK_MODEL,
        REVIEW_LANGUAGE_MODEL: REVIEW_LANGUAGE_FALLBACK_MODEL,
        REVIEW_AI_STYLE_MODEL: REVIEW_AI_STYLE_FALLBACK_MODEL,
        REVIEW_HOSTILE_MODEL: REVIEW_HOSTILE_FALLBACK_MODEL,
        REVIEW_VISION_MODEL: REVIEW_VISION_FALLBACK_MODEL,
    }
    candidate = str(mapping.get(model, "") or "").strip()
    if candidate and candidate != model:
        return candidate
    return ""


def _ollama_healthcheck() -> bool:
    try:
        r = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=REVIEW_OLLAMA_HEALTH_TIMEOUT_SECONDS,
            headers={"User-Agent": REVIEW_USER_AGENT},
        )
        return r.ok
    except Exception:
        return False


def _default_think_for_model(model: str) -> bool:
    """Return the configured reasoning policy for a reviewer model."""
    if model == REVIEW_VISION_MODEL:
        return REVIEW_VISION_THINK
    if model == REVIEW_COORDINATOR_MODEL:
        return REVIEW_COORDINATOR_THINK
    if model == REVIEW_LANGUAGE_MODEL:
        return REVIEW_LANGUAGE_THINK
    if model == REVIEW_AI_STYLE_MODEL:
        return REVIEW_AI_STYLE_THINK
    if model == REVIEW_HOSTILE_MODEL:
        return REVIEW_HOSTILE_THINK
    if model == REVIEW_CRITICAL_MODEL:
        return REVIEW_CRITICAL_THINK
    return REVIEW_GENERAL_THINK


def _prompt_source_span(prompt: str) -> tuple[int, int] | None:
    """Find the largest source/evidence-like payload in a review prompt."""
    markers = (
        "\nSOURCE:\n",
        "\nSOURCE EXCERPTS:\n",
        "\nSOURCE TEXT:\n",
        "\nDOCUMENT TEXT:\n",
        "\nDOCUMENT:\n",
        "\nEVIDENCE:\n",
        "\nSPECIALIST REPORTS:\n",
        "\nREVIEW MATERIAL:\n",
    )
    candidates: list[tuple[int, int]] = []
    for marker in markers:
        start = prompt.find(marker)
        while start >= 0:
            payload_start = start + len(marker)
            if len(prompt) - payload_start >= REVIEW_STALLED_LENGTH_MIN_PROMPT_CHARS:
                candidates.append((payload_start, len(prompt)))
            start = prompt.find(marker, payload_start)
    if not candidates:
        return None
    return max(candidates, key=lambda span: span[1] - span[0])


def _shrink_prompt_payload(prompt: str, target_chars: int) -> tuple[str, bool, int]:
    """Shrink only the largest source payload while preserving prompt instructions."""
    span = _prompt_source_span(prompt)
    if not span:
        return prompt, False, 0

    start, end = span
    source = prompt[start:end]
    target_chars = max(2000, int(target_chars))
    if target_chars >= len(source):
        return prompt, False, len(source)

    # Prefer page/block-aware compression so a data-review prompt retains coverage
    # across the document instead of dropping only the middle or only the end.
    pieces = re.split(r"(?=\n(?:PAGE\s+\d+|---\s*PAGE\s+\d+))", source)
    pieces = [p for p in pieces if p]
    omitted_marker = "\n\n[INPUT REDUCED FOR CONTEXT-CONSTRAINED RECOVERY]\n\n"

    if len(pieces) >= 2:
        # Allocate the target proportionally across pages/blocks, with a floor so that
        # small pages are not erased completely.
        available = max(2000, target_chars - len(omitted_marker))
        total = sum(len(p) for p in pieces)
        kept: list[str] = []
        for piece in pieces:
            share = max(600, int(available * (len(piece) / max(1, total))))
            kept.append(piece[:share])
        candidate_source = omitted_marker.join(kept)
        if len(candidate_source) > target_chars:
            candidate_source = candidate_source[:target_chars]
    else:
        head = int(target_chars * 0.70)
        tail = max(0, target_chars - head - len(omitted_marker))
        candidate_source = source[:head] + omitted_marker + (source[-tail:] if tail else "")

    new_prompt = prompt[:start] + candidate_source + prompt[end:]
    if len(new_prompt) >= len(prompt):
        return prompt, False, len(source)
    return new_prompt, True, len(source)


def _looks_context_constrained(
    prompt_eval_count: Any,
    eval_count: Any,
    attempt_tokens: int,
    current_ctx: int,
) -> bool:
    """Detect accepted generations truncated by the input already filling the context."""
    try:
        prompt_eval = int(prompt_eval_count)
        eval_tokens = int(eval_count)
    except (TypeError, ValueError):
        return False
    if prompt_eval <= 0 or eval_tokens <= 0:
        return False

    combined = prompt_eval + eval_tokens
    output_far_below_request = eval_tokens < max(256, int(attempt_tokens * 0.70))
    prompt_uses_most_context = prompt_eval >= int(current_ctx * 0.78)
    context_almost_full = combined >= int(current_ctx * 0.95)
    return output_far_below_request and prompt_uses_most_context and context_almost_full


def _build_retry_budgets(base_tokens: int) -> list[int]:
    """Build increasing generation budgets up to REVIEW_LLM_MAX_TOKENS.

    Example with base=4500, multiplier=1.5, max=12000:
        [4500, 6750, 10125, 12000]

    The final entry is always the configured ceiling when the initial budget is
    below it. This is what allows a reasoning-heavy request to keep thinking
    rather than switching to no-thinking too early.
    """
    base = max(1, int(base_tokens))
    ceiling = max(base, REVIEW_LLM_MAX_TOKENS)
    budgets = [min(base, ceiling)]
    while budgets[-1] < ceiling:
        current = budgets[-1]
        nxt = int(round(current * REVIEW_LLM_RETRY_TOKEN_MULTIPLIER))
        if nxt <= current:
            nxt = current + 1
        nxt = min(ceiling, nxt)
        budgets.append(nxt)
    return budgets




def _document_source_hash(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _build_summary_input(pages: list[PageRecord], outline: str, max_chars: int | None = None) -> str:
    """Create a high-coverage summary input without silently dropping whole sections."""
    safe_chars = int(max_chars or REVIEW_CONTEXT_SUMMARY_INPUT_CHARS)
    pieces: list[str] = [f"DOCUMENT OUTLINE:\n{outline[:min(12000, safe_chars // 3)]}"]
    per_page_cap = max(700, int(safe_chars / max(1, len(pages))))
    for p in pages:
        text = re.sub(r"\s+", " ", p.text).strip()
        if len(text) <= per_page_cap:
            excerpt = text
        else:
            head = int(per_page_cap * 0.62)
            tail = max(200, per_page_cap - head)
            excerpt = text[:head] + " [..page-middle omitted..] " + text[-tail:]
        pieces.append(f"\n--- PAGE {p.page} ---\n{excerpt}")
    result = "\n".join(pieces)
    return result[:safe_chars]


def build_context_pack(pages: list[PageRecord], outline: str, references: list[str], session_dir: Path) -> dict[str, Any]:
    """Build/load a reusable review context summary once per document."""
    global ACTIVE_CONTEXT_SUMMARY, ACTIVE_CONTEXT_PACK_PATH
    ACTIVE_CONTEXT_SUMMARY = ""
    ACTIVE_CONTEXT_PACK_PATH = None
    pack_path = session_dir / "context_pack.json"
    # The session PDF hash is supplied by the caller through the session directory cache marker.
    # For backward compatibility, also tolerate an existing pack without a hash.
    if REVIEW_CONTEXT_REUSE and pack_path.exists():
        try:
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
            if pack.get("version") == REVIEW_CONTEXT_PACK_VERSION and pack.get("summary"):
                ACTIVE_CONTEXT_SUMMARY = str(pack["summary"])
                ACTIVE_CONTEXT_PACK_PATH = pack_path
                return pack
        except Exception:
            pass

    # A single compact synthesis is cheaper than repeatedly feeding 30-50k chars to every
    # specialist. Keep enough input room for the requested output inside Ollama's context.
    # The raw PDF remains authoritative for targeted checks.
    summary_input_tokens = max(1800, int(REVIEW_CONTEXT_SUMMARY_CTX * 0.68))
    summary_budget_chars = min(
        REVIEW_CONTEXT_SUMMARY_INPUT_CHARS,
        max(9000, int(summary_input_tokens * 3.0)),
    )
    summary_source = _build_summary_input(pages, outline, max_chars=summary_budget_chars)
    prompt = f"""
You are the CONTEXT-BUILDER for a rigorous academic PDF review system.
Create a dense, factual review context for downstream specialist agents.
Do NOT write a generic abstract. Preserve exact technical details that reviewers may need.

Preserve, where present:
- title/problem/application
- exact methods, algorithms, model names, architectures and system components
- equations/variable names that can be read from the text
- datasets, sample sizes and experimental conditions
- exact numerical results, ranges, percentages, units and reported errors
- figure/table/equation numbers and what each contains
- major claims and what evidence is claimed for each
- baselines/comparators
- limitations stated by the authors
- section structure and major topic of each section

Output concise bullet sections:
1. PAPER MAP
2. METHODS / SYSTEM
3. EXPERIMENTAL SETUP
4. KEY NUMERICAL EVIDENCE
5. FIGURES / TABLES / EQUATIONS
6. MAIN CLAIMS AND CONTRIBUTIONS
7. LIMITATIONS / RISKS STATED BY AUTHORS
8. TERMS / ACRONYMS / SYMBOLS

Never invent a value. If something is unclear, say "not clear from extracted text".
The raw PDF remains authoritative; this context is a navigation/compression aid.

{summary_source}
"""
    summary = ollama_chat(
        REVIEW_CONTEXT_SUMMARY_MODEL,
        prompt,
        ctx=REVIEW_CONTEXT_SUMMARY_CTX,
        tokens=REVIEW_CONTEXT_SUMMARY_TOKENS,
        label="pre-review smart context summary",
        think=False,
        preprocess=False,
        timeout=REVIEW_SHORT_TIMEOUT if ACTIVE_MODE == "short" else REVIEW_LLM_TIMEOUT,
    )
    pack = {
        "version": REVIEW_CONTEXT_PACK_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode_created": ACTIVE_MODE,
        "summary": summary,
        "outline": outline,
        "reference_count": len(references),
        "pages": [
            {"page": p.page, "visual_needed": p.visual_needed, "math": p.has_math_signals,
             "text_chars": len(p.text)} for p in pages
        ],
    }
    save_json(pack_path, pack)
    ACTIVE_CONTEXT_SUMMARY = summary
    ACTIVE_CONTEXT_PACK_PATH = pack_path
    save_text(session_dir / "context_pack.md", summary)
    return pack


def _runtime_source_target(label: str) -> int | None:
    """Return a proactive source character target for a specialist prompt."""
    low = label.lower()
    if ACTIVE_MODE == "short":
        if "coordinator" in low or "arbiter" in low or "hostile" in low:
            return REVIEW_SHORT_SOURCE_TARGET_COORDINATOR
        if any(k in low for k in ("technical", "equation", "mathemat", "statistics", "experiment", "reproduc", "novelty")):
            return REVIEW_SHORT_SOURCE_TARGET_CRITICAL
        return REVIEW_SHORT_SOURCE_TARGET_GENERAL
    if not REVIEW_FULL_PROACTIVE_CONTEXT:
        return None
    if "coordinator" in low or "arbiter" in low or "hostile" in low:
        return REVIEW_FULL_SOURCE_TARGET_COORDINATOR
    if any(k in low for k in ("technical", "equation", "mathemat", "statistics", "experiment", "reproduc", "novelty")):
        return REVIEW_FULL_SOURCE_TARGET_CRITICAL
    return REVIEW_FULL_SOURCE_TARGET_GENERAL


def _prepare_prompt_for_runtime(prompt: str, label: str) -> str:
    """Proactively compact large source payloads before Ollama hits the limit."""
    if not ACTIVE_CONTEXT_SUMMARY:
        return prompt
    target = _runtime_source_target(label)
    if target is None:
        return prompt
    span = _prompt_source_span(prompt)
    if not span:
        return prompt
    start, end = span
    source_len = end - start
    if source_len <= target:
        return prompt
    reduced_prompt, changed, _ = _shrink_prompt_payload(prompt, target)
    if not changed:
        return prompt
    insert = f"\n\nSMART DOCUMENT CONTEXT (compression/navigation aid; raw source excerpts below are authoritative):\n{ACTIVE_CONTEXT_SUMMARY}\n"
    return reduced_prompt[:start] + insert + reduced_prompt[start:]


def ollama_chat(model: str, prompt: str, *, images: list[str] | None = None,
                ctx: int, tokens: int, label: str, timeout: float | None = None,
                keep_alive: str | int = "5m", think: bool | str | None = None,
                retries: int | None = None, preprocess: bool = True) -> str:
    """Robust Ollama call with independent recovery policies.

    Generation-limit failures:
      - done_reason == length or empty final content -> increase num_predict
        while preserving context and thinking, up to the token ceiling.

    Timeout failures:
      - timeout #1 -> 1.5x original wall-clock timeout, same full context and tokens,
        thinking remains ON.
      - timeout #2 -> 2.0x original timeout, same full context and tokens,
        thinking remains ON.
      - timeout #3 -> same 2.0x timeout/context, thinking OFF, final token budget 1.5x.

    Other failures:
      - transient connection / 429 / 5xx -> exponential backoff and same request.
      - context overflow -> reduce context and retry without changing tokens/thinking.
      - OOM/CUDA-memory errors -> reduce context and retry; optionally use a configured
        fallback model after the context-recovery ceiling is reached.
      - model-not-found -> immediately use the configured fallback model if available.
      - malformed/unexpected JSON -> retry the same request with backoff.
      - token-repeat abort -> change sampling parameters, then use a final no-thinking
        recovery with a 1.5x token budget instead of replaying the exact same request.
      - accepted `done_reason=length` with a nearly full input context -> classify as
        context-constrained generation, shrink only the source payload, reset the token
        ladder and retry with thinking preserved.
      - non-retryable 4xx -> fail clearly rather than wasting generation retries.
    """
    configured_think = _default_think_for_model(model) if think is None else think
    original_model = model
    active_model = model
    base_tokens = max(1, int(tokens))
    budgets = _build_retry_budgets(base_tokens)

    retry_cap = REVIEW_LLM_MAX_RETRIES if retries is None else max(0, int(retries))
    if retry_cap > 0:
        budgets = budgets[:retry_cap + 1]

    original_timeout = float(timeout or REVIEW_LLM_TIMEOUT)
    current_ctx = max(1024, int(ctx))
    current_timeout = original_timeout
    active_prompt = _prepare_prompt_for_runtime(prompt, label) if preprocess else prompt
    timeout_count = 0
    forced_no_think = False
    budget_index = 0
    transient_retries = 0
    response_retries = 0
    context_recoveries = 0
    oom_recoveries = 0
    model_fallback_used = False
    repeat_recoveries = 0
    repeat_forced_no_think = False
    repeat_temperature: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    stalled_length_recoveries = 0
    last_length_signature: tuple[int, int, str] | None = None

    while True:
        attempt_tokens = budgets[budget_index]
        # After the timeout ladder reaches the final no-thinking fallback, use the
        # timeout-adjusted token budget but never restart the normal reasoning ladder.
        if forced_no_think or repeat_forced_no_think:
            current_think = False
        else:
            current_think = configured_think

        total_attempts = len(budgets)
        attempt_no = budget_index + 1
        payload: dict[str, Any] = {
            "model": active_model,
            "messages": [{"role": "user", "content": active_prompt}],
            "stream": False,
            "think": current_think,
            "keep_alive": keep_alive,
            "options": {
                "temperature": repeat_temperature if repeat_temperature is not None else 0,
                "num_ctx": current_ctx,
                "num_predict": attempt_tokens,
            },
        }
        if repeat_penalty is not None:
            payload["options"]["repeat_penalty"] = repeat_penalty
        if repeat_last_n is not None:
            payload["options"]["repeat_last_n"] = repeat_last_n
        if images:
            payload["messages"][0]["images"] = images

        label_attempt = f"{label} | attempt {attempt_no}/{total_attempts}"
        started = time.monotonic()
        log(
            "AGENT",
            f"{label_attempt} | {active_model} | think={current_think} | ctx={current_ctx} | "
            f"timeout={current_timeout:.0f}s | max_tokens={attempt_tokens} | working...",
            "blue",
        )

        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=current_timeout,
                headers={"User-Agent": REVIEW_USER_AGENT},
            )

            if not r.ok:
                category, body = _classify_ollama_http(r)
                status = r.status_code
                if category == "model_unavailable":
                    raise OllamaModelUnavailableError(
                        f"HTTP {status}: Ollama model unavailable: {body[:600]}"
                    )
                if category == "context":
                    raise OllamaContextError(f"HTTP {status}: context limit: {body[:600]}")
                if category == "oom":
                    raise OllamaOOMError(f"HTTP {status}: memory failure: {body[:600]}")
                if category == "repeat":
                    raise OllamaRepeatError(f"HTTP {status}: token repeat limit reached: {body[:600]}")
                if category in {"rate_limit", "server", "transient"}:
                    raise requests.exceptions.HTTPError(
                        f"HTTP {status}: transient Ollama error: {body[:600]}", response=r
                    )
                raise requests.exceptions.HTTPError(
                    f"HTTP {status}: non-retryable Ollama error: {body[:600]}", response=r
                )

            try:
                data = r.json()
            except ValueError as exc:
                response_retries += 1
                if response_retries <= REVIEW_OLLAMA_RESPONSE_RETRIES:
                    delay = _sleep_backoff(
                        REVIEW_OLLAMA_BACKOFF_INITIAL_SECONDS,
                        response_retries,
                        REVIEW_OLLAMA_BACKOFF_MAX_SECONDS,
                    )
                    log(
                        "AGENT",
                        f"{label_attempt} | malformed Ollama JSON | retry {response_retries}/"
                        f"{REVIEW_OLLAMA_RESPONSE_RETRIES} after {delay:.1f}s with same settings...",
                        "yellow",
                    )
                    continue
                raise RuntimeError(f"Ollama returned malformed JSON after retries: {exc}") from exc

            if not isinstance(data, dict) or not isinstance(data.get("message") or {}, dict):
                response_retries += 1
                if response_retries <= REVIEW_OLLAMA_RESPONSE_RETRIES:
                    delay = _sleep_backoff(
                        REVIEW_OLLAMA_BACKOFF_INITIAL_SECONDS,
                        response_retries,
                        REVIEW_OLLAMA_BACKOFF_MAX_SECONDS,
                    )
                    log(
                        "AGENT",
                        f"{label_attempt} | unexpected Ollama response schema | retry "
                        f"{response_retries}/{REVIEW_OLLAMA_RESPONSE_RETRIES} after {delay:.1f}s...",
                        "yellow",
                    )
                    continue
                raise RuntimeError("Ollama returned an unexpected response schema")

        except requests.exceptions.Timeout as exc:
            timeout_count += 1

            if timeout_count == 1 and REVIEW_TIMEOUT_MAX_RECOVERIES >= 1:
                current_timeout = original_timeout * REVIEW_TIMEOUT_FIRST_MULTIPLIER
                log(
                    "AGENT",
                    f"{label_attempt} | timeout recovery 1/{REVIEW_TIMEOUT_MAX_RECOVERIES} | "
                    f"keeping ctx={current_ctx} | timeout->{current_timeout:.0f}s | "
                    f"keeping think={configured_think} | keeping max_tokens={attempt_tokens} | "
                    f"retrying same generation attempt...",
                    "yellow",
                )
                continue

            if timeout_count == 2 and REVIEW_TIMEOUT_MAX_RECOVERIES >= 2:
                current_timeout = original_timeout * REVIEW_TIMEOUT_SECOND_MULTIPLIER
                log(
                    "AGENT",
                    f"{label_attempt} | timeout recovery 2/{REVIEW_TIMEOUT_MAX_RECOVERIES} | "
                    f"keeping ctx={current_ctx} | timeout->{current_timeout:.0f}s | "
                    f"keeping think={configured_think} | keeping max_tokens={attempt_tokens} | "
                    f"retrying same generation attempt...",
                    "yellow",
                )
                continue

            if timeout_count == 3 and REVIEW_TIMEOUT_FALLBACK_NO_THINK:
                forced_no_think = True
                fallback_tokens = min(
                    REVIEW_LLM_MAX_TOKENS,
                    max(attempt_tokens, int(round(attempt_tokens * REVIEW_TIMEOUT_FALLBACK_TOKEN_MULTIPLIER))),
                )
                log(
                    "AGENT",
                    f"{label_attempt} | timeout recovery 3/{REVIEW_TIMEOUT_MAX_RECOVERIES} | "
                    f"keeping ctx={current_ctx} | keeping timeout={current_timeout:.0f}s | "
                    f"switching to think=False | max_tokens {attempt_tokens}->{fallback_tokens} | "
                    f"retrying same generation attempt...",
                    "yellow",
                )
                budgets[budget_index] = fallback_tokens
                continue

            raise RuntimeError(
                f"Ollama request timed out after {timeout_count} timeout recovery attempts: {exc}"
            ) from exc

        except OllamaContextError as exc:
            if context_recoveries < REVIEW_CONTEXT_ERROR_MAX_RECOVERIES:
                old_ctx = current_ctx
                reduced = int(round(current_ctx * REVIEW_CONTEXT_ERROR_REDUCTION))
                current_ctx = max(REVIEW_CONTEXT_ERROR_MIN, reduced)
                if current_ctx < old_ctx:
                    context_recoveries += 1
                    transient_retries = 0
                    log(
                        "AGENT",
                        f"{label_attempt} | context-size failure | recovery {context_recoveries}/"
                        f"{REVIEW_CONTEXT_ERROR_MAX_RECOVERIES} | ctx {old_ctx}->{current_ctx} | "
                        f"keeping think={current_think} and max_tokens={attempt_tokens} | retrying...",
                        "yellow",
                    )
                    continue
            fallback_model = _fallback_model_for(active_model)
            if fallback_model and not model_fallback_used:
                old_model = active_model
                active_model = fallback_model
                model_fallback_used = True
                log(
                    "AGENT",
                    f"{label_attempt} | context failure persisted | switching model {old_model}->{active_model} "
                    f"and retrying with ctx={current_ctx} | think={current_think} | max_tokens={attempt_tokens}...",
                    "yellow",
                )
                continue
            raise RuntimeError(f"Ollama context-size failure after recovery attempts: {exc}") from exc

        except OllamaOOMError as exc:
            if oom_recoveries < REVIEW_OOM_MAX_RECOVERIES:
                old_ctx = current_ctx
                reduced = int(round(current_ctx * REVIEW_OOM_CONTEXT_REDUCTION))
                current_ctx = max(REVIEW_OOM_CONTEXT_MIN, reduced)
                if current_ctx < old_ctx:
                    oom_recoveries += 1
                    transient_retries = 0
                    log(
                        "AGENT",
                        f"{label_attempt} | GPU/CPU memory failure | recovery {oom_recoveries}/"
                        f"{REVIEW_OOM_MAX_RECOVERIES} | ctx {old_ctx}->{current_ctx} | "
                        f"keeping think={current_think} and max_tokens={attempt_tokens} | retrying...",
                        "yellow",
                    )
                    continue
            fallback_model = _fallback_model_for(active_model)
            if fallback_model and not model_fallback_used:
                old_model = active_model
                active_model = fallback_model
                model_fallback_used = True
                log(
                    "AGENT",
                    f"{label_attempt} | memory failure persisted | switching model {old_model}->{active_model} "
                    f"and retrying with ctx={current_ctx} | think={current_think} | max_tokens={attempt_tokens}...",
                    "yellow",
                )
                continue
            raise RuntimeError(f"Ollama memory failure after recovery attempts: {exc}") from exc

        except OllamaModelUnavailableError as exc:
            fallback_model = _fallback_model_for(active_model)
            if fallback_model and not model_fallback_used:
                old_model = active_model
                active_model = fallback_model
                model_fallback_used = True
                log(
                    "AGENT",
                    f"{label_attempt} | model unavailable | switching {old_model}->{active_model} "
                    f"and retrying same ctx/tokens/think...",
                    "yellow",
                )
                continue
            raise RuntimeError(f"Ollama model unavailable and no fallback model configured: {exc}") from exc

        except OllamaRepeatError as exc:
            repeat_recoveries += 1
            if repeat_recoveries == 1 and REVIEW_REPEAT_MAX_RECOVERIES >= 1:
                repeat_temperature = REVIEW_REPEAT_TEMPERATURE_1
                repeat_penalty = REVIEW_REPEAT_PENALTY_1
                repeat_last_n = REVIEW_REPEAT_LAST_N
                log(
                    "AGENT",
                    f"{label_attempt} | token-repeat failure | recovery 1/{REVIEW_REPEAT_MAX_RECOVERIES} | "
                    f"temperature 0->{repeat_temperature:.2f} | repeat_penalty={repeat_penalty:.2f} | "
                    f"repeat_last_n={repeat_last_n} | keeping think={current_think} | "
                    f"retrying same ctx/max_tokens...",
                    "yellow",
                )
                continue

            if repeat_recoveries == 2 and REVIEW_REPEAT_MAX_RECOVERIES >= 2:
                repeat_temperature = REVIEW_REPEAT_TEMPERATURE_2
                repeat_penalty = REVIEW_REPEAT_PENALTY_2
                repeat_last_n = REVIEW_REPEAT_LAST_N
                log(
                    "AGENT",
                    f"{label_attempt} | token-repeat failure | recovery 2/{REVIEW_REPEAT_MAX_RECOVERIES} | "
                    f"temperature={repeat_temperature:.2f} | repeat_penalty={repeat_penalty:.2f} | "
                    f"repeat_last_n={repeat_last_n} | keeping think={current_think} | "
                    f"retrying same ctx/max_tokens...",
                    "yellow",
                )
                continue

            if repeat_recoveries >= 3 and REVIEW_REPEAT_MAX_RECOVERIES >= 3:
                fallback_tokens = min(
                    REVIEW_LLM_MAX_TOKENS,
                    max(attempt_tokens, int(round(attempt_tokens * REVIEW_REPEAT_FALLBACK_TOKEN_MULTIPLIER))),
                )
                repeat_forced_no_think = True
                repeat_temperature = REVIEW_REPEAT_FALLBACK_TEMPERATURE
                repeat_penalty = REVIEW_REPEAT_FALLBACK_PENALTY
                repeat_last_n = REVIEW_REPEAT_LAST_N
                budgets[budget_index] = fallback_tokens
                log(
                    "AGENT",
                    f"{label_attempt} | token-repeat ceiling reached | final recovery | "
                    f"think=False | max_tokens {attempt_tokens}->{fallback_tokens} | "
                    f"temperature={repeat_temperature:.2f} | repeat_penalty={repeat_penalty:.2f} | "
                    f"repeat_last_n={repeat_last_n} | retrying same ctx...",
                    "yellow",
                )
                continue

            raise RuntimeError(f"Ollama token-repeat failure after {repeat_recoveries} recoveries: {exc}") from exc

        except requests.exceptions.HTTPError as exc:
            response = getattr(exc, "response", None)
            status = response.status_code if response is not None else None
            if status in {429, 408} or (status is not None and 500 <= status <= 599):
                transient_retries += 1
                if transient_retries <= REVIEW_OLLAMA_TRANSIENT_RETRIES:
                    delay = _sleep_backoff(
                        REVIEW_OLLAMA_BACKOFF_INITIAL_SECONDS,
                        transient_retries,
                        REVIEW_OLLAMA_BACKOFF_MAX_SECONDS,
                    )
                    log(
                        "AGENT",
                        f"{label_attempt} | transient HTTP failure status={status} | "
                        f"retry {transient_retries}/{REVIEW_OLLAMA_TRANSIENT_RETRIES} after {delay:.1f}s "
                        f"with same ctx/think/max_tokens...",
                        "yellow",
                    )
                    continue
                raise RuntimeError(f"Ollama transient HTTP failure after retries: {exc}") from exc
            raise RuntimeError(f"Ollama HTTP failure: {exc}") from exc

        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
            transient_retries += 1
            if transient_retries <= REVIEW_OLLAMA_TRANSIENT_RETRIES:
                delay = _sleep_backoff(
                    REVIEW_OLLAMA_BACKOFF_INITIAL_SECONDS,
                    transient_retries,
                    REVIEW_OLLAMA_BACKOFF_MAX_SECONDS,
                )
                healthy = _ollama_healthcheck()
                health_msg = "Ollama reachable" if healthy else "Ollama healthcheck still unavailable"
                log(
                    "AGENT",
                    f"{label_attempt} | connection failure | retry {transient_retries}/"
                    f"{REVIEW_OLLAMA_TRANSIENT_RETRIES} after {delay:.1f}s | {health_msg} | "
                    f"same ctx/think/max_tokens...",
                    "yellow",
                )
                continue
            raise RuntimeError(f"Ollama connection failed after retries: {exc}") from exc

        except Exception as exc:
            # Catch other transport/runtime exceptions conservatively. Do NOT alter the
            # token ladder for a generic runtime failure unless it is an actual model
            # generation-limit response handled below.
            transient_retries += 1
            if transient_retries <= REVIEW_OLLAMA_TRANSIENT_RETRIES:
                delay = _sleep_backoff(
                    REVIEW_OLLAMA_BACKOFF_INITIAL_SECONDS,
                    transient_retries,
                    REVIEW_OLLAMA_BACKOFF_MAX_SECONDS,
                )
                log(
                    "AGENT",
                    f"{label_attempt} | unexpected Ollama failure: {exc} | retry "
                    f"{transient_retries}/{REVIEW_OLLAMA_TRANSIENT_RETRIES} after {delay:.1f}s "
                    f"with same settings...",
                    "yellow",
                )
                continue
            raise RuntimeError(f"Ollama inference failed after retries: {exc}") from exc

        # Successful HTTP response: inspect model termination metadata.
        message = data.get("message") or {}
        content = str(message.get("content") or "").strip()
        thinking = str(message.get("thinking") or "").strip()
        done_reason = str(data.get("done_reason") or "")
        eval_count = data.get("eval_count")
        prompt_eval_count = data.get("prompt_eval_count")
        truncated = done_reason.lower() == "length"

        if content and not truncated:
            log(
                "AGENT",
                f"{label_attempt} | completed in {time.monotonic() - started:.1f}s"
                f" | prompt_eval={prompt_eval_count} | eval={eval_count} | done={done_reason or 'n/a'}",
                "green",
            )
            return content

        if truncated:
            log(
                "AGENT",
                f"{label_attempt} | generation hit token limit"
                f" | content_chars={len(content)} | thinking_chars={len(thinking)}"
                f" | prompt_eval={prompt_eval_count} | eval={eval_count} | done={done_reason or 'n/a'}",
                "yellow",
            )
        else:
            log(
                "AGENT",
                f"{label_attempt} | empty final response"
                f" | thinking_chars={len(thinking)} | prompt_eval={prompt_eval_count}"
                f" | eval={eval_count} | done={done_reason or 'n/a'}",
                "yellow",
            )

        # Critical recovery distinction: if the request was accepted but the model used
        # almost the entire context window for prompt + output, increasing num_predict
        # cannot create more room. Shrink the source payload instead, preserve the full
        # model context size, and restart the token ladder from the original budget.
        length_signature = (
            len(content),
            int(eval_count) if isinstance(eval_count, (int, float)) else -1,
            hashlib.sha1(content.encode("utf-8", errors="ignore")[:4000]).hexdigest(),
        )
        same_stall = truncated and last_length_signature == length_signature
        last_length_signature = length_signature if truncated else None
        context_constrained = truncated and (
            _looks_context_constrained(prompt_eval_count, eval_count, attempt_tokens, current_ctx)
            or same_stall
        )

        if context_constrained and stalled_length_recoveries < REVIEW_STALLED_LENGTH_MAX_RECOVERIES:
            try:
                prompt_tokens = int(prompt_eval_count or 0)
            except (TypeError, ValueError):
                prompt_tokens = 0

            # Reserve room for the current generation budget, but never demand an
            # unrealistically tiny source. The target is based on the actual prompt token
            # count reported by Ollama, so it adapts to tokenization better than a fixed
            # character limit.
            reserve_tokens = min(attempt_tokens, max(2048, int(current_ctx * 0.45)))
            target_prompt_tokens = max(4096, int(current_ctx - reserve_tokens))
            if prompt_tokens > 0:
                target_chars = int(len(active_prompt) * (target_prompt_tokens / prompt_tokens))
            else:
                target_chars = int(len(active_prompt) * REVIEW_STALLED_LENGTH_PROMPT_TARGET_FRACTION)

            floor_chars = max(
                REVIEW_STALLED_LENGTH_MIN_PROMPT_CHARS,
                int(len(active_prompt) * REVIEW_STALLED_LENGTH_MIN_PROMPT_FRACTION),
            )
            target_chars = max(floor_chars, target_chars)
            target_chars = min(target_chars, int(len(active_prompt) * 0.90))

            reduced_prompt, changed, source_chars = _shrink_prompt_payload(active_prompt, target_chars)
            if changed and len(reduced_prompt) < len(active_prompt):
                old_len = len(active_prompt)
                active_prompt = reduced_prompt
                stalled_length_recoveries += 1
                budget_index = 0
                transient_retries = 0
                response_retries = 0
                log(
                    "AGENT",
                    f"{label} | context-constrained generation detected | recovery "
                    f"{stalled_length_recoveries}/{REVIEW_STALLED_LENGTH_MAX_RECOVERIES} | "
                    f"prompt_chars {old_len}->{len(active_prompt)} | source_chars={source_chars} | "
                    f"prompt_eval={prompt_eval_count} | eval={eval_count} | "
                    f"keeping ctx={current_ctx} | restarting token ladder at "
                    f"{budgets[0]} tokens | think={current_think}...",
                    "yellow",
                )
                continue

        if forced_no_think and timeout_count >= 3:
            raise RuntimeError(
                "timeout fallback returned no complete response; "
                f"content_chars={len(content)}, prompt_eval={prompt_eval_count}, "
                f"eval={eval_count}, done={done_reason or 'n/a'}"
            )

        if budget_index + 1 < len(budgets):
            next_tokens = budgets[budget_index + 1]
            log(
                "AGENT",
                f"{label} | increasing reasoning budget to {next_tokens} tokens "
                f"and retrying with think={current_think}...",
                "yellow",
            )
            budget_index += 1
            continue

        if current_think not in (False, "none") and REVIEW_LLM_FALLBACK_NO_THINK:
            forced_no_think = True
            timeout_recoveries_note = " with timeout-recovery settings" if timeout_count else ""
            log(
                "AGENT",
                f"{label} | reasoning ceiling reached | fallback recovery | think=False | "
                f"ctx={current_ctx} | timeout={current_timeout:.0f}s | max_tokens={attempt_tokens}"
                f"{timeout_recoveries_note} | retrying...",
                "yellow",
            )
            continue

        raise RuntimeError(
            "no complete model response after exhausting configured recovery paths; "
            f"content_chars={len(content)}, thinking_chars={len(thinking)}, "
            f"prompt_eval={prompt_eval_count}, eval={eval_count}, done={done_reason or 'n/a'}"
        )


        raise RuntimeError(
            "no complete model response after exhausting configured recovery paths; "
            f"content_chars={len(content)}, thinking_chars={len(thinking)}, "
            f"eval={eval_count}, done={done_reason or 'n/a'}"
        )


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Web research helpers
# -----------------------------------------------------------------------------
def web_search(query: str, limit: int = REVIEW_WEB_RESULTS_PER_QUERY) -> dict[str, Any]:
    """Search the web with retries; distinguish failure from zero results."""
    if DDGS is None:
        return {
            "status": "unavailable",
            "attempts": 0,
            "results": [],
            "error": "ddgs package is unavailable",
        }

    last_error = ""
    for attempt in range(1, REVIEW_WEB_RETRIES + 2):
        try:
            with DDGS() as ddgs:
                out = []
                for item in ddgs.text(query, max_results=limit):
                    out.append({
                        "title": str(item.get("title") or ""),
                        "url": str(item.get("href") or item.get("url") or ""),
                        "snippet": str(item.get("body") or item.get("snippet") or ""),
                    })
            log("WEB", f"search succeeded | attempt {attempt} | results={len(out)} | query={query!r}", "green")
            return {"status": "success", "attempts": attempt, "results": out}
        except Exception as exc:
            last_error = str(exc)
            if attempt <= REVIEW_WEB_RETRIES:
                delay = _sleep_backoff(
                    REVIEW_WEB_BACKOFF_INITIAL_SECONDS,
                    attempt,
                    REVIEW_WEB_BACKOFF_MAX_SECONDS,
                )
                log(
                    "WEB",
                    f"search failed | attempt {attempt}/{REVIEW_WEB_RETRIES + 1} | "
                    f"retrying after {delay:.1f}s | query={query!r} | error={exc}",
                    "yellow",
                )
            else:
                log("WEB", f"search failed permanently after {attempt} attempt(s) | query={query!r} | error={exc}", "red")
    return {"status": "failed", "attempts": REVIEW_WEB_RETRIES + 1, "results": [], "error": last_error}


def crossref_search(query: str, limit: int = REVIEW_WEB_RESULTS_PER_QUERY) -> dict[str, Any]:
    """Academic-search fallback using Crossref when general web search fails."""
    last_error = ""
    for attempt in range(1, REVIEW_CROSSREF_RETRIES + 2):
        try:
            r = requests.get(
                "https://api.crossref.org/works",
                params={"query.bibliographic": query[:700], "rows": limit},
                headers={"User-Agent": REVIEW_USER_AGENT},
                timeout=REVIEW_WEB_TIMEOUT,
            )
            if not r.ok:
                if r.status_code == 429 or 500 <= r.status_code <= 599:
                    raise requests.exceptions.HTTPError(
                        f"Crossref HTTP {r.status_code}: {_response_text(r)[:500]}", response=r
                    )
                r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            out = []
            for x in items:
                out.append({
                    "title": " ".join(x.get("title") or []) if isinstance(x.get("title"), list) else str(x.get("title") or ""),
                    "url": str(x.get("URL") or ""),
                    "snippet": str(x.get("container-title") or ""),
                    "DOI": str(x.get("DOI") or ""),
                    "year": ((x.get("published-print") or x.get("published-online") or {}).get("date-parts") or [[None]])[0][0],
                    "publisher": str(x.get("publisher") or ""),
                })
            return {"status": "crossref_success", "attempts": attempt, "results": out}
        except Exception as exc:
            last_error = str(exc)
            if attempt <= REVIEW_CROSSREF_RETRIES:
                delay = _sleep_backoff(
                    REVIEW_CROSSREF_BACKOFF_INITIAL_SECONDS,
                    attempt,
                    REVIEW_CROSSREF_BACKOFF_MAX_SECONDS,
                )
                log("WEB", f"Crossref fallback failed | attempt {attempt}/{REVIEW_CROSSREF_RETRIES + 1} | retrying after {delay:.1f}s | error={exc}", "yellow")
            else:
                log("WEB", f"Crossref fallback failed permanently | error={exc}", "red")
    return {"status": "crossref_failed", "attempts": REVIEW_CROSSREF_RETRIES + 1, "results": [], "error": last_error}


def crossref_lookup(reference: str) -> dict[str, Any]:
    """Crossref lookup with retry/backoff and explicit unresolved status."""
    last_error = ""
    for attempt in range(1, REVIEW_CROSSREF_RETRIES + 2):
        try:
            r = requests.get(
                "https://api.crossref.org/works",
                params={"query.bibliographic": reference[:700], "rows": REVIEW_CITATION_RESULTS},
                headers={"User-Agent": REVIEW_USER_AGENT},
                timeout=REVIEW_WEB_TIMEOUT,
            )
            if not r.ok:
                if r.status_code == 429 or 500 <= r.status_code <= 599:
                    raise requests.exceptions.HTTPError(
                        f"Crossref HTTP {r.status_code}: {_response_text(r)[:500]}", response=r
                    )
                r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            top = []
            for x in items:
                top.append({
                    "title": " ".join(x.get("title") or []) if isinstance(x.get("title"), list) else str(x.get("title") or ""),
                    "DOI": str(x.get("DOI") or ""),
                    "publisher": str(x.get("publisher") or ""),
                    "container": " ".join(x.get("container-title") or []) if isinstance(x.get("container-title"), list) else str(x.get("container-title") or ""),
                    "year": ((x.get("published-print") or x.get("published-online") or {}).get("date-parts") or [[None]])[0][0],
                    "url": str(x.get("URL") or ""),
                })
            return {"reference": reference, "status": "success", "attempts": attempt, "matches": top}
        except Exception as exc:
            last_error = str(exc)
            if attempt <= REVIEW_CROSSREF_RETRIES:
                delay = _sleep_backoff(
                    REVIEW_CROSSREF_BACKOFF_INITIAL_SECONDS,
                    attempt,
                    REVIEW_CROSSREF_BACKOFF_MAX_SECONDS,
                )
                log("CITATIONS", f"Crossref failed | attempt {attempt}/{REVIEW_CROSSREF_RETRIES + 1} | retrying after {delay:.1f}s", "yellow")
            else:
                return {"reference": reference, "status": "failed", "attempts": attempt, "matches": [], "error": last_error}
    return {"reference": reference, "status": "failed", "attempts": REVIEW_CROSSREF_RETRIES + 1, "matches": [], "error": last_error}


# -----------------------------------------------------------------------------
# Specialist agents
# -----------------------------------------------------------------------------
def run_micro_review(chunks: list[dict[str, Any]], document_type: str, session_dir: Path) -> str:
    results: list[str] = []
    failures: list[dict[str, Any]] = []
    result_path = session_dir / "micro_review.md"
    failure_path = session_dir / "micro_review_failures.json"

    def persist_partial() -> None:
        joined = "\n\n".join(
            f"===== MICRO REVIEW {i+1} =====\n{x}" for i, x in enumerate(results)
        )
        save_text(result_path, joined)
        save_json(failure_path, {
            "stage": "micro_review",
            "total_chunks": len(chunks),
            "completed_chunks": len(results) - len(failures),
            "failed_chunks": len(failures),
            "failures": failures,
            "status": "partial" if failures else "completed",
        })

    for idx, chunk in enumerate(chunks, start=1):
        prompt = f"""
You are the LINE-AND-PARAGRAPH REVIEWER in a rigorous academic document review.
Review the supplied source text minutely. Treat every line and paragraph as reviewable.
Do not rewrite the whole document. Identify specific, actionable problems and preserve
page/line references exactly.

Check:
- grammar, spelling, punctuation, syntax
- ambiguous or vague wording
- unsupported claims
- logical gaps and contradictions
- undefined terms/acronyms
- repetition and unnecessary wording
- readability and academic style
- technical imprecision visible from the text
- numerical inconsistencies visible in the text
- bad transitions or missing explanation
- citation placement problems
- equations that appear malformed or unexplained
- claims that need evidence

For each issue use:
[SEVERITY: Critical/Major/Moderate/Minor]
LOCATION: p.X Lxxx-Lyyy or paragraph description
ISSUE: ...
WHY: ...
SUGGESTED CHANGE: ...

End with a concise list of the strongest positives.

DOCUMENT TYPE: {document_type}
CHUNK {idx}/{len(chunks)}
SOURCE:
{chunk['text']}
"""
        try:
            out = ollama_chat(
                REVIEW_GENERAL_MODEL, prompt, ctx=REVIEW_GENERAL_CTX,
                tokens=REVIEW_GENERAL_TOKENS,
                label=f"micro review {idx}/{len(chunks)}",
                think=REVIEW_MICRO_THINK,
            )
            results.append(out)
        except Exception as exc:
            failure = {
                "chunk": idx,
                "chunk_id": chunk.get("id", idx),
                "pages": chunk.get("pages", []),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "recovery_exhausted": True,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            failures.append(failure)
            results.append(
                "[MICRO REVIEW UNAVAILABLE]\n"
                f"Chunk: {idx}/{len(chunks)}\n"
                f"Pages: {chunk.get('pages', [])}\n"
                f"Reason: {type(exc).__name__}: {exc}\n"
                "All configured per-request recovery paths were exhausted. "
                "This chunk was not successfully reviewed and must not be treated as a clean pass."
            )
            log(
                "REVIEW",
                f"Micro review {idx}/{len(chunks)} failed after all recovery paths; continuing. "
                f"Failure {len(failures)}/{REVIEW_MAX_MICRO_FAILURES_BEFORE_ABORT if REVIEW_ABORT_ON_MICRO_FAILURES else 'unbounded'} | {type(exc).__name__}: {exc}",
                "yellow",
            )
            persist_partial()

            if REVIEW_ABORT_ON_MICRO_FAILURES and len(failures) > REVIEW_MAX_MICRO_FAILURES_BEFORE_ABORT:
                log(
                    "ERROR",
                    f"Micro-review failure ceiling exceeded: {len(failures)} failures > "
                    f"allowed {REVIEW_MAX_MICRO_FAILURES_BEFORE_ABORT}. Aborting review for coverage safety.",
                    "red",
                )
                raise RuntimeError(
                    f"Micro-review failure ceiling exceeded: {len(failures)} failures; "
                    f"allowed {REVIEW_MAX_MICRO_FAILURES_BEFORE_ABORT}. See {failure_path.name}."
                ) from exc
            continue

        persist_partial()

    status = "completed_with_failures" if failures else "completed"
    save_json(failure_path, {
        "stage": "micro_review",
        "total_chunks": len(chunks),
        "completed_chunks": len(chunks) - len(failures),
        "failed_chunks": len(failures),
        "failures": failures,
        "status": status,
    })
    if failures:
        log(
            "REVIEW",
            f"Micro review stage completed with {len(failures)} unavailable chunk(s); downstream stages will continue.",
            "yellow",
        )
    return "\n\n".join(f"===== MICRO REVIEW {i+1} =====\n{x}" for i, x in enumerate(results))


def run_section_quality_review(document_type: str, outline: str, chunks: list[dict[str, Any]], micro_review: str, session_dir: Path) -> str:
    # Feed representative document text plus the micro review, not an unbounded corpus.
    body = "\n\n".join(c["text"] for c in chunks)
    body = body[:REVIEW_SECTION_MAX_CHARS]
    prompt = f"""
You are the SENIOR SECTION REVIEWER for a {document_type}.

Review the document at the section level using the outline and source excerpts below.
Focus on whether each section fulfills its scientific purpose, whether sections connect
logically, and whether important material is missing.

Return:
1. Section-by-section findings.
2. Missing or weak subsections.
3. Claims that should be strengthened or qualified.
4. Places where methods/results/discussion are mixed incorrectly.
5. Places where the contribution is unclear.
6. Specific proposed improvements, including replacement paragraphs when useful.

Do not invent facts. Clearly separate what is supported by the supplied document from
what would require outside evidence.

OUTLINE:
{outline}

SOURCE EXCERPTS:
{body}

MICRO-REVIEW SIGNALS:
{micro_review[:42000]}
"""
    out = ollama_chat(REVIEW_GENERAL_MODEL, prompt, ctx=REVIEW_GENERAL_CTX, tokens=REVIEW_GENERAL_TOKENS, label="section quality review", think=REVIEW_SECTION_THINK)
    save_text(session_dir / "section_quality.md", out)
    return out


def run_technical_review(document_type: str, chunks: list[dict[str, Any]], session_dir: Path) -> str:
    body = "\n\n".join(c["text"] for c in chunks)
    body = body[:50000]
    prompt = f"""
You are the TECHNICAL ACCURACY REVIEWER for a {document_type}.

Audit the supplied document excerpts as a skeptical subject-matter reviewer.
Check internal technical correctness, assumptions, terminology, method descriptions,
causal claims, algorithm descriptions, control/robotics reasoning, experimental claims,
and interpretation of results.

For every important finding provide:
- LOCATION
- CLAIM/STATEMENT
- ASSESSMENT: supported / questionable / incorrect / cannot verify from document
- TECHNICAL REASONING
- WHAT SHOULD BE CHECKED OR CHANGED

Pay special attention to:
- confusing correlation with causation
- comparing methods with mismatched conditions
- claims stronger than the experiment supports
- invalid physical/control assumptions
- inconsistent coordinate frames, units, signs or conventions
- unstable or unjustified mathematical/control statements
- conclusions not supported by reported data

Do not silently correct the paper; explain why the issue matters.

SOURCE:
{body}
"""
    out = ollama_chat(REVIEW_CRITICAL_MODEL, prompt, ctx=REVIEW_CRITICAL_CTX, tokens=REVIEW_CRITICAL_TOKENS, label="technical accuracy review")
    save_text(session_dir / "technical_review.md", out)
    return out


def run_visual_review(pages: list[PageRecord], session_dir: Path) -> str:
    if not REVIEW_VISUAL_REVIEW:
        return "Visual review disabled by configuration."
    render_dir = session_dir / "page_renders"
    images = sorted(render_dir.glob("page_*.png"))
    if not images:
        return "No rendered pages available for visual review."
    results: list[str] = []
    for image_path in images:
        page_no = int(re.search(r"page_(\d+)", image_path.stem).group(1))
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except Exception as exc:
            results.append(f"p.{page_no}: unable to read render: {exc}")
            continue
        prompt = f"""
You are the FIGURE/PLOT/EQUATION VISUAL REVIEWER.

Inspect page {page_no} of an academic/technical document as an image.
Only make observations that are actually visible.

Check:
- figures, plots, diagrams and tables
- axis labels, legends, units and scales
- captions and whether they match the visual content
- unreadable labels, overlapping elements, clutter, misleading scales
- whether visual conclusions are supported by what is plotted
- curves, markers, colors, legends and annotations
- numerical labels and consistency with surrounding text when visible
- equations and mathematical notation on the page
- malformed symbols, missing terms, inconsistent notation, dimensional issues
- whether the figure/table communicates the intended scientific result

Report:
LOCATION: p.{page_no}
ELEMENT: figure/table/equation/diagram/text
OBSERVED: ...
POTENTIAL PROBLEM: ...
WHY IT MATTERS: ...
SUGGESTED IMPROVEMENT: ...

Do not invent values that are not legible.
"""
        out = ollama_chat(
            REVIEW_VISION_MODEL,
            prompt,
            images=[encoded],
            ctx=REVIEW_VISION_CTX,
            tokens=REVIEW_VISION_TOKENS,
            label=f"visual review p.{page_no}",
            timeout=REVIEW_VISION_TIMEOUT,
            keep_alive=0,
        )
        results.append(out)
    joined = "\n\n".join(f"===== VISUAL PAGE {i+1} =====\n{x}" for i, x in enumerate(results))
    save_text(session_dir / "visual_review.md", joined)
    return joined


def run_equation_review(pages: list[PageRecord], visual_review: str, session_dir: Path) -> str:
    eq_pages = [p for p in pages if p.has_math_signals]
    text = "\n\n".join(f"PAGE {p.page}\n{p.text}" for p in eq_pages)
    prompt = f"""
You are the MATHEMATICAL EQUATION REVIEWER.

Audit the equations and mathematical statements in the document. Check them carefully
rather than merely checking formatting.

For each identifiable equation or derivation, assess:
- algebraic consistency where the derivation is visible
- variable definitions
- dimensions/units where inferable
- signs and coordinate conventions
- assumptions and boundary conditions
- consistency between equations and surrounding prose
- consistency of symbols across sections
- missing terms, duplicated terms or undefined symbols
- whether the stated result actually follows from the presented derivation

If the PDF contains an equation image that the text extractor mangled, use the visual
review notes as evidence but do not invent missing characters.

Return precise location references and recommended corrections.

TEXT FROM MATH-SIGNAL PAGES:
{text[:50000]}

RELEVANT VISUAL NOTES:
{visual_review[:35000]}
"""
    out = ollama_chat(REVIEW_CRITICAL_MODEL, prompt, ctx=REVIEW_CRITICAL_CTX, tokens=REVIEW_CRITICAL_TOKENS, label="equation and mathematics review")
    save_text(session_dir / "equation_review.md", out)
    return out


def run_data_review(pages: list[PageRecord], session_dir: Path) -> str:
    numeric_pages = [p for p in pages if len(re.findall(r"\d+(?:\.\d+)?", p.text)) >= 12]
    body = "\n\n".join(f"PAGE {p.page}\n{p.text}" for p in numeric_pages)
    body = body[:REVIEW_DATA_INPUT_MAX_CHARS]
    prompt = f"""
You are the DATA AND NUMERICAL CONSISTENCY REVIEWER.

Audit tables, numerical results, percentages, units, sample sizes, ranges, means,
errors/uncertainties and numerical statements visible in the supplied pages.

Look for:
- numbers that disagree between abstract/results/discussion
- totals that do not add up when arithmetic is possible from the visible values
- percentages inconsistent with stated counts
- impossible units or scale changes
- inconsistent significant figures
- claims of improvement that do not match the numbers
- tables whose captions/labels disagree with contents
- suspiciously exact values without uncertainty where uncertainty should matter

Do not invent missing data. State exactly what arithmetic or source check is possible.

SOURCE:
{body}
"""
    out = ollama_chat(REVIEW_GENERAL_MODEL, prompt, ctx=REVIEW_GENERAL_CTX, tokens=REVIEW_GENERAL_TOKENS, label="data and numerical consistency review", think=REVIEW_DATA_THINK)
    save_text(session_dir / "data_review.md", out)
    return out


def run_style_review(chunks: list[dict[str, Any]], document_type: str, session_dir: Path) -> str:
    body = "\n\n".join(c["text"] for c in chunks)
    body = body[:50000]
    prompt = f"""
You are the LANGUAGE, READABILITY AND ACADEMIC STYLE REVIEWER.

Review this {document_type} for:
- grammar, spelling and punctuation
- sentence structure
- readability
- excessive passive voice when active voice is clearer
- vague pronouns and ambiguous references
- unnecessary jargon
- repetitive phrases
- poor paragraph structure
- abrupt transitions
- weak topic sentences
- inconsistent terminology
- inappropriate academic tone

Then separately identify textual signals that may make prose feel generic, formulaic,
unnaturally polished, repetitive or machine-like. Do NOT claim to determine authorship
or whether AI was used. Instead, recommend edits that make the prose more specific,
precise, evidence-linked and naturally aligned with an academic author's voice.

Provide before/after rewrites for representative problematic paragraphs.

SOURCE:
{body}
"""
    out = ollama_chat(REVIEW_GENERAL_MODEL, prompt, ctx=REVIEW_GENERAL_CTX, tokens=REVIEW_GENERAL_TOKENS, label="language and natural academic voice review", think=REVIEW_LANGUAGE_THINK)
    save_text(session_dir / "style_review.md", out)
    return out


def run_language_review(chunks: list[dict[str, Any]], document_type: str, session_dir: Path) -> str:
    """Dedicated language/voice review, including cautious machine-like style signals.

    This is not an AI detector and does not claim authorship attribution. It identifies
    textual patterns that may feel formulaic, generic, repetitive or over-polished and
    provides evidence-preserving rewrites.
    """
    if not REVIEW_LANGUAGE_ENABLED:
        out = "Dedicated language review disabled by configuration."
        save_text(session_dir / "language_review.md", out)
        return out

    body = "\n\n".join(c["text"] for c in chunks)[:65000]
    prompt = f"""
You are the DEDICATED LANGUAGE AND ACADEMIC VOICE REVIEWER for a {document_type}.

Read the supplied manuscript excerpts carefully and review language quality as a specialist
academic editor. This is a separate pass from grammar spotting in the micro-review.

Check in detail:
- grammar, spelling, punctuation, syntax and article/preposition usage
- sentence fragments, run-ons and awkward constructions
- subject-verb agreement and tense consistency
- pronoun/reference ambiguity
- paragraph coherence and topic-sentence quality
- transitions and logical flow between sentences
- excessive passive voice where active voice is clearer
- unnecessary nominalization and bloated phrasing
- repeated sentence templates and repetitive transition phrases
- vague claims and imprecise academic language
- inconsistent terminology, acronym usage and capitalization
- informal, promotional, exaggerated or unnecessarily hedged wording
- readability while preserving technical precision

Then perform a CAUTIOUS MACHINE-LIKE PROSE SIGNAL REVIEW. Do NOT claim to determine whether
AI was used and do NOT provide a probability of AI authorship. Instead identify concrete textual
signals that may make a passage feel generic, formulaic or machine-like, such as:
- generic introductory filler
- repetitive "Moreover/Furthermore/Additionally" style transitions
- uniform sentence lengths and patterns
- repeated conclusion formulas
- vague claims such as "plays a crucial role" without specificity
- excessive restatement of the same point
- polished but low-information wording
- templated subsection openings/closings
- abstract language where a concrete technical statement would be stronger

For representative issues provide:
LOCATION: p.X Lxxx-Lyyy or paragraph identifier
CATEGORY: Grammar / Readability / Academic Style / Terminology / Machine-like signal
CURRENT: ...
ISSUE: ...
SUGGESTED REVISION: ...
WHY: ...

Also provide:
1. TOP 20 LANGUAGE PROBLEMS ranked by importance.
2. A section-level language pattern summary.
3. A "humanize for authorship clarity" set of up to {REVIEW_HUMANIZE_EXAMPLES} rewrites.
   Preserve technical meaning, numbers, citations and claims supported by the source.
4. A list of passages that should NOT be rewritten because they are already precise.
5. A final editing checklist for the author.

IMPORTANT: The goal of the rewrite is stronger, more specific, genuinely authorial academic
prose—not evasion of AI detection systems.

SOURCE EXCERPTS:
{body}
"""
    out = ollama_chat(
        REVIEW_LANGUAGE_MODEL,
        prompt,
        ctx=REVIEW_LANGUAGE_CTX,
        tokens=REVIEW_LANGUAGE_TOKENS,
        label="dedicated language and academic voice review",
        timeout=REVIEW_LANGUAGE_TIMEOUT,
    )
    save_text(session_dir / "language_review.md", out)
    return out


def run_ai_style_signal_review(chunks: list[dict[str, Any]], document_type: str, session_dir: Path) -> str:
    """Audit machine-like prose signals without claiming AI authorship detection."""
    if not REVIEW_AI_STYLE_ENABLED:
        out = "Dedicated AI-style signal analysis disabled by configuration."
        save_text(session_dir / "ai_style_signal_review.md", out)
        return out

    body = "\n\n".join(c["text"] for c in chunks)[:65000]
    prompt = f"""
You are the AI-STYLE SIGNAL ANALYST for a {document_type}.

Important limitation: you CANNOT reliably determine whether a document was written by AI,
and you must not assign a probability of AI authorship or claim detector certainty.
Instead, identify textual characteristics that can make academic prose look generic,
templated, formulaic or overly machine-like to a human reader or to imperfect automated
style detectors.

Audit for concrete patterns such as:
- repetitive sentence templates
- repeated transition words or rhetorical formulas
- generic opening/closing sentences
- vague high-level claims with little document-specific content
- excessive use of "important", "crucial", "significant", "robust", etc. without evidence
- unnatural uniform sentence rhythm
- excessive parallel constructions
- repeated restatement of conclusions
- generic literature-summary language
- inflated or promotional wording
- unnecessary hedging or boilerplate caveats
- paragraphs that sound polished but contain low information density
- repeated phrases or semantic near-duplicates
- section openings that could be copied into almost any paper

For every high-value pattern give:
LOCATION: ...
SIGNAL: ...
EXAMPLE: ...
WHY IT FEELS GENERIC/FORMULAIC: ...
BETTER AUTHORIAL ALTERNATIVE: ...

Then provide:
1. TOP 15 machine-like prose signals.
2. Repeated phrase/formula inventory.
3. Sections with the strongest generic-language concentration.
4. A practical humanization plan that improves specificity, technical ownership and
   evidence linkage rather than attempting to evade detection.
5. A list of passages that should remain unchanged because they are already natural and precise.

SOURCE:
{body}
"""
    out = ollama_chat(
        REVIEW_AI_STYLE_MODEL, prompt, ctx=REVIEW_AI_STYLE_CTX,
        tokens=REVIEW_AI_STYLE_TOKENS, label="AI-style signal analysis",
        timeout=REVIEW_AI_STYLE_TIMEOUT,
        think=REVIEW_AI_STYLE_THINK,
    )
    save_text(session_dir / "ai_style_signal_review.md", out)
    return out


def run_citation_review(document_type: str, reference_entries: list[str], session_dir: Path) -> str:
    if not reference_entries:
        out = "No automatically detectable bibliography/reference block was found."
        save_text(session_dir / "citation_review.md", out)
        return out
    evidence = []
    for i, ref in enumerate(reference_entries[:REVIEW_REFERENCE_MAX_WEB], start=1):
        result = crossref_lookup(ref)
        evidence.append({"id": i, **result})
        if i % 10 == 0:
            log("CITATIONS", f"verified {i}/{len(reference_entries)} reference entries", "blue")
    save_json(session_dir / "citation_matches.json", evidence)

    compact = json.dumps(evidence, ensure_ascii=False, indent=2)
    if len(reference_entries) > REVIEW_REFERENCE_MAX_WEB:
        evidence.append({"note": f"Only first {REVIEW_REFERENCE_MAX_WEB} references were web-verified automatically; remaining references require targeted verification or a second pass."})
    prompt = f"""
You are the CITATION AND BIBLIOGRAPHY AUDITOR.

The document type is: {document_type}.
Below are reference entries from the document and candidate metadata from Crossref.

Audit:
- likely citation/reference mismatches
- wrong title/author/year/venue/DOI
- malformed or incomplete bibliography entries
- duplicate references
- citations that appear not to support the surrounding claim when the source metadata allows that assessment
- references that need manual verification
- missing DOI or metadata when important

Then explain what literature may be missing at a thematic level. Do not declare the
literature complete. Identify likely gaps and propose specific topic/query searches.

REFERENCE EVIDENCE:
{compact[:90000]}
"""
    out = ollama_chat(REVIEW_GENERAL_MODEL, prompt, ctx=REVIEW_GENERAL_CTX, tokens=REVIEW_GENERAL_TOKENS, label="citation and bibliography review", think=REVIEW_CITATION_THINK)
    save_text(session_dir / "citation_review.md", out)
    return out


def run_literature_gap_review(document_type: str, pages: list[PageRecord], session_dir: Path) -> str:
    body = "\n".join(p.text for p in pages[:max(1, len(pages))])[:50000]
    prompt = f"""
You are the LITERATURE-GAP AND CLAIM-VERIFICATION RESEARCHER.

Based on the document below, identify the major research topics, methods, claimed
contributions, comparison sets and application domains. Generate at most {REVIEW_LITERATURE_QUERIES}
precise web-search queries aimed at discovering important omitted literature or checking
whether strong claims are well-supported.

Return:
QUERY 1: ...
QUERY 2: ...
...

Do not answer the research question yet.

DOCUMENT:
{body}
"""
    query_text = ollama_chat(REVIEW_GENERAL_MODEL, prompt, ctx=REVIEW_GENERAL_CTX, tokens=2200, label="literature gap query planning", think=REVIEW_LITERATURE_PLAN_THINK)
    queries = re.findall(r"QUERY\s+\d+\s*:\s*(.+)", query_text, flags=re.I)
    if not queries:
        queries = [q.strip() for q in query_text.splitlines() if q.strip()][:REVIEW_LITERATURE_QUERIES]
    web_evidence = []
    for q in queries[:REVIEW_LITERATURE_QUERIES]:
        primary = web_search(q)
        record = {"query": q, "web": primary}
        if primary.get("status") in {"failed", "unavailable"} and REVIEW_WEB_CROSSREF_FALLBACK:
            log("WEB", f"Using Crossref academic fallback for failed search: {q!r}", "yellow")
            record["crossref_fallback"] = crossref_search(q)
        web_evidence.append(record)
    save_json(session_dir / "literature_web_search.json", web_evidence)
    prompt2 = f"""
You are the FINAL LITERATURE-RELEVANCE ANALYST.
Assess whether the document's literature coverage appears sufficient for its stated
scope. Use the web search results as discovery evidence, not as proof by themselves.

Discuss:
- well-covered areas
- likely under-covered areas
- important methodological families absent from the review
- missing foundational or highly relevant recent work
- claims whose citations should be strengthened
- suggested additional search directions

DOCUMENT TYPE: {document_type}
WEB DISCOVERY:
{json.dumps(web_evidence, ensure_ascii=False)[:60000]}
"""
    out = ollama_chat(REVIEW_GENERAL_MODEL, prompt2, ctx=REVIEW_GENERAL_CTX, tokens=REVIEW_GENERAL_TOKENS, label="literature completeness review", think=REVIEW_LITERATURE_ANALYSIS_THINK)
    save_text(session_dir / "literature_review.md", out)
    return out


COORDINATOR_AGENT_MAP = {
    "technical": "technical_review.md",
    "equations": "equation_review.md",
    "visual": "visual_review.md",
    "data": "data_review.md",
    "statistics": "statistical_review.md",
    "language": "language_review.md",
    "ai_style": "ai_style_signal_review.md",
    "citations": "citation_review.md",
    "literature": "literature_review.md",
    "novelty": "novelty_contribution_review.md",
    "reproducibility": "reproducibility_review.md",
    "experiments": "experimental_design_review.md",
    "venue": "venue_suitability_review.md",
    "front_matter": "front_matter_review.md",
    "nomenclature": "nomenclature_units_review.md",
    "section": "section_quality.md",
    "micro": "micro_review.md",
    "hostile": "hostile_reviewer2.md",
}


def _read_agent_reports(session_dir: Path, limit: int = 240000) -> str:
    reports = []
    for key, filename in COORDINATOR_AGENT_MAP.items():
        path = session_dir / filename
        if path.exists():
            reports.append(f"\n===== {key.upper()} ({filename}) =====\n{path.read_text(encoding='utf-8')[:30000]}")
    return "".join(reports)[:limit]


def _parse_coordinator_actions(text: str) -> list[dict[str, str]]:
    """Best-effort parser for coordinator action plans; avoids hard failure on imperfect JSON."""
    actions: list[dict[str, str]] = []
    # Prefer JSON if the model supplied a valid JSON array/object.
    candidates = []
    m = re.search(r"\{\s*\"actions\"\s*:\s*\[.*\]\s*\}", text, flags=re.S)
    if m:
        candidates.append(m.group(0))
    m2 = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.S)
    if m2:
        candidates.append(m2.group(0))
    for raw in candidates:
        try:
            obj = json.loads(raw)
            vals = obj.get("actions", []) if isinstance(obj, dict) else obj
            if isinstance(vals, list):
                for item in vals:
                    if isinstance(item, dict):
                        agent = str(item.get("agent", "")).strip().lower()
                        task = str(item.get("task", "")).strip()
                        reason = str(item.get("reason", "")).strip()
                        if agent in COORDINATOR_AGENT_MAP and task:
                            actions.append({"agent": agent, "task": task, "reason": reason})
                if actions:
                    return actions
        except Exception:
            pass

    # Fallback: parse lines like ACTION: technical | TASK: ... | REASON: ...
    for line in text.splitlines():
        m = re.match(r"\s*(?:ACTION|REVIEW)\s*[:\-]\s*([^|]+)\|\s*(?:TASK|QUESTION)\s*[:\-]\s*(.+?)(?:\|\s*(?:REASON)\s*[:\-]\s*(.*))?$", line, flags=re.I)
        if not m:
            continue
        agent = m.group(1).strip().lower()
        task = m.group(2).strip()
        reason = (m.group(3) or "").strip()
        if agent in COORDINATOR_AGENT_MAP and task:
            actions.append({"agent": agent, "task": task, "reason": reason})
    return actions


def run_targeted_coordinator_review(agent: str, task: str, session: ReviewSession, session_dir: Path, pages: list[PageRecord], chunks: list[dict[str, Any]]) -> str:
    """Run a focused second-pass review requested by the coordinator."""
    text = "\n\n".join(c["text"] for c in chunks)[:65000]
    specialist_prompts = {
        "technical": "Re-audit technical correctness and assumptions for this targeted question.",
        "equations": "Re-audit equations, derivations, signs, dimensions, symbols and mathematical logic for this targeted question.",
        "visual": "Re-audit figures, plots, tables and visual evidence relevant to this targeted question.",
        "data": "Re-audit numerical consistency, arithmetic, units and reported metrics relevant to this targeted question.",
        "statistics": "Re-audit statistical methodology, uncertainty, sample design, tests and quantitative claims relevant to this targeted question.",
        "language": "Re-audit grammar, readability and academic voice for this targeted question.",
        "ai_style": "Re-audit concrete machine-like prose signals and humanization opportunities for this targeted question. Do not claim AI authorship detection.",
        "citations": "Re-audit citations/bibliography and claim-source alignment for this targeted question.",
        "literature": "Re-audit literature coverage and missing relevant work for this targeted question.",
        "novelty": "Re-audit novelty and contribution claims for this targeted question.",
        "reproducibility": "Re-audit reproducibility and missing implementation details for this targeted question.",
        "experiments": "Re-audit experimental design and validity for this targeted question.",
        "venue": "Re-audit venue/scope suitability implications for this targeted question.",
        "front_matter": "Re-audit title/abstract/keywords against this targeted question.",
        "nomenclature": "Re-audit nomenclature, symbols, units and conventions for this targeted question.",
        "section": "Re-audit section structure and argument flow for this targeted question.",
        "micro": "Re-audit the manuscript at paragraph/line level for this targeted question.",
        "hostile": "Attack the manuscript again as Reviewer #2, focusing only on this targeted question.",
    }
    instruction = specialist_prompts.get(agent, "Perform a focused academic review for this targeted question.")
    prompt = f"""
You are a specialist reviewer in a coordinated second-pass academic review.

DOCUMENT TYPE: {session.document_type}
SPECIALIST: {agent}
COORDINATOR TASK: {task}
RATIONALE: targeted follow-up requested because an earlier review indicated a potential weakness.

{instruction}

Return only findings relevant to the coordinator task. Give exact page/line locations when
available, distinguish verified issues from uncertainties, and propose concrete corrections.
Do not invent evidence.

SOURCE EXCERPTS:
{text}
"""
    critical_agents = {"technical", "equations", "statistics", "experiments", "novelty"}
    if agent == "hostile":
        model, ctx, tokens, timeout = REVIEW_HOSTILE_MODEL, REVIEW_HOSTILE_CTX, REVIEW_HOSTILE_TOKENS, REVIEW_HOSTILE_TIMEOUT
    elif agent in critical_agents:
        model, ctx, tokens, timeout = REVIEW_CRITICAL_MODEL, REVIEW_CRITICAL_CTX, REVIEW_CRITICAL_TOKENS, REVIEW_LLM_TIMEOUT
    elif agent == "language":
        model, ctx, tokens, timeout = REVIEW_LANGUAGE_MODEL, REVIEW_LANGUAGE_CTX, REVIEW_LANGUAGE_TOKENS, REVIEW_LANGUAGE_TIMEOUT
    elif agent == "ai_style":
        model, ctx, tokens, timeout = REVIEW_AI_STYLE_MODEL, REVIEW_AI_STYLE_CTX, REVIEW_AI_STYLE_TOKENS, REVIEW_AI_STYLE_TIMEOUT
    else:
        model, ctx, tokens, timeout = REVIEW_GENERAL_MODEL, REVIEW_GENERAL_CTX, REVIEW_GENERAL_TOKENS, REVIEW_LLM_TIMEOUT
    label = f"coordinator targeted {agent}"
    targeted_think = {
        "technical": REVIEW_CRITICAL_THINK,
        "equations": REVIEW_CRITICAL_THINK,
        "statistics": REVIEW_STATISTICS_THINK,
        "experiments": REVIEW_EXPERIMENTAL_DESIGN_THINK,
        "novelty": REVIEW_NOVELTY_THINK,
        "hostile": REVIEW_HOSTILE_THINK,
        "reproducibility": REVIEW_REPRODUCIBILITY_THINK,
        "literature": REVIEW_LITERATURE_ANALYSIS_THINK,
        "section": REVIEW_SECTION_THINK,
        "micro": REVIEW_MICRO_THINK,
        "language": REVIEW_LANGUAGE_THINK,
        "ai_style": REVIEW_AI_STYLE_THINK,
        "citations": REVIEW_CITATION_THINK,
        "data": REVIEW_DATA_THINK,
        "visual": REVIEW_VISION_THINK,
        "venue": False,
        "front_matter": False,
        "nomenclature": False,
    }.get(agent, False)
    out = ollama_chat(model, prompt, ctx=ctx, tokens=tokens, label=label, timeout=timeout, think=targeted_think)
    path = session_dir / f"coordinator_{agent}_{int(time.time())}.md"
    save_text(path, out)
    return out


def run_review_coordinator(session: ReviewSession, session_dir: Path, pages: list[PageRecord], chunks: list[dict[str, Any]], round_no: int) -> list[dict[str, str]]:
    """Coordinator decides which specialist agents need another targeted pass."""
    if not REVIEW_COORDINATOR_ENABLED:
        return []
    reports = _read_agent_reports(session_dir)
    prompt = f"""
You are the REVIEW COORDINATOR for a deep academic document-review system.

Your task is NOT to write the final review. Your job is to inspect the specialist reports,
detect unresolved/high-risk contradictions or blind spots, and decide which specialist needs
a focused second pass. This makes the review adaptive rather than a fixed checklist.

Review round: {round_no}/{REVIEW_COORDINATOR_MAX_ROUNDS}
Document type: {session.document_type}
Pages: {session.pages}

Available specialist agents:
technical, equations, visual, data, statistics, language, ai_style, citations, literature, novelty,
reproducibility, experiments, venue, front_matter, nomenclature, section, micro, hostile

Select at most {REVIEW_COORDINATOR_MAX_ACTIONS_PER_ROUND} targeted actions. Prefer actions that:
- resolve disagreement between agents
- investigate a potentially fatal correctness issue
- check a high-impact unsupported claim
- verify a suspicious numerical/equation/figure inconsistency
- resolve a literature or citation uncertainty
- investigate an important novelty or experiment weakness
- clarify a language problem that changes meaning

Do not request generic re-review. Each action must have a precise question. If no targeted
re-review is needed, return an empty action list.

Return ONLY JSON in this form:
{{
  "actions": [
    {{"agent": "technical", "task": "exact focused question", "reason": "why another pass is needed"}}
  ]
}}

REVIEW COVERAGE / UNAVAILABLE PASSES:
{((session_dir / "review_coverage.json").read_text(encoding="utf-8") if (session_dir / "review_coverage.json").exists() else "No explicit coverage exceptions recorded.")}

SPECIALIST REPORTS:
{reports}
"""
    out = ollama_chat(
        REVIEW_COORDINATOR_MODEL, prompt, ctx=REVIEW_COORDINATOR_CTX,
        tokens=REVIEW_COORDINATOR_TOKENS, label=f"review coordinator round {round_no}",
        timeout=REVIEW_COORDINATOR_TIMEOUT,
    )
    save_text(session_dir / f"coordinator_round_{round_no}.md", out)
    actions = _parse_coordinator_actions(out)[:REVIEW_COORDINATOR_MAX_ACTIONS_PER_ROUND]
    save_json(session_dir / f"coordinator_round_{round_no}_actions.json", actions)
    return actions


def run_final_arbiter(session_dir: Path, session: ReviewSession) -> str:
    files = [
        "micro_review.md", "section_quality.md", "technical_review.md", "visual_review.md",
        "equation_review.md", "data_review.md", "statistical_review.md", "style_review.md",
        "citation_review.md", "literature_review.md", "novelty_contribution_review.md",
        "reproducibility_review.md", "experimental_design_review.md", "venue_suitability_review.md",
        "front_matter_review.md", "nomenclature_units_review.md", "hostile_reviewer2.md",
        "coordinator_round_1.md", "coordinator_round_2.md",
    ]
    evidence = []
    coverage_path = session_dir / "review_coverage.json"
    if coverage_path.exists():
        evidence.append(f"\n===== REVIEW COVERAGE =====\n{coverage_path.read_text(encoding='utf-8')[:12000]}")
    for name in files:
        p = session_dir / name
        if p.exists():
            evidence.append(f"\n===== {name} =====\n{p.read_text(encoding='utf-8')[:50000]}")
    targeted_files = sorted(session_dir.glob("coordinator_*.md"))
    for target_path in targeted_files:
        if target_path.name not in files:
            evidence.append(f"\n===== {target_path.name} =====\n{target_path.read_text(encoding='utf-8')[:30000]}")
    prompt = f"""
You are the SENIOR ACADEMIC REVIEW ARBITER.

Produce the final integrated review of the document. Do not merely repeat agent findings.
Resolve duplicates, distinguish hard errors from suggestions, and prioritize actions.

Document:
{session.pdf_path}
Type: {session.document_type}
Pages: {session.pages}

IMPORTANT COVERAGE RULE:
Any specialist or micro-review marked unavailable in REVIEW COVERAGE was not successfully completed.
Do not infer a clean result for missing coverage. Explicitly disclose unavailable passes in the final review.

Your review must contain these sections:
1. EXECUTIVE ASSESSMENT
2. TOP 10 HIGH-PRIORITY ISSUES
3. SECTION-BY-SECTION REVIEW
4. TECHNICAL AND SCIENTIFIC ACCURACY
5. EQUATIONS AND MATHEMATICS
6. FIGURES, PLOTS AND TABLES
7. DATA / NUMERICAL CONSISTENCY
8. CITATIONS AND BIBLIOGRAPHY
9. LITERATURE COMPLETENESS / MISSING AREAS
10. GRAMMAR / READABILITY / ACADEMIC STYLE
11. DEDICATED LANGUAGE REVIEW
12. NATURAL ACADEMIC VOICE / MACHINE-LIKE PROSE SIGNALS
13. NOVELTY / CONTRIBUTION ASSESSMENT
14. STATISTICAL METHODOLOGY
15. REPRODUCIBILITY
16. EXPERIMENTAL DESIGN
17. JOURNAL / CONFERENCE SUITABILITY
18. TITLE / ABSTRACT / KEYWORDS
19. NOMENCLATURE / UNITS / SYMBOL CONSISTENCY
20. REVIEWER #2 / HOSTILE REVIEW
21. COORDINATOR-IDENTIFIED HIGH-RISK ISSUES
22. SPECIFIC REWRITES
23. PUBLICATION / SUBMISSION READINESS CHECKLIST

For each serious issue provide location, why it matters, and a concrete change.
Where a reviewer cannot verify a claim from the document or web evidence, say so.
Do not pretend that web search establishes mathematical or experimental truth.

IMPORTANT: You are reviewing the supplied material. Do not silently rewrite technical
facts to fit generic knowledge. Proposed rewrites must preserve the author's evidence
unless you explicitly say additional evidence is needed.

AGENT REPORTS:
{''.join(evidence)[:220000]}
"""
    out = ollama_chat(REVIEW_CRITICAL_MODEL, prompt, ctx=REVIEW_CRITICAL_CTX, tokens=REVIEW_CRITICAL_TOKENS, label="senior final review")
    save_text(session_dir / "initial_review.md", out)
    return out


def followup_answer(session_dir: Path, session: ReviewSession, question: str) -> str:
    review = (session_dir / "initial_review.md").read_text(encoding="utf-8") if (session_dir / "initial_review.md").exists() else ""
    prompt = f"""
You are the follow-up reviewer for an academic document.

The user is asking about a previous review. Answer the question directly and critically.
You may disagree with a previous suggestion if it is not justified by the source.
When proposing a rewrite, preserve technical meaning and evidence.

Document: {session.pdf_path}
Type: {session.document_type}

INITIAL REVIEW:
{review[:80000]}

USER FOLLOW-UP:
{question}

Provide:
- direct answer
- reasoning tied to the document/review
- exact modification or rewrite when requested
- any caveat or evidence still needed
"""
    out = ollama_chat(REVIEW_GENERAL_MODEL, prompt, ctx=REVIEW_GENERAL_CTX, tokens=REVIEW_GENERAL_TOKENS, label="review follow-up", think=REVIEW_FOLLOWUP_THINK)
    with (session_dir / "followups.md").open("a", encoding="utf-8") as f:
        f.write(f"\n\n===== {datetime.now().isoformat(timespec='seconds')} =====\nUSER: {question}\n\n{out}\n")
    return out


def humanize_text(session: ReviewSession, paragraph: str) -> str:
    prompt = f"""
You are an academic language editor helping an author make prose sound natural,
specific, precise, and consistent with an authentic researcher's voice.
Do not try to bypass AI detectors and do not claim the text will be undetectable.
Instead, remove generic or formulaic phrasing, preserve technical meaning and all
supported facts, and make the writing more direct and authorial.

Document type: {session.document_type}

Original paragraph:
{paragraph}

Return:
1. Revised paragraph
2. What changed
3. Any technical fact that should be checked before replacing the original
"""
    return ollama_chat(REVIEW_GENERAL_MODEL, prompt, ctx=REVIEW_GENERAL_CTX, tokens=3000, label="natural academic rewrite", think=REVIEW_REWRITE_THINK)


# -----------------------------------------------------------------------------
# Advanced specialist agents
# -----------------------------------------------------------------------------
def run_novelty_review(document_type: str, outline: str, micro_review: str, literature: str, session_dir: Path) -> str:
    prompt = f"""
You are the NOVELTY AND CONTRIBUTION ASSESSOR.

Assess whether the document establishes a clear, technically meaningful contribution.
Do not decide novelty from generic claims alone. Distinguish:
- explicitly demonstrated contributions
- incremental engineering contributions
- potentially novel elements requiring stronger evidence
- claims of novelty that are not adequately supported
- overlap with prior work suggested by the supplied literature evidence

For each contribution, give:
1. CLAIMED CONTRIBUTION
2. EVIDENCE CURRENTLY PRESENT
3. NOVELTY RISK / OVERLAP RISK
4. WHAT MUST BE ADDED OR REWRITTEN
5. A stronger contribution statement where justified

Do not invent prior art. Flag items that need targeted literature verification.

DOCUMENT TYPE: {document_type}
OUTLINE:
{outline[:30000]}
MICRO REVIEW:
{micro_review[:45000]}
LITERATURE REVIEW:
{literature[:45000]}
"""
    out=ollama_chat(REVIEW_CRITICAL_MODEL,prompt,ctx=REVIEW_CRITICAL_CTX,tokens=REVIEW_CRITICAL_TOKENS,label="novelty and contribution review", think=REVIEW_NOVELTY_THINK)
    save_text(session_dir/"novelty_contribution_review.md",out)
    return out


def run_statistical_review(pages: list[PageRecord], session_dir: Path) -> str:
    body="\n\n".join(f"PAGE {p.page}\n{p.text}" for p in pages if re.search(r"\b(?:mean|median|std|standard deviation|variance|p[- ]?value|confidence interval|anova|t[- ]test|significant|sample size|n\s*=|rmse|mae|rms|error bar|uncertainty|regression|correlation)\b",p.text,re.I))
    body=body[:70000]
    prompt=f"""
You are the STATISTICAL METHODOLOGY AND QUANTITATIVE REVIEWER.

Audit the statistical and quantitative methodology actually present in the document.
Check, where applicable:
- sample size and justification
- train/test/validation separation
- repeated trials and independence
- mean/median choice
- variance/standard deviation/error bars
- confidence intervals and uncertainty
- hypothesis tests, p-values and effect sizes
- multiple comparisons
- regression/correlation claims
- benchmark fairness and baseline selection
- averaging across runs/subjects/scenarios
- data leakage and selection bias
- whether reported precision is justified
- whether conclusions exceed the presented statistics
- statistical reporting completeness

For engineering/robotics papers with limited classical statistics, explicitly say so
rather than forcing irrelevant tests. Identify exact pages/locations where possible.

SOURCE EXCERPTS:\n{body}
"""
    out=ollama_chat(REVIEW_CRITICAL_MODEL,prompt,ctx=REVIEW_CRITICAL_CTX,tokens=REVIEW_CRITICAL_TOKENS,label="statistical methodology review", think=REVIEW_STATISTICS_THINK)
    save_text(session_dir/"statistical_review.md",out)
    return out


def run_reproducibility_review(document_type: str, pages: list[PageRecord], session_dir: Path) -> str:
    body="\n\n".join(f"PAGE {p.page}\n{p.text}" for p in pages)[:70000]
    prompt=f"""
You are the REPRODUCIBILITY AUDITOR.

Assess whether an independent researcher could reproduce the study from the document.
Check:
- hardware/platform specification
- software versions and implementation details
- algorithms and parameter values
- controller gains / hyperparameters
- datasets and data splits
- random seeds / repeated runs
- calibration and preprocessing
- sensor configuration
- experimental environment
- baseline implementations
- evaluation metrics
- enough detail to recreate figures/tables
- code/data availability
- ablation studies where appropriate
- failure cases and exclusions
- timing/computational-resource reporting

Separate missing details from details that are adequately specified. Give concrete additions.

DOCUMENT TYPE: {document_type}
DOCUMENT EXCERPTS:\n{body}
"""
    out=ollama_chat(REVIEW_GENERAL_MODEL,prompt,ctx=REVIEW_GENERAL_CTX,tokens=REVIEW_GENERAL_TOKENS,label="reproducibility audit", think=REVIEW_REPRODUCIBILITY_THINK)
    save_text(session_dir/"reproducibility_review.md",out)
    return out


def run_experimental_design_review(document_type: str, pages: list[PageRecord], session_dir: Path) -> str:
    body="\n\n".join(f"PAGE {p.page}\n{p.text}" for p in pages)[:80000]
    prompt=f"""
You are the EXPERIMENTAL DESIGN CRITIC.

Evaluate whether the experiments genuinely test the stated research questions and claims.
Check:
- research hypotheses/questions versus experiments
- baseline and ablation design
- control conditions
- parameter fairness
- number and diversity of scenarios
- disturbance/environment coverage
- robustness and failure-case testing
- statistical/repeated-trial design
- simulation-to-real validation
- sensitivity analysis
- whether metrics actually measure the claimed contribution
- whether comparisons are apples-to-apples
- possible confounds and hidden variables
- threats to validity

Give specific additional experiments that would materially strengthen the paper, not a generic wish list.

DOCUMENT TYPE: {document_type}
SOURCE:\n{body}
"""
    out=ollama_chat(REVIEW_CRITICAL_MODEL,prompt,ctx=REVIEW_CRITICAL_CTX,tokens=REVIEW_CRITICAL_TOKENS,label="experimental design review", think=REVIEW_EXPERIMENTAL_DESIGN_THINK)
    save_text(session_dir/"experimental_design_review.md",out)
    return out


def run_venue_review(document_type: str, outline: str, session_dir: Path) -> str:
    prompt=f"""
You are the JOURNAL/CONFERENCE SUITABILITY REVIEWER.

Assess the supplied document's apparent maturity and scope for publication venues.
Do not rank or pretend to know acceptance probability. Instead identify suitable venue
categories and what each typically expects based on the manuscript's subject, contribution,
validation depth and presentation.

Discuss separately:
- journal-oriented fit
- robotics/control conference-oriented fit
- interdisciplinary venue fit
- manuscript maturity
- missing components that would matter for stronger venues
- likely scope mismatch risks

Do not invent current author guidelines. State when venue-specific verification is needed.

Document type: {document_type}
Outline:\n{outline[:30000]}
"""
    out=ollama_chat(REVIEW_GENERAL_MODEL,prompt,ctx=REVIEW_GENERAL_CTX,tokens=REVIEW_GENERAL_TOKENS,label="journal and conference suitability review", think=REVIEW_SUITABILITY_THINK)
    save_text(session_dir/"venue_suitability_review.md",out)
    return out


def run_front_matter_review(pages: list[PageRecord], session_dir: Path) -> str:
    body="\n\n".join(f"PAGE {p.page}\n{p.text}" for p in pages[:8])[:50000]
    prompt=f"""
You are the TITLE, ABSTRACT AND KEYWORD OPTIMIZATION EDITOR.

Evaluate the front matter for scientific precision and discoverability.
Check:
- title accuracy, specificity and non-hype wording
- whether the title reflects the actual contribution
- abstract structure, completeness and quantitative specificity
- whether abstract claims are supported by the manuscript
- keyword relevance, uniqueness and indexing usefulness
- mismatch between title/abstract/keywords and the actual paper

Provide:
1. Problems found
2. A revised title
3. A revised abstract only where enough source evidence exists
4. A proposed keyword list
5. Reasons for each major change

SOURCE:\n{body}
"""
    out=ollama_chat(REVIEW_GENERAL_MODEL,prompt,ctx=REVIEW_GENERAL_CTX,tokens=REVIEW_GENERAL_TOKENS,label="title abstract keyword review", think=REVIEW_TITLE_ABSTRACT_THINK)
    save_text(session_dir/"front_matter_review.md",out)
    return out


def run_nomenclature_units_review(pages: list[PageRecord], session_dir: Path) -> str:
    body="\n\n".join(f"PAGE {p.page}\n{p.text}" for p in pages)[:80000]
    prompt=f"""
You are the NOMENCLATURE, SYMBOL AND UNIT CONSISTENCY AUDITOR.

Check the whole document for consistency of:
- variable names and symbols
- acronyms and abbreviations
- coordinate frames and sign conventions
- SI units and prefixes
- dimensional consistency where inferable
- capitalization and notation
- singular/plural use of technical terms
- parameter names across equations, tables and figures
- identical symbols used for different quantities
- quantities described with inconsistent units

Create a concise consistency table with location, current notation, problem and recommended notation.
Do not invent definitions that are not in the document.

SOURCE:\n{body}
"""
    out=ollama_chat(REVIEW_GENERAL_MODEL,prompt,ctx=REVIEW_GENERAL_CTX,tokens=REVIEW_GENERAL_TOKENS,label="nomenclature and units review", think=REVIEW_NOMENCLATURE_THINK)
    save_text(session_dir/"nomenclature_units_review.md",out)
    return out


def run_hostile_reviewer(session_dir: Path, session: ReviewSession) -> str:
    files=[
        "initial_review.md","technical_review.md","equation_review.md","visual_review.md",
        "data_review.md","statistical_review.md","experimental_design_review.md",
        "reproducibility_review.md","novelty_contribution_review.md","citation_review.md",
        "literature_review.md","venue_suitability_review.md","nomenclature_units_review.md"
    ]
    evidence=[]
    for name in files:
        p=session_dir/name
        if p.exists():
            evidence.append(f"\n===== {name} =====\n{p.read_text(encoding='utf-8')[:35000]}")
    prompt=f"""
You are REVIEWER #2: a skeptical, technically demanding and intentionally hostile peer reviewer.

Your job is not to be insulting. Your job is to look for weaknesses that a demanding reviewer
would attack before publication.

Attack the manuscript on:
- novelty and actual contribution
- unsupported or overstated claims
- methodology flaws
- weak baselines
- unfair comparisons
- missing experiments
- statistical weaknesses
- reproducibility gaps
- equation/technical inconsistencies
- literature omissions and outdated framing
- ambiguous figures/tables
- unclear writing that hides weaknesses
- conclusions that exceed evidence

For every serious criticism give:
1. REVIEWER #2 COMMENT
2. EVIDENCE / LOCATION
3. WHY THIS IS A VULNERABILITY
4. AUTHOR RESPONSE OR MANUSCRIPT FIX
5. MINIMUM CHANGE NEEDED

At the end give:
- the five attacks most likely to matter
- which criticisms are fatal versus repairable
- a proposed revision strategy before submission

Do not invent defects that are not supported by the document/review evidence.

DOCUMENT: {session.pdf_copy}
TYPE: {session.document_type}
AGENT EVIDENCE:
{''.join(evidence)[:220000]}
"""
    out=ollama_chat(REVIEW_HOSTILE_MODEL,prompt,ctx=REVIEW_HOSTILE_CTX,tokens=REVIEW_HOSTILE_TOKENS,label="reviewer #2 hostile pass",timeout=REVIEW_HOSTILE_TIMEOUT)
    save_text(session_dir/"hostile_reviewer2.md",out)
    return out


# -----------------------------------------------------------------------------
# Session management
# -----------------------------------------------------------------------------
def slug(text: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return x[:70] or "review"


def create_session(pdf_path: Path, doc_type: str) -> tuple[ReviewSession, Path]:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    title = pdf_path.stem
    session_dir = SESSIONS_DIR / f"{session_id}_{slug(title)}"
    session_dir.mkdir(parents=True, exist_ok=True)
    pdf_copy = session_dir / pdf_path.name
    shutil.copy2(pdf_path, pdf_copy)
    pages = len(fitz.open(pdf_copy))
    now = datetime.now().isoformat(timespec="seconds")
    session = ReviewSession(session_id, title, str(pdf_path), str(pdf_copy), doc_type, now, now, pages, "created", pdf_path.name, ACTIVE_MODE)
    save_json(session_dir / "session.json", asdict(session))
    return session, session_dir


def load_session(session_dir: Path) -> tuple[ReviewSession, Path]:
    data = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    return ReviewSession(**data), session_dir


def list_sessions() -> list[tuple[Path, ReviewSession]]:
    out = []
    for p in sorted(SESSIONS_DIR.glob("*"), reverse=True):
        if not p.is_dir() or not (p / "session.json").exists():
            continue
        try:
            session, _ = load_session(p)
            out.append((p, session))
        except Exception:
            continue
    return out


# -----------------------------------------------------------------------------
# Review runner
# -----------------------------------------------------------------------------
def _update_review_stage(session_dir: Path, stage: str, status: str, *, detail: str = "") -> None:
    """Persist per-stage review progress after each specialist pass."""
    path = session_dir / "review_progress.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        state = {}
    state[stage] = {
        "status": status,
        "updated": datetime.now().isoformat(timespec="seconds"),
        "detail": detail,
    }
    save_json(path, state)


def run_initial_review(session: ReviewSession, session_dir: Path) -> str:
    log("REVIEW", "Loading PDF and extracting page text...", "cyan")
    pdf = Path(session.pdf_copy)
    if not pdf.exists():
        raise FileNotFoundError(pdf)
    if pdf.stat().st_size > REVIEW_MAX_PDF_MB * 1024 * 1024:
        raise ValueError(f"PDF exceeds {REVIEW_MAX_PDF_MB} MB configured limit")

    render_dir = session_dir / "page_renders"
    pages = extract_pages(pdf, render_dir)
    all_text = "\n".join(p.text for p in pages)
    outline = document_outline(pages)
    references = split_reference_entries(extract_reference_block(pages))
    chunks = chunk_pages(pages)

    save_json(session_dir / "document_map.json", {
        "pages": len(pages),
        "chunks": [{"id": c["id"], "pages": c["pages"]} for c in chunks],
        "visual_pages": [p.page for p in pages if p.visual_needed],
        "reference_count_detected": len(references),
        "outline": outline,
    })
    save_text(session_dir / "extracted_text.txt", all_text)
    save_text(session_dir / "outline.txt", outline)

    log("REVIEW", "Building/loading reusable smart review context...", "cyan")
    build_context_pack(pages, outline, references, session_dir)
    if ACTIVE_MODE == "short":
        log("REVIEW", "SHORT mode: thinking OFF; reduced ctx/num_predict/prompt payload; no stages or pages skipped.", "magenta")
    else:
        log("REVIEW", "FULL mode: reusable smart context + proactive source compression enabled; raw PDF remains authoritative.", "magenta")

    log("REVIEW", f"Pages={len(pages)} | chunks={len(chunks)} | visual/math pages={sum(p.visual_needed for p in pages)} | references={len(references)}", "cyan")

    # Base specialist passes. They are intentionally explicit so every document gets
    # the same foundational checks before the coordinator decides on adaptive re-review.
    # Independent specialist failures are isolated: all configured per-request recovery
    # happens first inside ollama_chat; if that still fails, the specialist is recorded as
    # unavailable and the downstream pipeline continues until the configured stage-failure ceiling.
    specialist_failure_records: list[dict[str, Any]] = []

    def _record_specialist_failure(name: str, exc: Exception) -> str:
        safe_name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "specialist"
        record = {
            "stage": name,
            "status": "failed",
            "updated": datetime.now().isoformat(timespec="seconds"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        specialist_failure_records.append(record)
        save_json(session_dir / "specialist_failures.json", {
            "mode": ACTIVE_MODE,
            "failed_count": len(specialist_failure_records),
            "failures": specialist_failure_records,
        })
        save_text(
            session_dir / f"{safe_name}_failure.md",
            "\n".join([
                f"# {name} — Specialist Unavailable",
                "",
                "The specialist exhausted its configured recovery paths and did not produce a complete report.",
                "",
                f"- Status: FAILED",
                f"- Error type: {type(exc).__name__}",
                f"- Error: {exc}",
                "",
                "This failure must not be interpreted as a successful clean review. Downstream aggregation stages are informed that this specialist evidence is unavailable.",
            ]),
        )
        limit = REVIEW_MAX_SPECIALIST_FAILURES_BEFORE_ABORT if REVIEW_ABORT_ON_SPECIALIST_FAILURES else "unbounded"
        log(
            "REVIEW",
            f"Specialist '{name}' failed after all recovery paths; continuing. Failure {len(specialist_failure_records)}/{limit} | {type(exc).__name__}: {exc}",
            "yellow",
        )
        if REVIEW_ABORT_ON_SPECIALIST_FAILURES and len(specialist_failure_records) > REVIEW_MAX_SPECIALIST_FAILURES_BEFORE_ABORT:
            raise RuntimeError(
                f"Specialist failure ceiling exceeded: {len(specialist_failure_records)} failures; "
                f"allowed {REVIEW_MAX_SPECIALIST_FAILURES_BEFORE_ABORT}. Aborting review for coverage safety."
            )
        return (
            f"[UNAVAILABLE SPECIALIST: {name}]\n"
            f"This specialist exhausted all configured recovery paths and did not complete.\n"
            f"Error: {type(exc).__name__}: {exc}\n"
            "Do not interpret the missing specialist report as evidence that no issue exists."
        )

    def run_stage(name: str, fn, *, continue_on_failure: bool = True):
        _update_review_stage(session_dir, name, "running")
        try:
            result = fn()
            if name == "micro_review":
                failure_file = session_dir / "micro_review_failures.json"
                try:
                    coverage = json.loads(failure_file.read_text(encoding="utf-8")) if failure_file.exists() else {}
                except Exception:
                    coverage = {}
                failed_count = int(coverage.get("failed_chunks", 0) or 0)
                if failed_count:
                    _update_review_stage(
                        session_dir, name, "completed_with_failures",
                        detail=f"{failed_count} micro chunk(s) unavailable; explicit failure records saved."
                    )
                else:
                    _update_review_stage(session_dir, name, "completed")
            else:
                _update_review_stage(session_dir, name, "completed")
            return result
        except Exception as exc:
            _update_review_stage(session_dir, name, "failed", detail=str(exc))
            if continue_on_failure:
                return _record_specialist_failure(name, exc)
            raise

    micro = run_stage("micro_review", lambda: run_micro_review(chunks, session.document_type, session_dir))
    try:
        micro_coverage = json.loads((session_dir / "micro_review_failures.json").read_text(encoding="utf-8"))
    except Exception:
        micro_coverage = {"stage": "micro_review", "status": "unknown", "failed_chunks": 0, "failures": []}
    save_json(session_dir / "review_coverage.json", {
        "mode": ACTIVE_MODE,
        "micro_review": micro_coverage,
        "specialist_failures": specialist_failure_records,
        "note": "Unavailable specialist chunks/stages are explicitly marked and must not be interpreted as successful review coverage.",
    })
    section = run_stage("section_quality", lambda: run_section_quality_review(session.document_type, outline, chunks, micro, session_dir))
    technical = run_stage("technical_review", lambda: run_technical_review(session.document_type, chunks, session_dir))
    visual = run_stage("visual_review", lambda: run_visual_review(pages, session_dir))
    equation = run_stage("equation_review", lambda: run_equation_review(pages, visual, session_dir))
    data = run_stage("data_review", lambda: run_data_review(pages, session_dir))
    statistical = run_stage("statistical_review", lambda: run_statistical_review(pages, session_dir))
    style = run_stage("style_review", lambda: run_style_review(chunks, session.document_type, session_dir))
    language = run_stage("language_review", lambda: run_language_review(chunks, session.document_type, session_dir))
    ai_style = run_stage("ai_style_review", lambda: run_ai_style_signal_review(chunks, session.document_type, session_dir))
    nomenclature = run_stage("nomenclature_review", lambda: run_nomenclature_units_review(pages, session_dir))
    citation = run_stage("citation_review", lambda: run_citation_review(session.document_type, references, session_dir))
    literature = run_stage("literature_review", lambda: run_literature_gap_review(session.document_type, pages, session_dir))
    novelty = run_stage("novelty_review", lambda: run_novelty_review(session.document_type, outline, micro, literature, session_dir))
    reproducibility = run_stage("reproducibility_review", lambda: run_reproducibility_review(session.document_type, pages, session_dir))
    experimental = run_stage("experimental_design_review", lambda: run_experimental_design_review(session.document_type, pages, session_dir))
    venue = run_stage("venue_review", lambda: run_venue_review(session.document_type, outline, session_dir))
    front = run_stage("front_matter_review", lambda: run_front_matter_review(pages, session_dir))
    save_json(session_dir / "review_coverage.json", {
        "mode": ACTIVE_MODE,
        "micro_review": micro_coverage,
        "specialist_failures": specialist_failure_records,
        "specialist_failure_count": len(specialist_failure_records),
        "specialist_failure_limit": REVIEW_MAX_SPECIALIST_FAILURES_BEFORE_ABORT if REVIEW_ABORT_ON_SPECIALIST_FAILURES else None,
        "note": "Unavailable specialist chunks/stages are explicitly marked and must not be interpreted as successful review coverage.",
    })
    if specialist_failure_records:
        log("REVIEW", f"Initial specialist stage completed with {len(specialist_failure_records)} unavailable specialist stage(s); downstream coordination will continue with explicit failure records.", "yellow")
    _ = section, technical, equation, data, statistical, style, language, ai_style, nomenclature, citation, literature, novelty, reproducibility, experimental, venue, front

    # Adaptive coordination loop. The coordinator reads specialist evidence, selects
    # only high-value second passes, and can do this for a configurable number of rounds.
    coordinator_actions_all: list[dict[str, Any]] = []
    _update_review_stage(session_dir, "coordinator", "running")
    for round_no in range(1, REVIEW_COORDINATOR_MAX_ROUNDS + 1):
        actions = run_review_coordinator(session, session_dir, pages, chunks, round_no)
        if not actions:
            log("COORD", f"No additional targeted reviews requested in round {round_no}.", "green")
            break
        log("COORD", f"Round {round_no}: {len(actions)} targeted re-review action(s) requested.", "cyan")
        for action in actions:
            log("COORD", f"{action['agent']}: {action['task']}", "blue")
            result = run_targeted_coordinator_review(action["agent"], action["task"], session, session_dir, pages, chunks)
            action_record = dict(action)
            action_record["result_file"] = f"coordinator_{action['agent']}_*.md"
            action_record["result_preview"] = result[:1500]
            coordinator_actions_all.append(action_record)
        save_json(session_dir / "coordinator_actions_all.json", coordinator_actions_all)
    _update_review_stage(session_dir, "coordinator", "completed")

    _update_review_stage(session_dir, "hostile_reviewer", "running")
    hostile = run_hostile_reviewer(session_dir, session)
    _update_review_stage(session_dir, "hostile_reviewer", "completed")

    # One post-hostile coordination pass catches conflicts introduced by the adversarial review.
    if REVIEW_COORDINATOR_ENABLED and REVIEW_COORDINATOR_MAX_ROUNDS > 1:
        post_actions = run_review_coordinator(session, session_dir, pages, chunks, REVIEW_COORDINATOR_MAX_ROUNDS + 1)
        if post_actions:
            log("COORD", f"Post-hostile round: {len(post_actions)} targeted action(s).", "cyan")
            for action in post_actions[:REVIEW_COORDINATOR_MAX_ACTIONS_PER_ROUND]:
                run_targeted_coordinator_review(action["agent"], action["task"], session, session_dir, pages, chunks)

    _update_review_stage(session_dir, "final_arbiter", "running")
    final = run_final_arbiter(session_dir, session)
    _update_review_stage(session_dir, "final_arbiter", "completed")
    save_text(session_dir / "final_integrated_with_hostile_review.md", final + "\n\n===== REVIEWER #2 =====\n\n" + hostile)

    session.status = "reviewed"
    session.mode = ACTIVE_MODE
    session.updated = datetime.now().isoformat(timespec="seconds")
    save_json(session_dir / "session.json", asdict(session))
    return final


# -----------------------------------------------------------------------------
# Interactive CLI
# -----------------------------------------------------------------------------
def sync_reviewer_pdfs() -> list[Path]:
    REVIEWER_PDF_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in REVIEWER_PDF_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"], key=lambda p: (p.name.lower(), str(p).lower()))
    log("PDFS", f"synced reviewer PDF folder: {len(files)} PDF(s) found", "green")
    return files


def list_reviewer_pdfs() -> list[Path]:
    return sorted([p for p in REVIEWER_PDF_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"], key=lambda p: (p.name.lower(), str(p).lower()))


def choose_pdf_from_folder() -> Path | None:
    while True:
        files = list_reviewer_pdfs()
        print("\nReviewer PDF folder:")
        print(f"  {REVIEWER_PDF_DIR}")
        if not files:
            print("\nNo PDF files are currently visible in this folder.")
            print("Add your PDF to this folder and choose 's' to sync/rescan.")
            print("Choose: s=sync | p=enter an explicit PDF path | q=cancel")
        else:
            print("\nAvailable PDFs:")
            for i, p in enumerate(files, 1):
                rel = p.relative_to(REVIEWER_PDF_DIR)
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  {i}. {rel} ({size_mb:.1f} MB)")
            print("\nSelect a number, or s=sync | p=explicit path | q=cancel")
        raw = input("PDF selection: ").strip()
        if raw.lower() in {"q", "quit", "cancel"}:
            return None
        if raw.lower() in {"s", "sync"}:
            sync_reviewer_pdfs()
            continue
        if raw.lower() in {"p", "path"}:
            while True:
                entered = input("PDF path: ").strip().strip("\"'")
                if not entered:
                    break
                p = Path(os.path.expanduser(entered)).resolve()
                if p.exists() and p.is_file() and p.suffix.lower() == ".pdf":
                    return p
                print("Invalid PDF path.")
            continue
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except ValueError:
            pass
        print("Invalid selection. Use a listed number, s, p, or q.")


def start_new_review() -> tuple[ReviewSession, Path, str] | None:
    # Every new review begins with document selection from the dedicated PDF attachment folder.
    pdf = choose_pdf_from_folder()
    if pdf is None:
        return None
    print("\nDocument type: [a]uto  [p]aper  [t]hesis  [r]eport  [q]proposal")
    choice = input("Type [a/p/t/r/q, Enter=auto]: ").strip().lower()
    mapping = {"p": "research paper/article", "t": "thesis/dissertation", "r": "technical/report document", "q": "research proposal"}
    if choice in mapping:
        doc_type = mapping[choice]
    else:
        with fitz.open(pdf) as doc:
            sample = "\n".join((pg.get_text("text") or "") for pg in list(doc)[:5])
        doc_type = detect_document_type(sample)
    session, session_dir = create_session(pdf, doc_type)
    log("SESSION", f"Created {session_dir.name}", "green")
    print(f"Document: {pdf}")
    print(f"Detected type: {doc_type}")
    if ACTIVE_MODE == "short":
        print("\nStarting SHORT mode: every review stage, every configured specialist, every visual page, coordinator, Reviewer #2 and final arbiter still run; only reasoning/context/output budgets are reduced.\n")
    else:
        print("\nStarting FULL mode: complete quality-first review with proactive context compression and recovery.\n")
    review = run_initial_review(session, session_dir)
    return session, session_dir, review

def resume_review() -> tuple[ReviewSession, Path] | None:
    sessions = list_sessions()
    if not sessions:
        print("No previous review sessions found.")
        return None
    print("\nPrevious review sessions:\n")
    for i, (path, s) in enumerate(sessions, start=1):
        ready = "reviewed" if s.status == "reviewed" else s.status
        print(f"  {i}. {s.title} | {s.document_type} | {s.pages} pages | {ready} | {s.updated}")
    while True:
        raw = input("Select session number (Enter to cancel): ").strip()
        if not raw:
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(sessions):
                return load_session(sessions[idx][0])
        except ValueError:
            pass
        print("Invalid selection.")


def session_loop(session: ReviewSession, session_dir: Path, initial_review: str | None = None) -> int:
    if initial_review:
        print("\n" + "=" * 110)
        print(colour("INITIAL REVIEW", "cyan"))
        print("=" * 110)
        print(initial_review)
        print("=" * 110)
        print(f"Saved to: {session_dir / 'initial_review.md'}")
    else:
        print("\n" + "=" * 110)
        print(colour(f"RESUMED REVIEW: {session.title}", "cyan"))
        print(f"Document : {session.pdf_copy}")
        print(f"Type     : {session.document_type}")
        print(f"Status   : {session.status}")
        print("=" * 110)

    print("\nCommands: q=quit/save | review=show full review | files=list review files | humanize=natural academic rewrite")
    print("You can ask any follow-up question, challenge a suggestion, or request a rewrite.\n")
    while True:
        try:
            q = input("Reviewer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nReview session saved.")
            return 0
        if not q:
            continue
        if q.lower() in {"q", "quit", "exit"}:
            print(f"Session saved: {session_dir}")
            return 0
        if q.lower() in {"review", "full review"}:
            p = session_dir / "final_integrated_with_hostile_review.md"
            if not p.exists():
                p = session_dir / "initial_review.md"
            print("\n" + (p.read_text(encoding="utf-8") if p.exists() else "No review yet.") + "\n")
            continue
        if q.lower() in {"hostile", "reviewer 2", "reviewer #2"}:
            p = session_dir / "hostile_reviewer2.md"
            print("\n" + (p.read_text(encoding="utf-8") if p.exists() else "No Reviewer #2 report yet.") + "\n")
            continue
        if q.lower() in {"files", "list files"}:
            for p in sorted(session_dir.iterdir()):
                print(f"  {p.name}")
            continue
        if q.lower() in {"humanize", "humanize paragraph", "rewrite naturally"}:
            print("Paste the paragraph to rewrite. Finish with a blank line on its own line.")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            paragraph = "\n".join(lines).strip()
            if paragraph:
                answer = humanize_text(session, paragraph)
                print("\n" + "=" * 110)
                print(answer)
                print("=" * 110 + "\n")
            continue
        answer = followup_answer(session_dir, session, q)
        print("\n" + "=" * 110)
        print(answer)
        print("=" * 110 + "\n")


def self_test() -> int:
    checks = {
        "requests": requests is not None,
        "pymupdf": fitz is not None,
        "ddgs": DDGS is not None,
        "ollama": False,
    }
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        checks["ollama"] = r.ok
    except Exception:
        pass
    for k, v in checks.items():
        print(f"{k:12s}: {'OK' if v else 'MISSING/UNAVAILABLE'}")
    print(f"sessions_dir : {SESSIONS_DIR}")
    print(f"reviewer_pdf_dir : {REVIEWER_PDF_DIR}")
    return 0 if all(checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep academic PDF reviewer")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("full", "short"), default=REVIEW_MODE_ENV)
    parser.add_argument("--short", action="store_true", help="Run the complete review pipeline with reduced context/output budgets and thinking disabled")
    args = parser.parse_args()
    _set_runtime_mode("short" if args.short else args.mode)
    if args.self_test:
        return self_test()

    print("\n" + "=" * 110)
    print(colour("RESEARCH REVIEWER", "cyan"))
    print("Independent multi-agent reviewer for papers, theses, reports and proposals")
    print(colour(f"Runtime mode: {ACTIVE_MODE.upper()} | all review stages/pages remain enabled", "magenta"))
    print("=" * 110)
    print("1. Start a new review")
    print("2. Resume a previous review")
    print("3. Sync reviewer PDFs")
    print("q. Quit")
    print(f"Reviewer PDF folder: {REVIEWER_PDF_DIR}")
    print("\nIMPORTANT: every new review begins by asking for the PDF.")
    print("The reviewer uses a coordinator + specialist agents, including dedicated language/voice analysis. It does not determine AI authorship; it flags machine-like style signals and offers natural academic edits.\n")

    while True:
        choice = input("Select: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return 0
        if choice == "1":
            try:
                result = start_new_review()
                if result:
                    session, session_dir, review = result
                    return session_loop(session, session_dir, review)
            except Exception as exc:
                log("ERROR", str(exc), "red")
            continue
        if choice == "2":
            selected = resume_review()
            if selected:
                session, session_dir = selected
                return session_loop(session, session_dir)
            continue
        if choice == "3":
            sync_reviewer_pdfs()
            continue
        print("Please select 1, 2, 3 or q.")


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import time
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

try:
    import pymupdf as fitz
except ImportError:
    import fitz
import requests
import trafilatura
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rapidfuzz.fuzz import ratio
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env.deep_research_engine")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Model roles:
#   QwQ 32B       -> planning, adversarial review, final review
#   Qwen3.5 35B   -> high-volume search strategy, web/paper research, extraction
#   Qwen3 32B     -> synthesis and final report writing
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwq:32b")
SEARCH_STRATEGIST_MODEL = os.getenv("SEARCH_STRATEGIST_MODEL", "qwen3.5:35b-a3b")
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "qwen3.5:35b-a3b")
DEEP_ANALYSER_MODEL = os.getenv("DEEP_ANALYSER_MODEL", "qwen3.5:35b-a3b")
BLIND_RESEARCH_MODEL = os.getenv("BLIND_RESEARCH_MODEL", "qwen3.5:35b-a3b")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "qwq:32b")
GAP_MODEL = os.getenv("GAP_MODEL", "qwen3.5:35b-a3b")
WRITER_MODEL = os.getenv("WRITER_MODEL", "qwen3:32b")

# Research-mode workload settings. The deep-research values are the defaults.
# Short research temporarily overrides these with SHORT_* settings.
DEEP_MIN_CANDIDATE_PAPERS = int(os.getenv("DEEP_MIN_CANDIDATE_PAPERS", "150"))
DEEP_DISCOVERY_TARGET = int(os.getenv("DEEP_DISCOVERY_TARGET", "225"))
DEEP_PAPERS_TO_DEEP_READ = int(os.getenv("DEEP_PAPERS_TO_DEEP_READ", "50"))
DEEP_MAX_LLM_CONCURRENCY = int(os.getenv("DEEP_MAX_LLM_CONCURRENCY", "1"))
DEEP_MAX_TASK_CONCURRENCY = int(os.getenv("DEEP_MAX_TASK_CONCURRENCY", "1"))
DEEP_SEARCH_QUERY_COUNT = int(os.getenv("DEEP_SEARCH_QUERY_COUNT", "20"))
DEEP_WEB_QUERY_COUNT = int(os.getenv("DEEP_WEB_QUERY_COUNT", "12"))
DEEP_BLIND_QUERY_COUNT = int(os.getenv("DEEP_BLIND_QUERY_COUNT", "20"))
DEEP_MAX_SEARCH_PAGES = int(os.getenv("DEEP_MAX_SEARCH_PAGES", "3"))
DEEP_MAX_CITATION_HUBS = int(os.getenv("DEEP_MAX_CITATION_HUBS", "12"))
DEEP_RUN_WEB_DEEP_READ = os.getenv("DEEP_RUN_WEB_DEEP_READ", "true").lower() == "true"

SHORT_MIN_CANDIDATE_PAPERS = int(os.getenv("SHORT_MIN_CANDIDATE_PAPERS", "20"))
SHORT_DISCOVERY_TARGET = int(os.getenv("SHORT_DISCOVERY_TARGET", "30"))
SHORT_PAPERS_TO_DEEP_READ = int(os.getenv("SHORT_PAPERS_TO_DEEP_READ", "5"))
SHORT_MAX_LLM_CONCURRENCY = int(os.getenv("SHORT_MAX_LLM_CONCURRENCY", "1"))
SHORT_MAX_TASK_CONCURRENCY = int(os.getenv("SHORT_MAX_TASK_CONCURRENCY", "1"))
SHORT_SEARCH_QUERY_COUNT = int(os.getenv("SHORT_SEARCH_QUERY_COUNT", "5"))
SHORT_WEB_QUERY_COUNT = int(os.getenv("SHORT_WEB_QUERY_COUNT", "3"))
SHORT_BLIND_QUERY_COUNT = int(os.getenv("SHORT_BLIND_QUERY_COUNT", "5"))
SHORT_MAX_SEARCH_PAGES = int(os.getenv("SHORT_MAX_SEARCH_PAGES", "1"))
SHORT_MAX_CITATION_HUBS = int(os.getenv("SHORT_MAX_CITATION_HUBS", "3"))
SHORT_MAX_WEB_RESULTS = int(os.getenv("SHORT_MAX_WEB_RESULTS", "5"))
SHORT_RUN_WEB_DEEP_READ = os.getenv("SHORT_RUN_WEB_DEEP_READ", "false").lower() == "true"

# A few document/web limits are shared across both modes. The result-count
# setting is mode-specific because web retrieval volume should scale with the
# selected research depth.
DEEP_MAX_WEB_RESULTS = int(os.getenv("DEEP_MAX_WEB_RESULTS", "10"))
PDF_MAX_CHARS = int(os.getenv("PDF_MAX_CHARS", "160000"))
WEB_MAX_CHARS = int(os.getenv("WEB_MAX_CHARS", "90000"))

# Active mode values start with the deep-research defaults; apply_research_mode()
# resets them before the run starts.
RESEARCH_MODE = "deep"
MIN_CANDIDATE_PAPERS = DEEP_MIN_CANDIDATE_PAPERS
DISCOVERY_TARGET = DEEP_DISCOVERY_TARGET
DEEP_READ_TOP = DEEP_PAPERS_TO_DEEP_READ
MAX_LLM_CONCURRENCY = DEEP_MAX_LLM_CONCURRENCY
MAX_TASK_CONCURRENCY = DEEP_MAX_TASK_CONCURRENCY
SEARCH_QUERY_COUNT = DEEP_SEARCH_QUERY_COUNT
WEB_QUERY_COUNT = DEEP_WEB_QUERY_COUNT
BLIND_QUERY_COUNT = DEEP_BLIND_QUERY_COUNT
MAX_SEARCH_PAGES = DEEP_MAX_SEARCH_PAGES
MAX_CITATION_HUBS = DEEP_MAX_CITATION_HUBS
MAX_WEB_RESULTS = DEEP_MAX_WEB_RESULTS

# Scholarly API behaviour. The engine deliberately uses slow, serialized
# requests for public/anonymous access. Each provider can disable itself for
# the remainder of a run after repeated transient/auth/rate-limit failures.
SEMANTIC_SCHOLAR_ENABLED = os.getenv("SEMANTIC_SCHOLAR_ENABLED", "true").lower() == "true"
S2_MAX_RETRIES = int(os.getenv("S2_MAX_RETRIES", "2"))
S2_MIN_INTERVAL = float(os.getenv("S2_MIN_INTERVAL", "2.0"))
S2_BACKOFF_BASE = float(os.getenv("S2_BACKOFF_BASE", "3.0"))
S2_BACKOFF_MAX = float(os.getenv("S2_BACKOFF_MAX", "60"))
S2_AUTO_DISABLE_ON_ERROR = os.getenv("S2_AUTO_DISABLE_ON_ERROR", "true").lower() == "true"
S2_FAILURES_BEFORE_DISABLE = int(os.getenv("S2_FAILURES_BEFORE_DISABLE", "1"))

API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
API_MIN_INTERVAL = float(os.getenv("API_MIN_INTERVAL", "1.5"))
API_BACKOFF_BASE = float(os.getenv("API_BACKOFF_BASE", "3.0"))
API_BACKOFF_MAX = float(os.getenv("API_BACKOFF_MAX", "60"))

OPENALEX_ENABLED = os.getenv("OPENALEX_ENABLED", "true").lower() == "true"
OPENALEX_MIN_INTERVAL = float(os.getenv("OPENALEX_MIN_INTERVAL", "2.0"))
OPENALEX_MAX_RETRIES = int(os.getenv("OPENALEX_MAX_RETRIES", "2"))
OPENALEX_BACKOFF_MAX = float(os.getenv("OPENALEX_BACKOFF_MAX", "90"))
OPENALEX_AUTO_DISABLE_ON_ERROR = os.getenv("OPENALEX_AUTO_DISABLE_ON_ERROR", "true").lower() == "true"
OPENALEX_FAILURES_BEFORE_DISABLE = int(os.getenv("OPENALEX_FAILURES_BEFORE_DISABLE", "2"))
OPENALEX_PAGES = int(os.getenv("OPENALEX_PAGES", "1"))

# OpenAIRE is a public scholarly aggregation API and is enabled by default as
# an additional discovery source. Requests remain deliberately slow.
OPENAIRE_ENABLED = os.getenv("OPENAIRE_ENABLED", "true").lower() == "true"
OPENAIRE_MIN_INTERVAL = float(os.getenv("OPENAIRE_MIN_INTERVAL", "2.0"))
OPENAIRE_MAX_RETRIES = int(os.getenv("OPENAIRE_MAX_RETRIES", "2"))
OPENAIRE_BACKOFF_MAX = float(os.getenv("OPENAIRE_BACKOFF_MAX", "60"))
OPENAIRE_AUTO_DISABLE_ON_ERROR = os.getenv("OPENAIRE_AUTO_DISABLE_ON_ERROR", "true").lower() == "true"
OPENAIRE_FAILURES_BEFORE_DISABLE = int(os.getenv("OPENAIRE_FAILURES_BEFORE_DISABLE", "2"))
OPENAIRE_PAGE_SIZE = int(os.getenv("OPENAIRE_PAGE_SIZE", "100"))

# DBLP scholarly discovery. DBLP recommends roughly 1–2 seconds between requests.
DBLP_ENABLED = os.getenv("DBLP_ENABLED", "true").lower() == "true"
DBLP_MIN_INTERVAL = float(os.getenv("DBLP_MIN_INTERVAL", "2.0"))
DBLP_MAX_RETRIES = int(os.getenv("DBLP_MAX_RETRIES", "2"))
DBLP_BACKOFF_MAX = float(os.getenv("DBLP_BACKOFF_MAX", "60"))
DBLP_AUTO_DISABLE_ON_ERROR = os.getenv("DBLP_AUTO_DISABLE_ON_ERROR", "true").lower() == "true"
DBLP_FAILURES_BEFORE_DISABLE = int(os.getenv("DBLP_FAILURES_BEFORE_DISABLE", "2"))
DBLP_PAGE_SIZE = int(os.getenv("DBLP_PAGE_SIZE", "100"))

# arXiv scholarly discovery. arXiv recommends a 3-second delay between API calls.
ARXIV_ENABLED = os.getenv("ARXIV_ENABLED", "true").lower() == "true"
ARXIV_MIN_INTERVAL = float(os.getenv("ARXIV_MIN_INTERVAL", "3.0"))
ARXIV_MAX_RETRIES = int(os.getenv("ARXIV_MAX_RETRIES", "2"))
ARXIV_BACKOFF_MAX = float(os.getenv("ARXIV_BACKOFF_MAX", "90"))
ARXIV_AUTO_DISABLE_ON_ERROR = os.getenv("ARXIV_AUTO_DISABLE_ON_ERROR", "true").lower() == "true"
ARXIV_FAILURES_BEFORE_DISABLE = int(os.getenv("ARXIV_FAILURES_BEFORE_DISABLE", "2"))
ARXIV_PAGE_SIZE = int(os.getenv("ARXIV_PAGE_SIZE", "100"))

# GitHub is a technical/code source, not part of the scholarly-paper count.
GITHUB_ENABLED = os.getenv("GITHUB_ENABLED", "true").lower() == "true"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_MIN_INTERVAL = float(os.getenv("GITHUB_MIN_INTERVAL", "4.0"))
GITHUB_MAX_RETRIES = int(os.getenv("GITHUB_MAX_RETRIES", "2"))
GITHUB_BACKOFF_MAX = float(os.getenv("GITHUB_BACKOFF_MAX", "90"))
GITHUB_AUTO_DISABLE_ON_ERROR = os.getenv("GITHUB_AUTO_DISABLE_ON_ERROR", "true").lower() == "true"
GITHUB_FAILURES_BEFORE_DISABLE = int(os.getenv("GITHUB_FAILURES_BEFORE_DISABLE", "2"))
GITHUB_MAX_RESULTS = int(os.getenv("GITHUB_MAX_RESULTS", "20"))

# Reddit is a technical/community source. We use public web search instead of
# direct scraping/API automation, and never count Reddit posts as papers.
REDDIT_ENABLED = os.getenv("REDDIT_ENABLED", "true").lower() == "true"
REDDIT_MIN_INTERVAL = float(os.getenv("REDDIT_MIN_INTERVAL", "4.0"))
REDDIT_MAX_RESULTS = int(os.getenv("REDDIT_MAX_RESULTS", "10"))

# Google Scholar is available through an optional third-party structured API.
# Direct scraping is intentionally not used. Leave disabled until a provider
# such as SerpApi is configured.
GOOGLE_SCHOLAR_ENABLED = os.getenv("GOOGLE_SCHOLAR_ENABLED", "false").lower() == "true"
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()
GOOGLE_SCHOLAR_MIN_INTERVAL = float(os.getenv("GOOGLE_SCHOLAR_MIN_INTERVAL", "3.0"))
GOOGLE_SCHOLAR_MAX_RETRIES = int(os.getenv("GOOGLE_SCHOLAR_MAX_RETRIES", "2"))
GOOGLE_SCHOLAR_BACKOFF_MAX = float(os.getenv("GOOGLE_SCHOLAR_BACKOFF_MAX", "90"))
GOOGLE_SCHOLAR_AUTO_DISABLE_ON_ERROR = os.getenv("GOOGLE_SCHOLAR_AUTO_DISABLE_ON_ERROR", "true").lower() == "true"
GOOGLE_SCHOLAR_FAILURES_BEFORE_DISABLE = int(os.getenv("GOOGLE_SCHOLAR_FAILURES_BEFORE_DISABLE", "2"))
GOOGLE_SCHOLAR_RUNTIME_ENABLED = GOOGLE_SCHOLAR_ENABLED and bool(SERPAPI_API_KEY)

# Unpaywall is useful for DOI -> open-access resolution, not broad search.
UNPAYWALL_ENABLED = os.getenv("UNPAYWALL_ENABLED", "true").lower() == "true"
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "").strip()

# Local LLM behaviour. Long-running inference is expected on this machine.
# These defaults are tuned for a 16-GB VRAM GPU where Qwen3:32B is CPU/GPU split.

LLM_HEARTBEAT_SECONDS = float(os.getenv("LLM_HEARTBEAT_SECONDS", "15"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "2"))
LLM_HTTP_TIMEOUT_SECONDS = float(os.getenv("LLM_HTTP_TIMEOUT_SECONDS", "1800"))

# Role-specific context/output limits. These avoid spending a large context
# window on small search/planning requests while keeping final writing large.
LLM_CTX_DEFAULT = int(os.getenv("LLM_CTX_DEFAULT", "16384"))
LLM_CTX_DEEP = int(os.getenv("LLM_CTX_DEEP", "24576"))
LLM_CTX_SYNTHESIS = int(os.getenv("LLM_CTX_SYNTHESIS", "24576"))
LLM_CTX_WRITER = int(os.getenv("LLM_CTX_WRITER", "24576"))
LLM_CTX_TASK_SYNTHESIS = int(os.getenv("LLM_CTX_TASK_SYNTHESIS", "16384"))
LLM_TOKENS_PLANNER = int(os.getenv("LLM_TOKENS_PLANNER", "4096"))
LLM_TOKENS_SEARCH = int(os.getenv("LLM_TOKENS_SEARCH", "2048"))
LLM_TOKENS_SOURCE = int(os.getenv("LLM_TOKENS_SOURCE", "1024"))
LLM_TOKENS_PAPER = int(os.getenv("LLM_TOKENS_PAPER", "4096"))
LLM_TOKENS_CRITIC = int(os.getenv("LLM_TOKENS_CRITIC", "4096"))
LLM_TOKENS_GAP = int(os.getenv("LLM_TOKENS_GAP", "3072"))
LLM_TOKENS_TASK_SYNTHESIS = int(os.getenv("LLM_TOKENS_TASK_SYNTHESIS", "3000"))
LLM_TOKENS_SYNTHESIS = int(os.getenv("LLM_TOKENS_SYNTHESIS", "5000"))
LLM_TOKENS_WRITER = int(os.getenv("LLM_TOKENS_WRITER", "7000"))
LLM_TOKENS_REVIEW = int(os.getenv("LLM_TOKENS_REVIEW", "3072"))
LLM_TOKENS_WEB = int(os.getenv("LLM_TOKENS_WEB", "4096"))
LLM_TOKENS_REPAIR = int(os.getenv("LLM_TOKENS_REPAIR", "7000"))

# Hierarchical synthesis limits. We never send the complete multi-task evidence
# corpus to a single local-model context window.
SYNTHESIS_TASK_INPUT_CHARS = int(os.getenv("SYNTHESIS_TASK_INPUT_CHARS", "30000"))
SYNTHESIS_TASK_SUMMARY_CHARS = int(os.getenv("SYNTHESIS_TASK_SUMMARY_CHARS", "6500"))
SYNTHESIS_GLOBAL_INPUT_CHARS = int(os.getenv("SYNTHESIS_GLOBAL_INPUT_CHARS", "60000"))
SYNTHESIS_FINAL_INPUT_CHARS = int(os.getenv("SYNTHESIS_FINAL_INPUT_CHARS", "24000"))

# Reasoning flags. QwQ keeps its native reasoning behaviour. High-volume
# Qwen3.5/Qwen3 tasks disable extra reasoning to reduce needless latency.
LLM_REASONING_PLANNER = os.getenv("LLM_REASONING_PLANNER", "native").lower()
LLM_REASONING_SEARCH = os.getenv("LLM_REASONING_SEARCH", "false").lower()
LLM_REASONING_SOURCE = os.getenv("LLM_REASONING_SOURCE", "false").lower()
LLM_REASONING_PAPER = os.getenv("LLM_REASONING_PAPER", "false").lower()
LLM_REASONING_CRITIC = os.getenv("LLM_REASONING_CRITIC", "native").lower()
LLM_REASONING_GAP = os.getenv("LLM_REASONING_GAP", "native").lower()
LLM_REASONING_REVIEW = os.getenv("LLM_REASONING_REVIEW", "native").lower()
LLM_REASONING_WRITER = os.getenv("LLM_REASONING_WRITER", "false").lower()

RUN_WEB_DEEP_READ = DEEP_RUN_WEB_DEEP_READ
ENABLE_YOUTUBE = os.getenv("ENABLE_YOUTUBE", "true").lower() == "true"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "").strip()  # retained for compatibility; OpenAlex no longer uses the old polite pool.
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "").strip()

USER_AGENT = os.getenv(
    "RESEARCH_USER_AGENT",
    "LocalDeepResearchEngine/2.1 (academic research workflow; contact via configured email)",
)

HEADERS = {"User-Agent": USER_AGENT}

if not OPENALEX_API_KEY:
    print(
        "[OpenAlex] No API key configured; running in slow anonymous mode. "
        "A free key can be added later without code changes.",
        flush=True,
    )
if SEMANTIC_SCHOLAR_ENABLED and not SEMANTIC_SCHOLAR_API_KEY:
    print(
        "[Semantic Scholar] No API key configured; using slow shared anonymous mode. "
        "A free key can be added later without code changes.",
        flush=True,
    )
if GOOGLE_SCHOLAR_ENABLED and not SERPAPI_API_KEY:
    print(
        "[Google Scholar] Enabled but no SERPAPI_API_KEY is configured; source disabled for this run.",
        flush=True,
    )
if GITHUB_ENABLED and not GITHUB_TOKEN:
    print(
        "[GitHub] No token configured; using slow anonymous repository search. A token can be added later for higher limits.",
        flush=True,
    )
if REDDIT_ENABLED:
    print(
        "[Reddit] Using slow web-search discovery; Reddit posts are contextual evidence and do not count as papers.",
        flush=True,
    )

LLM_SEM = asyncio.Semaphore(MAX_LLM_CONCURRENCY)
TASK_SEM = asyncio.Semaphore(MAX_TASK_CONCURRENCY)
S2_LOCK = threading.Lock()
S2_LAST_REQUEST = 0.0
S2_RUNTIME_ENABLED = SEMANTIC_SCHOLAR_ENABLED
S2_ERROR_STREAK = 0

API_LOCK = threading.Lock()
API_LAST_REQUEST = 0.0
OPENALEX_LOCK = threading.Lock()
OPENALEX_LAST_REQUEST = 0.0
OPENALEX_RUNTIME_ENABLED = OPENALEX_ENABLED
OPENALEX_ERROR_STREAK = 0
OPENAIRE_LOCK = threading.Lock()
OPENAIRE_LAST_REQUEST = 0.0
OPENAIRE_RUNTIME_ENABLED = OPENAIRE_ENABLED
OPENAIRE_ERROR_STREAK = 0
GOOGLE_SCHOLAR_LOCK = threading.Lock()
GOOGLE_SCHOLAR_LAST_REQUEST = 0.0
GOOGLE_SCHOLAR_ERROR_STREAK = 0
DBLP_LOCK = threading.Lock()
DBLP_LAST_REQUEST = 0.0
DBLP_RUNTIME_ENABLED = DBLP_ENABLED
DBLP_ERROR_STREAK = 0
ARXIV_LOCK = threading.Lock()
ARXIV_LAST_REQUEST = 0.0
ARXIV_RUNTIME_ENABLED = ARXIV_ENABLED
ARXIV_ERROR_STREAK = 0
GITHUB_LOCK = threading.Lock()
GITHUB_LAST_REQUEST = 0.0
GITHUB_RUNTIME_ENABLED = GITHUB_ENABLED
GITHUB_ERROR_STREAK = 0
REDDIT_LOCK = threading.Lock()
REDDIT_LAST_REQUEST = 0.0
REDDIT_RUNTIME_ENABLED = REDDIT_ENABLED
TECH_SOURCE_CACHE: dict[tuple[str, str], list[TechnicalSource]] = {}


# -----------------------------------------------------------------------------
# Run directories
# -----------------------------------------------------------------------------

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "runs" / RUN_STAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

class ResearchTask(BaseModel):
    name: str
    objective: str
    scope_notes: str = ""
    search_concepts: list[str] = Field(default_factory=list)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    title: str
    research_question: str
    tasks: list[ResearchTask]
    date_scope: str
    key_terms: list[str]


class QuerySet(BaseModel):
    # Search-strategy output is deliberately tolerant. Local models can occasionally
    # return only the academic query list even when the full schema is requested.
    # Missing auxiliary fields should not terminate an otherwise healthy run.
    academic_queries: list[str] = Field(default_factory=list)
    web_queries: list[str] = Field(default_factory=list)
    youtube_queries: list[str] = Field(default_factory=list)
    alternate_terminology: list[str] = Field(default_factory=list)
    source_types_to_seek: list[str] = Field(default_factory=list)


class SourceDecision(BaseModel):
    # LLM structured output is deliberately tolerant here. Local models can
    # occasionally omit an explanatory field even when the schema requests it.
    # Missing values must not terminate an otherwise healthy research run.
    use_youtube: bool = False
    website_types: list[str] = Field(default_factory=list)
    suggested_domains: list[str] = Field(default_factory=list)
    rationale: str = ""


class EvidenceCard(BaseModel):
    reference_key: str
    paper_title: str
    year: Optional[int] = None
    authors: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    research_problem: str
    method: str
    platform_or_dataset: str
    experiment_type: str
    disturbance_or_environment: str
    key_findings: list[str]
    limitations: list[str]
    relevance_to_task: str
    evidence_strength: str
    important_equations_or_control_structure: list[str] = Field(default_factory=list)


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    evidence_source: Optional[str] = None
    correction: Optional[str] = None


class Critique(BaseModel):
    strengths: list[str]
    missing_topics: list[str]
    unsupported_claims: list[ClaimCheck]
    conflicting_evidence: list[str]
    likely_missed_sources: list[str]
    additional_queries_by_task: dict[str, list[str]]
    required_repairs: list[str]


class GapPlan(BaseModel):
    # QwQ can occasionally omit some fields even when structured output is
    # requested. Defaults make the gap stage recoverable; make_gap_plans()
    # fills task-specific values and derives queries from alternate terms.
    task_name: str = ""
    overlooked_angles: list[str] = Field(default_factory=list)
    alternate_search_terms: list[str] = Field(default_factory=list)
    adjacent_fields: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


class FinalReview(BaseModel):
    factual_issues: list[str]
    citation_issues: list[str]
    structural_issues: list[str]
    missing_explanation: list[str]
    required_research: list[str]
    ready_for_pdf: bool


@dataclass
class Paper:
    title: str
    abstract: str = ""
    year: Optional[int] = None
    authors: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    pdf_url: str = ""
    source: str = ""
    citation_count: int = 0
    external_id: str = ""
    paper_id: str = ""


@dataclass
class TechnicalSource:
    title: str
    url: str = ""
    source: str = ""
    snippet: str = ""
    query: str = ""
    metadata: dict[str, Any] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "snippet": self.snippet,
            "query": self.query,
            "metadata": self.metadata or {},
        }


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------

class PaperDB:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT,
                year INTEGER,
                authors TEXT,
                venue TEXT,
                doi TEXT,
                url TEXT,
                pdf_url TEXT,
                source TEXT,
                external_id TEXT,
                citation_count INTEGER DEFAULT 0,
                base_score REAL DEFAULT 0,
                rerank_score REAL DEFAULT 0,
                deep_read INTEGER DEFAULT 0,
                last_pass TEXT DEFAULT ''
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_task ON papers(task)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_task_score ON papers(task, rerank_score DESC, citation_count DESC)")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS technical_sources (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT,
                source TEXT,
                snippet TEXT,
                query TEXT,
                metadata_json TEXT,
                pass_name TEXT DEFAULT ''
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tech_task ON technical_sources(task)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def normalize_title(title: str) -> str:
        title = re.sub(r"\s+", " ", title.lower()).strip()
        return re.sub(r"[^a-z0-9 ]", "", title)

    @classmethod
    def base_key_for(cls, paper: Paper) -> str:
        if paper.doi:
            return "doi:" + paper.doi.lower().strip().replace("https://doi.org/", "")
        if paper.external_id:
            return f"{paper.source}:{paper.external_id}"
        return "title:" + cls.normalize_title(paper.title)

    @classmethod
    def key_for(cls, task: str, paper: Paper) -> str:
        # The same paper may legitimately belong to several research tasks.
        return f"task:{task}::{cls.base_key_for(paper)}"

    def add(self, task: str, paper: Paper, base_score: float = 0.0, pass_name: str = "") -> bool:
        if not paper.title.strip():
            return False
        pid = self.key_for(task, paper)
        rows = self.conn.execute(
            "SELECT id, title, doi FROM papers WHERE task=? LIMIT 10000", (task,)
        ).fetchall()
        for row_id, title, doi in rows:
            if paper.doi and doi and paper.doi.lower() == doi.lower():
                return False
            if ratio(title.lower(), paper.title.lower()) >= 96:
                return False

        self.conn.execute(
            """
            INSERT OR IGNORE INTO papers
            (id, task, title, abstract, year, authors, venue, doi, url, pdf_url,
             source, external_id, citation_count, base_score, rerank_score, last_pass)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                task,
                paper.title.strip(),
                paper.abstract,
                paper.year,
                paper.authors,
                paper.venue,
                paper.doi,
                paper.url,
                paper.pdf_url,
                paper.source,
                paper.external_id,
                paper.citation_count,
                base_score,
                base_score,
                pass_name,
            ),
        )
        self.conn.commit()
        return True

    def count(self, task: str) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM papers WHERE task=?", (task,)).fetchone()[0])

    def source_counts(self, task: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT source, COUNT(*) FROM papers WHERE task=? GROUP BY source ORDER BY COUNT(*) DESC", (task,)
        ).fetchall()
        return {source: int(count) for source, count in rows}

    def add_technical_source(self, task: str, source: TechnicalSource, pass_name: str = "") -> bool:
        if not source.title.strip() or not source.url.strip():
            return False
        normalized = self.normalize_title(source.title) + "::" + source.url.strip().lower()
        sid = f"task:{task}::tech:{source.source}:{hashlib.sha1(normalized.encode('utf-8')).hexdigest()}"
        self.conn.execute(
            """
            INSERT OR IGNORE INTO technical_sources
            (id, task, title, url, source, snippet, query, metadata_json, pass_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sid, task, source.title.strip(), source.url.strip(), source.source, source.snippet,
             source.query, json.dumps(source.metadata or {}, ensure_ascii=False), pass_name),
        )
        inserted = self.conn.total_changes > 0
        self.conn.commit()
        return inserted

    def technical_sources(self, task: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT title, url, source, snippet, query, metadata_json, pass_name FROM technical_sources WHERE task=? ORDER BY id",
            (task,),
        ).fetchall()
        out = []
        for title, url, source, snippet, query, metadata_json, pass_name in rows:
            try:
                metadata = json.loads(metadata_json or "{}")
            except json.JSONDecodeError:
                metadata = {}
            out.append({"title": title, "url": url, "source": source, "snippet": snippet, "query": query, "metadata": metadata, "pass_name": pass_name})
        return out

    def technical_count(self, task: str) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM technical_sources WHERE task=?", (task,)).fetchone()[0])

    def all_for_task(self, task: str) -> list[Paper]:
        rows = self.conn.execute(
            """
            SELECT title, abstract, year, authors, venue, doi, url, pdf_url,
                   source, citation_count, external_id, id
            FROM papers WHERE task=?
            ORDER BY rerank_score DESC, citation_count DESC, year DESC
            """,
            (task,),
        ).fetchall()
        return [Paper(*r[:10], external_id=r[10], paper_id=r[11]) for r in rows]

    def top(self, task: str, n: int) -> list[Paper]:
        rows = self.conn.execute(
            """
            SELECT title, abstract, year, authors, venue, doi, url, pdf_url,
                   source, citation_count, external_id, id
            FROM papers WHERE task=?
            ORDER BY rerank_score DESC, citation_count DESC, year DESC
            LIMIT ?
            """,
            (task, n),
        ).fetchall()
        return [Paper(*r[:10], external_id=r[10], paper_id=r[11]) for r in rows]

    def update_rerank(self, paper_id: str, score: float) -> None:
        self.conn.execute("UPDATE papers SET rerank_score=? WHERE id=?", (score, paper_id))
        self.conn.commit()

    def mark_deep_read(self, paper_id: str, score: float) -> None:
        self.conn.execute(
            "UPDATE papers SET rerank_score=?, deep_read=1 WHERE id=?",
            (score, paper_id),
        )
        self.conn.commit()

    def export_csv(self, path: Path) -> None:
        rows = self.conn.execute(
            "SELECT task,title,year,authors,venue,doi,url,pdf_url,source,citation_count,rerank_score,deep_read FROM papers ORDER BY task, rerank_score DESC"
        ).fetchall()
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "task", "title", "year", "authors", "venue", "doi", "url", "pdf_url",
                "source", "citation_count", "rerank_score", "deep_read"
            ])
            writer.writerows(rows)


# -----------------------------------------------------------------------------
# LLM helpers
# -----------------------------------------------------------------------------

ROLE_CONFIG = {
    "planner": {
        "ctx": LLM_CTX_DEFAULT,
        "tokens": LLM_TOKENS_PLANNER,
        "reasoning": LLM_REASONING_PLANNER,
    },
    "search": {
        "ctx": LLM_CTX_DEFAULT,
        "tokens": LLM_TOKENS_SEARCH,
        "reasoning": LLM_REASONING_SEARCH,
    },
    "source": {
        "ctx": LLM_CTX_DEFAULT,
        "tokens": LLM_TOKENS_SOURCE,
        "reasoning": LLM_REASONING_SOURCE,
    },
    "paper": {
        "ctx": LLM_CTX_DEEP,
        "tokens": LLM_TOKENS_PAPER,
        "reasoning": LLM_REASONING_PAPER,
    },
    "critic": {
        "ctx": LLM_CTX_DEEP,
        "tokens": LLM_TOKENS_CRITIC,
        "reasoning": LLM_REASONING_CRITIC,
    },
    "gap": {
        "ctx": LLM_CTX_DEFAULT,
        "tokens": LLM_TOKENS_GAP,
        "reasoning": LLM_REASONING_GAP,
    },
    "task_synthesis": {
        "ctx": LLM_CTX_TASK_SYNTHESIS,
        "tokens": LLM_TOKENS_TASK_SYNTHESIS,
        "reasoning": LLM_REASONING_WRITER,
    },
    "synthesis": {
        "ctx": LLM_CTX_SYNTHESIS,
        "tokens": LLM_TOKENS_SYNTHESIS,
        "reasoning": LLM_REASONING_WRITER,
    },
    "writer": {
        "ctx": LLM_CTX_WRITER,
        "tokens": LLM_TOKENS_WRITER,
        "reasoning": LLM_REASONING_WRITER,
    },
    "review": {
        "ctx": LLM_CTX_DEFAULT,
        "tokens": LLM_TOKENS_REVIEW,
        "reasoning": LLM_REASONING_REVIEW,
    },
    "web": {
        "ctx": LLM_CTX_DEFAULT,
        "tokens": LLM_TOKENS_WEB,
        "reasoning": LLM_REASONING_SEARCH,
    },
    "repair": {
        "ctx": LLM_CTX_WRITER,
        "tokens": LLM_TOKENS_REPAIR,
        "reasoning": LLM_REASONING_WRITER,
    },
}


def _reasoning_value(role: str):
    value = ROLE_CONFIG[role]["reasoning"]
    if value in ("", "none", "native"):
        return None
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return value


def make_llm(model_name: str, temperature: float = 0, role: str = "search") -> ChatOllama:
    cfg = ROLE_CONFIG.get(role, ROLE_CONFIG["search"])
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "num_ctx": cfg["ctx"],
        "num_predict": cfg["tokens"],
        "client_kwargs": {"timeout": LLM_HTTP_TIMEOUT_SECONDS},
    }
    reasoning = _reasoning_value(role)
    if reasoning is not None:
        kwargs["reasoning"] = reasoning

    return ChatOllama(**kwargs)


async def invoke_with_progress(
    runnable,
    prompt,
    label: str,
    *,
    retries: int = LLM_RETRIES,
    heartbeat: float = LLM_HEARTBEAT_SECONDS,
):
    """Wait for a local LLM result while showing progress and retrying failures."""
    last_error = None
    for attempt in range(1, retries + 2):
        started = time.monotonic()
        print(f"[LLM] {label} | attempt {attempt}/{retries + 1} | waiting for response...", flush=True)
        task = asyncio.create_task(asyncio.to_thread(runnable.invoke, prompt))

        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=heartbeat)
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - started
                print(f"[LLM] {label} | still working | elapsed {elapsed:.0f}s", flush=True)

        try:
            result = task.result()
            elapsed = time.monotonic() - started
            print(f"[LLM] {label} | completed in {elapsed:.1f}s", flush=True)
            return result
        except Exception as exc:
            last_error = exc
            elapsed = time.monotonic() - started
            print(f"[LLM] {label} | failed after {elapsed:.1f}s: {exc}", flush=True)
            if attempt <= retries:
                delay = min(5.0 * attempt, 30.0)
                print(f"[LLM] {label} | retrying in {delay:.0f}s", flush=True)
                await asyncio.sleep(delay)

    raise RuntimeError(f"{label} failed after {retries + 1} attempts: {last_error}")


async def agent_invoke_with_progress(
    agent,
    messages,
    label: str,
    *,
    retries: int = LLM_RETRIES,
    heartbeat: float = LLM_HEARTBEAT_SECONDS,
):
    """Run a LangChain agent with progress reporting and retries."""
    last_error = None
    for attempt in range(1, retries + 2):
        started = time.monotonic()
        print(f"[AGENT] {label} | attempt {attempt}/{retries + 1} | waiting...", flush=True)
        task = asyncio.create_task(agent.ainvoke({"messages": messages}))

        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=heartbeat)
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - started
                print(f"[AGENT] {label} | still working | elapsed {elapsed:.0f}s", flush=True)

        try:
            result = task.result()
            elapsed = time.monotonic() - started
            print(f"[AGENT] {label} | completed in {elapsed:.1f}s", flush=True)
            return result
        except Exception as exc:
            last_error = exc
            elapsed = time.monotonic() - started
            print(f"[AGENT] {label} | failed after {elapsed:.1f}s: {exc}", flush=True)
            if attempt <= retries:
                delay = min(5.0 * attempt, 30.0)
                print(f"[AGENT] {label} | retrying in {delay:.0f}s", flush=True)
                await asyncio.sleep(delay)

    raise RuntimeError(f"{label} failed after {retries + 1} attempts: {last_error}")


planner_llm = make_llm(PLANNER_MODEL, 0, "planner").with_structured_output(ResearchPlan)
search_strategist_llm = make_llm(SEARCH_STRATEGIST_MODEL, 0, "search").with_structured_output(QuerySet)
source_decider_llm = make_llm(SEARCH_STRATEGIST_MODEL, 0, "source").with_structured_output(SourceDecision)
critic_llm = make_llm(CRITIC_MODEL, 0, "critic").with_structured_output(Critique)
gap_planner_llm = make_llm(GAP_MODEL, 0, "search").with_structured_output(GapPlan)
final_review_llm = make_llm(CRITIC_MODEL, 0, "review").with_structured_output(FinalReview)
evidence_llm = make_llm(DEEP_ANALYSER_MODEL, 0, "paper").with_structured_output(EvidenceCard)


# -----------------------------------------------------------------------------
# Search / fetch utilities
# -----------------------------------------------------------------------------

def _source_state(label: str):
    if label == "OpenAlex":
        return ("openalex", OPENALEX_LOCK, OPENALEX_MIN_INTERVAL, OPENALEX_MAX_RETRIES, OPENALEX_BACKOFF_MAX)
    if label == "OpenAIRE":
        return ("openaire", OPENAIRE_LOCK, OPENAIRE_MIN_INTERVAL, OPENAIRE_MAX_RETRIES, OPENAIRE_BACKOFF_MAX)
    if label == "Google Scholar":
        return ("google_scholar", GOOGLE_SCHOLAR_LOCK, GOOGLE_SCHOLAR_MIN_INTERVAL, GOOGLE_SCHOLAR_MAX_RETRIES, GOOGLE_SCHOLAR_BACKOFF_MAX)
    if label == "DBLP":
        return ("dblp", DBLP_LOCK, DBLP_MIN_INTERVAL, DBLP_MAX_RETRIES, DBLP_BACKOFF_MAX)
    if label == "arXiv":
        return ("arxiv", ARXIV_LOCK, ARXIV_MIN_INTERVAL, ARXIV_MAX_RETRIES, ARXIV_BACKOFF_MAX)
    if label == "GitHub":
        return ("github", GITHUB_LOCK, GITHUB_MIN_INTERVAL, GITHUB_MAX_RETRIES, GITHUB_BACKOFF_MAX)
    return ("generic", API_LOCK, API_MIN_INTERVAL, API_MAX_RETRIES, API_BACKOFF_MAX)


def _record_source_failure(source: str, detail: str) -> None:
    global OPENALEX_RUNTIME_ENABLED, OPENALEX_ERROR_STREAK
    global OPENAIRE_RUNTIME_ENABLED, OPENAIRE_ERROR_STREAK
    global GOOGLE_SCHOLAR_RUNTIME_ENABLED, GOOGLE_SCHOLAR_ERROR_STREAK
    global DBLP_RUNTIME_ENABLED, DBLP_ERROR_STREAK
    global ARXIV_RUNTIME_ENABLED, ARXIV_ERROR_STREAK
    global GITHUB_RUNTIME_ENABLED, GITHUB_ERROR_STREAK
    global REDDIT_RUNTIME_ENABLED

    if source == "openalex":
        OPENALEX_ERROR_STREAK += 1
        if OPENALEX_AUTO_DISABLE_ON_ERROR and OPENALEX_ERROR_STREAK >= OPENALEX_FAILURES_BEFORE_DISABLE:
            OPENALEX_RUNTIME_ENABLED = False
            print(f"[OpenAlex] Disabling source for the remainder of this run after repeated failures: {detail}", flush=True)
    elif source == "openaire":
        OPENAIRE_ERROR_STREAK += 1
        if OPENAIRE_AUTO_DISABLE_ON_ERROR and OPENAIRE_ERROR_STREAK >= OPENAIRE_FAILURES_BEFORE_DISABLE:
            OPENAIRE_RUNTIME_ENABLED = False
            print(f"[OpenAIRE] Disabling source for the remainder of this run after repeated failures: {detail}", flush=True)
    elif source == "google_scholar":
        GOOGLE_SCHOLAR_ERROR_STREAK += 1
        if GOOGLE_SCHOLAR_AUTO_DISABLE_ON_ERROR and GOOGLE_SCHOLAR_ERROR_STREAK >= GOOGLE_SCHOLAR_FAILURES_BEFORE_DISABLE:
            GOOGLE_SCHOLAR_RUNTIME_ENABLED = False
            print(f"[Google Scholar] Disabling source for the remainder of this run after repeated failures: {detail}", flush=True)
    elif source == "dblp":
        DBLP_ERROR_STREAK += 1
        if DBLP_AUTO_DISABLE_ON_ERROR and DBLP_ERROR_STREAK >= DBLP_FAILURES_BEFORE_DISABLE:
            DBLP_RUNTIME_ENABLED = False
            print(f"[DBLP] Disabling source for the remainder of this run after repeated failures: {detail}", flush=True)
    elif source == "arxiv":
        ARXIV_ERROR_STREAK += 1
        if ARXIV_AUTO_DISABLE_ON_ERROR and ARXIV_ERROR_STREAK >= ARXIV_FAILURES_BEFORE_DISABLE:
            ARXIV_RUNTIME_ENABLED = False
            print(f"[arXiv] Disabling source for the remainder of this run after repeated failures: {detail}", flush=True)
    elif source == "github":
        GITHUB_ERROR_STREAK += 1
        if GITHUB_AUTO_DISABLE_ON_ERROR and GITHUB_ERROR_STREAK >= GITHUB_FAILURES_BEFORE_DISABLE:
            GITHUB_RUNTIME_ENABLED = False
            print(f"[GitHub] Disabling source for the remainder of this run after repeated failures: {detail}", flush=True)


def _record_source_success(source: str) -> None:
    global OPENALEX_ERROR_STREAK, OPENAIRE_ERROR_STREAK, GOOGLE_SCHOLAR_ERROR_STREAK
    global DBLP_ERROR_STREAK, ARXIV_ERROR_STREAK, GITHUB_ERROR_STREAK
    if source == "openalex":
        OPENALEX_ERROR_STREAK = 0
    elif source == "openaire":
        OPENAIRE_ERROR_STREAK = 0
    elif source == "google_scholar":
        GOOGLE_SCHOLAR_ERROR_STREAK = 0
    elif source == "dblp":
        DBLP_ERROR_STREAK = 0
    elif source == "arxiv":
        ARXIV_ERROR_STREAK = 0
    elif source == "github":
        GITHUB_ERROR_STREAK = 0


def safe_get_json(
    url: str,
    params: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
    timeout: int = 45,
    label: str = "HTTP",
) -> dict[str, Any]:
    """Provider-aware request with slow pacing, retries, and self-disabling sources."""
    global API_LAST_REQUEST, OPENALEX_LAST_REQUEST, OPENAIRE_LAST_REQUEST, GOOGLE_SCHOLAR_LAST_REQUEST
    global DBLP_LAST_REQUEST, ARXIV_LAST_REQUEST, GITHUB_LAST_REQUEST

    source, lock, min_interval, max_retries, backoff_max = _source_state(label)
    last_error: Optional[Exception] = None

    with lock:
        for attempt in range(1, max_retries + 2):
            if source == "openalex" and not OPENALEX_RUNTIME_ENABLED:
                return {}
            if source == "openaire" and not OPENAIRE_RUNTIME_ENABLED:
                return {}
            if source == "google_scholar" and not GOOGLE_SCHOLAR_RUNTIME_ENABLED:
                return {}
            if source == "dblp" and not DBLP_RUNTIME_ENABLED:
                return {}
            if source == "arxiv" and not ARXIV_RUNTIME_ENABLED:
                return {}
            if source == "github" and not GITHUB_RUNTIME_ENABLED:
                return {}

            if source == "openalex":
                wait = min_interval - (time.monotonic() - OPENALEX_LAST_REQUEST)
            elif source == "openaire":
                wait = min_interval - (time.monotonic() - OPENAIRE_LAST_REQUEST)
            elif source == "google_scholar":
                wait = min_interval - (time.monotonic() - GOOGLE_SCHOLAR_LAST_REQUEST)
            elif source == "dblp":
                wait = min_interval - (time.monotonic() - DBLP_LAST_REQUEST)
            elif source == "arxiv":
                wait = min_interval - (time.monotonic() - ARXIV_LAST_REQUEST)
            elif source == "github":
                wait = min_interval - (time.monotonic() - GITHUB_LAST_REQUEST)
            else:
                wait = min_interval - (time.monotonic() - API_LAST_REQUEST)
            if wait > 0:
                time.sleep(wait)

            try:
                r = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
                now = time.monotonic()
                if source == "openalex":
                    OPENALEX_LAST_REQUEST = now
                elif source == "openaire":
                    OPENAIRE_LAST_REQUEST = now
                elif source == "google_scholar":
                    GOOGLE_SCHOLAR_LAST_REQUEST = now
                elif source == "dblp":
                    DBLP_LAST_REQUEST = now
                elif source == "arxiv":
                    ARXIV_LAST_REQUEST = now
                elif source == "github":
                    GITHUB_LAST_REQUEST = now
                else:
                    API_LAST_REQUEST = now

                if r.status_code == 200:
                    _record_source_success(source)
                    return r.json()

                retryable = r.status_code in {408, 425, 428, 429, 500, 502, 503, 504}
                # 400/422 generally indicate a malformed or unsupported query;
                # retrying the exact same request only wastes time.
                hard = r.status_code in {400, 401, 403, 422}
                if retryable:
                    retry_after = r.headers.get("Retry-After")
                    reset_after = r.headers.get("X-RateLimit-Reset")
                    if source == "openalex" and r.status_code == 429 and r.headers.get("X-RateLimit-Remaining") == "0":
                        _record_source_failure(source, "HTTP 429 with X-RateLimit-Remaining=0 (daily budget likely exhausted)")
                        print(
                            "[OpenAlex] Daily/credit budget appears exhausted; disabling OpenAlex for the remainder of this run.",
                            flush=True,
                        )
                        break
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = API_BACKOFF_BASE * (2 ** (attempt - 1))
                    elif reset_after:
                        try:
                            raw_reset = float(reset_after)
                            # Most providers expose an epoch timestamp in reset headers.
                            delay = max(0.0, raw_reset - time.time()) if raw_reset > 1_000_000_000 else raw_reset
                        except ValueError:
                            delay = API_BACKOFF_BASE * (2 ** (attempt - 1))
                    else:
                        delay = API_BACKOFF_BASE * (2 ** (attempt - 1))
                    delay = min(max(delay, min_interval), backoff_max)
                    print(
                        f"[{label}] HTTP {r.status_code}; retrying in {delay:.1f}s "
                        f"(attempt {attempt}/{max_retries + 1})",
                        flush=True,
                    )
                    last_error = requests.HTTPError(f"HTTP {r.status_code} for {url}", response=r)
                    if attempt <= max_retries:
                        time.sleep(delay)
                        continue
                    _record_source_failure(source, f"HTTP {r.status_code}")
                    break

                if hard:
                    last_error = requests.HTTPError(f"HTTP {r.status_code} for {url}", response=r)
                    _record_source_failure(source, f"HTTP {r.status_code}")
                    print(f"[{label}] HTTP {r.status_code}; source will be disabled after the configured failure threshold.", flush=True)
                    break

                r.raise_for_status()

            except requests.RequestException as exc:
                last_error = exc
                if attempt > max_retries:
                    _record_source_failure(source, str(exc))
                    break
                delay = min(max(API_BACKOFF_BASE * (2 ** (attempt - 1)), min_interval), backoff_max)
                print(
                    f"[{label}] request error; retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{max_retries + 1}): {exc}",
                    flush=True,
                )
                time.sleep(delay)

    if last_error is not None:
        raise last_error
    return {}


def s2_get_json(url: str, params: dict[str, Any], timeout: int = 45) -> dict[str, Any]:
    """Slow Semantic Scholar request with race-safe automatic source disable."""
    global S2_LAST_REQUEST, S2_RUNTIME_ENABLED, S2_ERROR_STREAK
    if not S2_RUNTIME_ENABLED:
        return {}

    headers = dict(HEADERS)
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    with S2_LOCK:
        # Another concurrent search may have disabled S2 while this worker was
        # waiting for the provider lock. Do not issue another request.
        if not S2_RUNTIME_ENABLED:
            return {}

        max_retries = S2_MAX_RETRIES
        for attempt in range(max_retries + 1):
            if not S2_RUNTIME_ENABLED:
                return {}

            wait = S2_MIN_INTERVAL - (time.monotonic() - S2_LAST_REQUEST)
            if wait > 0:
                time.sleep(wait)

            # Re-check after sleeping too, because another thread cannot acquire
            # S2_LOCK while we hold it, but this keeps the state transition explicit.
            if not S2_RUNTIME_ENABLED:
                return {}

            try:
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
                S2_LAST_REQUEST = time.monotonic()

                if response.status_code == 200:
                    S2_ERROR_STREAK = 0
                    try:
                        return response.json()
                    except ValueError as exc:
                        S2_ERROR_STREAK += 1
                        print(f"[Semantic Scholar] invalid JSON response: {exc}", flush=True)
                        if S2_AUTO_DISABLE_ON_ERROR and S2_ERROR_STREAK >= S2_FAILURES_BEFORE_DISABLE:
                            S2_RUNTIME_ENABLED = False
                            print("[Semantic Scholar] Disabling source for the remainder of this run after invalid response.", flush=True)
                        return {}

                retryable = response.status_code in {408, 425, 428, 429, 500, 502, 503, 504}
                hard = response.status_code in {401, 403}

                if retryable:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else S2_BACKOFF_BASE * (2 ** attempt)
                    except ValueError:
                        delay = S2_BACKOFF_BASE * (2 ** attempt)
                    delay = min(max(delay, S2_MIN_INTERVAL), S2_BACKOFF_MAX)

                    if attempt < max_retries:
                        print(
                            f"[Semantic Scholar] HTTP {response.status_code}; retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries + 1})", flush=True,
                        )
                        time.sleep(delay)
                        continue

                    # One request has exhausted all its retries. Treat that as one
                    # source failure, and disable immediately by default.
                    S2_ERROR_STREAK += 1
                    print(
                        f"[Semantic Scholar] HTTP {response.status_code}; retries exhausted "
                        f"for this request.", flush=True,
                    )
                    if S2_AUTO_DISABLE_ON_ERROR and S2_ERROR_STREAK >= S2_FAILURES_BEFORE_DISABLE:
                        S2_RUNTIME_ENABLED = False
                        print(
                            "[Semantic Scholar] Disabling source for the remainder of this run "
                            f"after repeated failures (last HTTP {response.status_code}).",
                            flush=True,
                        )
                    return {}

                if hard:
                    S2_ERROR_STREAK += 1
                    S2_RUNTIME_ENABLED = False
                    print(
                        f"[Semantic Scholar] HTTP {response.status_code}; disabling source "
                        "for the remainder of this run.", flush=True,
                    )
                    return {}

                S2_ERROR_STREAK += 1
                if S2_AUTO_DISABLE_ON_ERROR and S2_ERROR_STREAK >= S2_FAILURES_BEFORE_DISABLE:
                    S2_RUNTIME_ENABLED = False
                    print("[Semantic Scholar] Disabling source for the remainder of this run after provider error.", flush=True)
                return {}

            except requests.RequestException as exc:
                if attempt < max_retries:
                    delay = min(max(S2_BACKOFF_BASE * (2 ** attempt), S2_MIN_INTERVAL), S2_BACKOFF_MAX)
                    print(
                        f"[Semantic Scholar] request error; retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries + 1}): {exc}", flush=True,
                    )
                    time.sleep(delay)
                    continue

                S2_ERROR_STREAK += 1
                print(f"[Semantic Scholar] request failed after retries: {exc}", flush=True)
                if S2_AUTO_DISABLE_ON_ERROR and S2_ERROR_STREAK >= S2_FAILURES_BEFORE_DISABLE:
                    S2_RUNTIME_ENABLED = False
                    print("[Semantic Scholar] Disabling source for the remainder of this run after repeated request errors.", flush=True)
                return {}

    return {}


def s2_search(query: str, limit: int = 100, offset: int = 0) -> list[Paper]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "offset": offset,
        "limit": min(limit, 100),
        "fields": "title,abstract,year,authors,venue,externalIds,url,openAccessPdf,citationCount,paperId",
    }
    headers = dict(HEADERS)
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    data = s2_get_json(url, params)
    output: list[Paper] = []
    for x in data.get("data", []):
        ext = x.get("externalIds") or {}
        oa = x.get("openAccessPdf") or {}
        title = (x.get("title") or "").strip()
        if not title:
            continue
        output.append(
            Paper(
                title=title,
                abstract=x.get("abstract") or "",
                year=x.get("year"),
                authors=", ".join(a.get("name", "") for a in (x.get("authors") or [])),
                venue=x.get("venue") or "",
                doi=ext.get("DOI") or "",
                url=x.get("url") or "",
                pdf_url=oa.get("url") or "",
                source="semantic_scholar",
                citation_count=int(x.get("citationCount") or 0),
                external_id=x.get("paperId") or "",
            )
        )
    return output


def s2_related(paper_id: str, kind: str) -> list[Paper]:
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/{kind}"
    params = {
        "limit": 100,
        "fields": "title,abstract,year,authors,venue,externalIds,url,openAccessPdf,citationCount,paperId",
    }
    headers = dict(HEADERS)
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    data = s2_get_json(url, params)
    out: list[Paper] = []
    for entry in data.get("data", []):
        x = entry.get("citedPaper") or entry.get("citingPaper") or {}
        title = (x.get("title") or "").strip()
        if not title:
            continue
        ext = x.get("externalIds") or {}
        oa = x.get("openAccessPdf") or {}
        out.append(
            Paper(
                title=title,
                abstract=x.get("abstract") or "",
                year=x.get("year"),
                authors=", ".join(a.get("name", "") for a in (x.get("authors") or [])),
                venue=x.get("venue") or "",
                doi=ext.get("DOI") or "",
                url=x.get("url") or "",
                pdf_url=oa.get("url") or "",
                source=f"semantic_scholar_{kind}",
                citation_count=int(x.get("citationCount") or 0),
                external_id=x.get("paperId") or "",
            )
        )
    return out


def _extract_openaire_doi(product: dict[str, Any]) -> str:
    for pid in product.get("pids") or []:
        if (pid.get("scheme") or "").lower() == "doi" and pid.get("value"):
            return str(pid["value"]).strip()
    for inst in product.get("instances") or []:
        for pid in inst.get("alternateIdentifiers") or []:
            if (pid.get("scheme") or "").lower() == "doi" and pid.get("value"):
                return str(pid["value"]).strip()
    return ""


def _extract_openaire_url(product: dict[str, Any]) -> str:
    for inst in product.get("instances") or []:
        for url in inst.get("urls") or []:
            if url:
                return str(url)
    return ""


def _simplify_boolean_search_query(query: str, max_chars: int = 220) -> str:
    """Convert an LLM Boolean query into a safe keyword query for strict APIs."""
    value = re.sub(r"\b(?:AND|OR|NOT)\b", " ", query or "", flags=re.IGNORECASE)
    value = value.replace("(", " ").replace(")", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value[:max_chars]


def openaire_search(query: str, page_size: int = OPENAIRE_PAGE_SIZE) -> list[Paper]:
    # OpenAIRE's V3 search supports Boolean expressions, but operands used with
    # AND/OR/NOT need explicit quoting. LLM-generated queries frequently omit
    # those quotes. A simplified keyword query is safer and remains useful as
    # OpenAIRE is only one member of the scholarly source pool.
    safe_query = _simplify_boolean_search_query(query)
    url = "https://api.openaire.eu/graph/v3/research-products"
    params = {
        "search": safe_query,
        "type": "publication",
        "page": 1,
        "pageSize": min(page_size, 100),
        "sortBy": "relevance DESC",
    }
    data = safe_get_json(url, params, HEADERS, timeout=45, label="OpenAIRE")
    out: list[Paper] = []
    for x in data.get("results", []):
        title = (x.get("mainTitle") or "").strip()
        if not title:
            continue
        authors = ", ".join(
            a.get("fullName") or " ".join(v for v in [a.get("name"), a.get("surname")] if v)
            for a in (x.get("authors") or [])
        )
        year = None
        date = x.get("publicationDate") or ""
        m = re.search(r"(\d{4})", str(date))
        if m:
            year = int(m.group(1))
        indicators = x.get("indicators") or {}
        citation_count = int((indicators.get("citationImpact") or {}).get("citationCount") or 0)
        doi = _extract_openaire_doi(x)
        pdf_url = ""
        url_value = _extract_openaire_url(x)
        for inst in x.get("instances") or []:
            urls = inst.get("urls") or []
            if urls and inst.get("accessRight", {}).get("label") == "OPEN":
                pdf_url = str(urls[0])
                break
        descriptions = x.get("descriptions") or []
        abstract = str(descriptions[0]) if descriptions else ""
        venue = ((x.get("container") or {}).get("name") or "")
        out.append(
            Paper(
                title=title,
                abstract=abstract,
                year=year,
                authors=authors,
                venue=venue,
                doi=doi,
                url=url_value,
                pdf_url=pdf_url,
                source="openaire",
                citation_count=citation_count,
                external_id=x.get("id") or "",
            )
        )
    return out


def google_scholar_search(query: str, limit: int = 20) -> list[Paper]:
    """Google Scholar adapter via SerpApi; disabled unless an API key is configured."""
    if not GOOGLE_SCHOLAR_RUNTIME_ENABLED:
        return []
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": min(limit, 20),
    }
    data = safe_get_json(
        "https://serpapi.com/search.json",
        params,
        HEADERS,
        timeout=60,
        label="Google Scholar",
    )
    out: list[Paper] = []
    for x in data.get("organic_results", []):
        title = (x.get("title") or "").strip()
        if not title:
            continue
        publication_info = x.get("publication_info") or {}
        summary = publication_info.get("summary") or ""
        authors = ", ".join((a.get("name") or "") for a in publication_info.get("authors", []) if a.get("name"))
        resources = x.get("resources") or []
        pdf_url = resources[0].get("link") if resources else ""
        snippet = x.get("snippet") or ""
        out.append(
            Paper(
                title=title,
                abstract=snippet,
                authors=authors,
                venue=summary,
                url=x.get("link") or "",
                pdf_url=pdf_url or "",
                source="google_scholar",
                citation_count=int(((x.get("inline_links") or {}).get("cited_by") or {}).get("total") or 0),
                external_id=str(x.get("result_id") or ""),
            )
        )
    return out


def openalex_search(query: str, per_page: int = 100, pages: int = OPENALEX_PAGES) -> list[Paper]:
    if not OPENALEX_RUNTIME_ENABLED:
        return []
    url = "https://api.openalex.org/works"
    out: list[Paper] = []
    for page in range(1, pages + 1):
        params = {
            "search": query,
            "per-page": min(per_page, 100),
            "page": page,
            "select": "id,title,publication_year,authorships,primary_location,doi,open_access,cited_by_count,abstract_inverted_index",
        }
        if OPENALEX_MAILTO:
            params["mailto"] = OPENALEX_MAILTO
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY
        data = safe_get_json(url, params, HEADERS, label="OpenAlex")
        for x in data.get("results", []):
            title = (x.get("title") or "").strip()
            if not title:
                continue
            abstract = reconstruct_openalex_abstract(x.get("abstract_inverted_index"))
            authors = ", ".join(
                (a.get("author") or {}).get("display_name", "")
                for a in (x.get("authorships") or [])
            )
            primary = x.get("primary_location") or {}
            source = primary.get("source") or {}
            doi = (x.get("doi") or "").replace("https://doi.org/", "")
            open_access = x.get("open_access") or {}
            pdf_url = ""
            if open_access.get("is_oa") and primary.get("pdf_url"):
                pdf_url = primary.get("pdf_url")
            out.append(
                Paper(
                    title=title,
                    abstract=abstract,
                    year=x.get("publication_year"),
                    authors=authors,
                    venue=source.get("display_name", "") if isinstance(source, dict) else "",
                    doi=doi,
                    url=x.get("id") or "",
                    pdf_url=pdf_url,
                    source="openalex",
                    citation_count=int(x.get("cited_by_count") or 0),
                    external_id=x.get("id") or "",
                )
            )
        if len(data.get("results", [])) < per_page:
            break
        time.sleep(0.2)
    return out


def reconstruct_openalex_abstract(index: Optional[dict[str, list[int]]]) -> str:
    if not index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in index.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions))


def crossref_search(query: str, rows: int = 100) -> list[Paper]:
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": query,
        "rows": min(rows, 100),
        "select": "DOI,title,abstract,published,author,container-title,URL,type",
    }
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO
    data = safe_get_json(url, params, HEADERS, label="Crossref")
    out: list[Paper] = []
    for x in data.get("message", {}).get("items", []):
        titles = x.get("title") or []
        title = (titles[0] if titles else "").strip()
        if not title:
            continue
        abstract = re.sub(r"<[^>]+>", " ", x.get("abstract") or "").strip()
        date_parts = (x.get("published") or {}).get("date-parts") or [[]]
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        authors = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip() for a in x.get("author", [])
        )
        doi = x.get("DOI") or ""
        out.append(
            Paper(
                title=title,
                abstract=abstract,
                year=year,
                authors=authors,
                venue=(x.get("container-title") or [""])[0],
                doi=doi,
                url=x.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                source="crossref",
                external_id=doi,
            )
        )
    return out


def dblp_search(query: str, limit: int = DBLP_PAGE_SIZE) -> list[Paper]:
    if not DBLP_RUNTIME_ENABLED:
        return []

    params = {"q": query, "format": "json", "h": min(limit, 100), "f": 0}
    raw = safe_get_text(
        "https://dblp.org/search/publ/api",
        params,
        HEADERS,
        timeout=45,
        label="DBLP",
    )
    if not raw:
        return []

    hits: list[dict[str, Any]] = []

    # Prefer the documented JSON format.
    try:
        data = json.loads(raw)
        json_hits = ((data.get("result") or {}).get("hits") or {}).get("hit") or []
        if isinstance(json_hits, dict):
            json_hits = [json_hits]
        hits = [h for h in json_hits if isinstance(h, dict)]
    except (TypeError, ValueError):
        # Some responses/proxies can still return XML despite format=json.
        try:
            root = ET.fromstring(raw)
            for hit_el in root.findall(".//hit"):
                info_el = hit_el.find("info")
                if info_el is None:
                    continue
                info: dict[str, Any] = {}
                for child in info_el:
                    tag = child.tag.rsplit("}", 1)[-1]
                    if tag == "authors":
                        authors = []
                        for a in child.findall("author"):
                            text = (a.text or "").strip()
                            if text:
                                authors.append({"text": text})
                        info["authors"] = {"author": authors}
                    elif tag == "ee":
                        info.setdefault("ee", []).append((child.text or "").strip())
                    else:
                        info[tag] = (child.text or "").strip()
                hits.append({"info": info})
        except ET.ParseError as exc:
            _record_source_failure("dblp", f"response parse error: {exc}")
            print(f"[DBLP] response could not be parsed; skipping this query: {exc}", flush=True)
            return []

    out: list[Paper] = []
    for hit in hits:
        info = hit.get("info") or {}
        title = re.sub(r"\s+", " ", str(info.get("title") or "")).strip()
        if not title:
            continue
        authors_raw = info.get("authors") or {}
        authors_items = authors_raw.get("author") if isinstance(authors_raw, dict) else authors_raw
        if isinstance(authors_items, dict):
            authors_items = [authors_items]
        authors = ", ".join(
            a.get("text") if isinstance(a, dict) else str(a) for a in (authors_items or [])
        )
        year = None
        try:
            year = int(info.get("year")) if info.get("year") else None
        except (TypeError, ValueError):
            pass
        doi = ""
        ee_values = info.get("ee") or []
        if isinstance(ee_values, str):
            ee_values = [ee_values]
        for ee in ee_values:
            if isinstance(ee, str) and "doi.org/" in ee.lower():
                doi = ee.split("doi.org/", 1)[1].split("?", 1)[0]
                break
        venue = str(info.get("venue") or "")
        out.append(
            Paper(
                title=title,
                year=year,
                authors=authors,
                venue=venue,
                doi=doi,
                url=str(info.get("url") or ""),
                source="dblp",
                external_id=str(info.get("key") or ""),
            )
        )
    _record_source_success("dblp")
    return out


def arxiv_search(query: str, limit: int = ARXIV_PAGE_SIZE) -> list[Paper]:
    if not ARXIV_RUNTIME_ENABLED:
        return []
    # Keep search syntax simple and robust; Qwen-generated academic queries are
    # typically ordinary keyword phrases and arXiv's `all:` field covers title/abstract.
    clean = re.sub(r"[{}]", "", query).strip()
    search_query = f"all:{clean}"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": min(limit, 100),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    data = safe_get_text("http://export.arxiv.org/api/query", params, HEADERS, timeout=60, label="arXiv")
    if not data:
        return []
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        _record_source_failure("arxiv", f"XML parse error: {exc}")
        return []
    ns = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}
    out: list[Paper] = []
    for entry in root.findall("a:entry", ns):
        title = re.sub(r"\s+", " ", entry.findtext("a:title", default="", namespaces=ns)).strip()
        if not title:
            continue
        abstract = re.sub(r"\s+", " ", entry.findtext("a:summary", default="", namespaces=ns)).strip()
        authors = ", ".join(
            (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("a:author", ns)
        )
        entry_id = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        pdf_url = ""
        for link in entry.findall("a:link", ns):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
        published = entry.findtext("a:published", default="", namespaces=ns) or ""
        m = re.search(r"(\d{4})", published)
        year = int(m.group(1)) if m else None
        cats = [c.attrib.get("term", "") for c in entry.findall("a:category", ns)]
        out.append(Paper(title=title, abstract=abstract, year=year, authors=authors,
                         venue="arXiv" + (" / " + ", ".join(cats[:3]) if cats else ""),
                         url=entry_id, pdf_url=pdf_url, source="arxiv",
                         external_id=entry_id.rsplit("/", 1)[-1]))
    return out


def safe_get_text(url: str, params: dict[str, Any], headers: Optional[dict[str, str]] = None,
                  timeout: int = 60, label: str = "HTTP") -> str:
    """Same slow/self-disabling policy as safe_get_json, but returns raw text."""
    global API_LAST_REQUEST, DBLP_LAST_REQUEST, ARXIV_LAST_REQUEST, GITHUB_LAST_REQUEST
    source, lock, min_interval, max_retries, backoff_max = _source_state(label)
    last_error: Optional[Exception] = None
    with lock:
        for attempt in range(1, max_retries + 2):
            runtime = {
                "dblp": DBLP_RUNTIME_ENABLED, "arxiv": ARXIV_RUNTIME_ENABLED, "github": GITHUB_RUNTIME_ENABLED
            }.get(source, True)
            if not runtime:
                return ""
            last_attr = {"dblp": "DBLP_LAST_REQUEST", "arxiv": "ARXIV_LAST_REQUEST", "github": "GITHUB_LAST_REQUEST"}.get(source)
            last_time = globals().get(last_attr, API_LAST_REQUEST)
            wait = min_interval - (time.monotonic() - last_time)
            if wait > 0:
                time.sleep(wait)
            try:
                r = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
                now = time.monotonic()
                if last_attr:
                    globals()[last_attr] = now
                else:
                    API_LAST_REQUEST = now
                if r.status_code == 200:
                    _record_source_success(source)
                    return r.text
                retryable = r.status_code in {408, 425, 428, 429, 500, 502, 503, 504}
                hard = r.status_code in {401, 403}
                if retryable:
                    retry_after = r.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else min(backoff_max, max(min_interval, API_BACKOFF_BASE * (2 ** (attempt - 1))))
                    delay = min(max(delay, min_interval), backoff_max)
                    print(f"[{label}] HTTP {r.status_code}; retrying in {delay:.1f}s (attempt {attempt}/{max_retries + 1})", flush=True)
                    last_error = requests.HTTPError(f"HTTP {r.status_code} for {url}", response=r)
                    if attempt <= max_retries:
                        time.sleep(delay)
                        continue
                    _record_source_failure(source, f"HTTP {r.status_code}")
                    break
                if hard:
                    last_error = requests.HTTPError(f"HTTP {r.status_code} for {url}", response=r)
                    _record_source_failure(source, f"HTTP {r.status_code}")
                    break
                r.raise_for_status()
            except requests.RequestException as exc:
                last_error = exc
                if attempt > max_retries:
                    _record_source_failure(source, str(exc))
                    break
                delay = min(backoff_max, max(min_interval, API_BACKOFF_BASE * (2 ** (attempt - 1))))
                print(f"[{label}] request error; retrying in {delay:.1f}s (attempt {attempt}/{max_retries + 1}): {exc}", flush=True)
                time.sleep(delay)
    return ""


def github_search(query: str, limit: int = GITHUB_MAX_RESULTS) -> list[TechnicalSource]:
    if not GITHUB_RUNTIME_ENABLED:
        return []
    headers = dict(HEADERS)
    # Repository search is less tolerant of complex Boolean expressions than
    # the LLM may generate. Use a simple keyword query for the implementation
    # discovery channel; academic sources retain the richer Boolean vocabulary.
    query = _simplify_boolean_search_query(query)
    headers["Accept"] = "application/vnd.github+json"
    headers["X-GitHub-Api-Version"] = "2026-03-10"
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    data = safe_get_json("https://api.github.com/search/repositories",
                         {"q": query, "per_page": min(limit, 30), "sort": "stars", "order": "desc"},
                         headers, timeout=45, label="GitHub")
    out: list[TechnicalSource] = []
    for repo in data.get("items", []):
        out.append(TechnicalSource(
            title=str(repo.get("full_name") or repo.get("name") or ""),
            url=str(repo.get("html_url") or ""),
            source="github",
            snippet=str(repo.get("description") or ""),
            query=query,
            metadata={
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "language": repo.get("language", ""),
                "topics": repo.get("topics") or [],
                "updated_at": repo.get("updated_at", ""),
            },
        ))
    return out


def reddit_search(query: str, limit: int = REDDIT_MAX_RESULTS) -> list[TechnicalSource]:
    if not REDDIT_RUNTIME_ENABLED or DDGS is None:
        return []
    global REDDIT_LAST_REQUEST
    with REDDIT_LOCK:
        wait = REDDIT_MIN_INTERVAL - (time.monotonic() - REDDIT_LAST_REQUEST)
        if wait > 0:
            time.sleep(wait)
        try:
            results = ddgs_search(f"site:reddit.com {query}", max_results=limit)
            REDDIT_LAST_REQUEST = time.monotonic()
            return [TechnicalSource(title=r.get("title", ""), url=r.get("url", ""), source="reddit",
                                     snippet=r.get("snippet", ""), query=query)
                    for r in results if r.get("url")]
        except Exception as exc:
            REDDIT_LAST_REQUEST = time.monotonic()
            print(f"[Reddit] search error: {exc}", flush=True)
            return []


def ddgs_search(query: str, max_results: int | None = None) -> list[dict[str, str]]:
    if max_results is None:
        max_results = MAX_WEB_RESULTS
    if DDGS is None:
        return []
    try:
        with DDGS(timeout=30) as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return [
                {
                    "title": str(x.get("title", "")),
                    "url": str(x.get("href", "")),
                    "snippet": str(x.get("body", "")),
                }
                for x in results
            ]
    except Exception as exc:
        return [{"title": "SEARCH ERROR", "url": "", "snippet": str(exc)}]


def youtube_search(query: str, max_results: int = 25) -> list[dict[str, str]]:
    if not YOUTUBE_API_KEY:
        return [
            {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
            for r in ddgs_search(f"site:youtube.com {query}", max_results=max_results)
        ]
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, 50),
        "order": "relevance",
        "key": YOUTUBE_API_KEY,
    }
    data = safe_get_json(url, params, timeout=30, label="YouTube")
    out: list[dict[str, str]] = []
    for item in data.get("items", []):
        vid = (item.get("id") or {}).get("videoId")
        sn = item.get("snippet") or {}
        if not vid:
            continue
        out.append(
            {
                "title": sn.get("title", ""),
                "channel": sn.get("channelTitle", ""),
                "published": sn.get("publishedAt", ""),
                "url": f"https://www.youtube.com/watch?v={vid}",
                "snippet": sn.get("description", ""),
            }
        )
    return out


def fetch_page(url: str, max_chars: int | None = None) -> str:
    if not url or not url.startswith(("http://", "https://")):
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_links=True,
                include_tables=True,
                favor_precision=True,
            )
            if text:
                return text[:max_chars]
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return re.sub(r"\s+", " ", r.text)[:max_chars]
    except Exception as exc:
        return f"PAGE FETCH ERROR: {exc}"


def fetch_pdf_text(url: str, max_chars: int | None = None) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        doc = fitz.open(stream=r.content, filetype="pdf")
        chunks = []
        total = 0
        for page in doc:
            txt = page.get_text("text")
            chunks.append(txt)
            total += len(txt)
            if total >= max_chars:
                break
        return "\n".join(chunks)[:max_chars]
    except Exception as exc:
        return f"PDF FETCH ERROR: {exc}"


# -----------------------------------------------------------------------------
# LangChain tools for browsing agents
# -----------------------------------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the public web with DuckDuckGo and return titles, URLs and snippets."""
    results = ddgs_search(query, max_results=MAX_WEB_RESULTS)
    return json.dumps(results, ensure_ascii=False)[:60000]


@tool
def open_web_page(url: str) -> str:
    """Fetch and extract readable text from a public web page."""
    return fetch_page(url)


@tool
def open_pdf(url: str) -> str:
    """Fetch and extract text from a public PDF URL."""
    return fetch_pdf_text(url)


BROWSER_TOOLS = [search_web, open_web_page, open_pdf]


# -----------------------------------------------------------------------------
# Planning
# -----------------------------------------------------------------------------

PLANNER_PROMPT = """
You are the lead scientist planning a deep literature review.

Create 6-10 independent research tasks specifically derived from the user's research question.
First identify the major methodological, theoretical, application, evaluation, limitation,
and emerging-work dimensions that are actually relevant to the question.

Do NOT force a predefined taxonomy or repeat the same generic categories for every topic.
The example dimensions below are optional guidance only; use them only when they materially
help answer the specific question:
- methods/theory
- competing approaches
- applications
- disturbances/failure modes
- experimental validation
- sensing/estimation
- limitations/reproducibility
- recent/emerging work
- adjacent fields

For some questions, several of these dimensions may be irrelevant, while other dimensions
may be much more important. Add question-specific tasks when needed.

Do not create duplicate or merely overlapping tasks.
Make each task precise enough that different researchers can work independently.
Prefer complementary tasks that collectively cover the user's question with minimal redundancy.
"""



async def plan_research(question: str) -> ResearchPlan:
    prompt = f"""
{PLANNER_PROMPT}

User research question:
{question}

Use the current date as the review cutoff. State a useful date scope in the plan.
"""
    return await invoke_with_progress(
        planner_llm,
        prompt,
        "QwQ research planner",
    )


async def make_queries(task: ResearchTask, question: str, blind: bool = False) -> QuerySet:
    mode = "BLIND SECOND-PASS" if blind else "FIRST-PASS"
    prompt = f"""
You are a search strategist running a {mode} literature search.

Main question:
{question}

Task name:
{task.name}

Task objective:
{task.objective}

Scope notes:
{task.scope_notes}

Concepts:
{task.search_concepts}

Generate high-recall, diverse search strategies. Include:
- exact method names
- abbreviations
- older terminology
- alternative terminology
- application-specific wording
- failure-mode wording
- sensor/actuator wording
- neighboring disciplines
- historical terms
- experimental terms
- review/survey terms
- benchmark terms
- citation-chasing ideas
- web/technical-document terms
- YouTube search terms when useful

Prefer clear natural-language queries. Use Boolean operators only when their grouping is explicit with parentheses.
Do not create queries that are so broad that they mostly retrieve irrelevant records.

The blind pass must deliberately search for concepts that may not appear in the first-pass vocabulary.
Return no more than {SEARCH_QUERY_COUNT} academic queries and no more than {WEB_QUERY_COUNT} web queries.
Prefer exactly those counts when enough distinct queries are available.
Return [] for any field that is not useful for this task rather than omitting the field.
Keep queries diverse and task-specific; do not pad with repetitive variations.

The structured result contains these fields:
- academic_queries
- web_queries
- youtube_queries
- alternate_terminology
- source_types_to_seek

Populate every field. Use an empty list when a field is not applicable.
"""
    result = await invoke_with_progress(
        search_strategist_llm,
        prompt,
        f"Qwen3.5 search strategy: {task.name}",
    )

    # Normalize list fields so imperfect local-model output cannot propagate
    # malformed values into the harvesting stage.
    return QuerySet(
        academic_queries=[str(x).strip() for x in (result.academic_queries or []) if str(x).strip()],
        web_queries=[str(x).strip() for x in (result.web_queries or []) if str(x).strip()],
        youtube_queries=[str(x).strip() for x in (result.youtube_queries or []) if str(x).strip()],
        alternate_terminology=[str(x).strip() for x in (result.alternate_terminology or []) if str(x).strip()],
        source_types_to_seek=[str(x).strip() for x in (result.source_types_to_seek or []) if str(x).strip()],
    )


async def decide_sources(task: ResearchTask, question: str) -> SourceDecision:
    prompt = f"""
Decide which non-journal sources are worth searching for this research task.

Question: {question}
Task: {task.name}
Objective: {task.objective}

Think about official documentation, standards, university/lab pages, project sites,
manufacturer technical pages, conference pages, repositories, datasets, technical reports,
video lectures/talks/demonstrations, and other specialist sources.

OUTPUT CONSTRAINTS:
- use_youtube: true or false
- website_types: at most 8 concise categories
- suggested_domains: at most 6 normal hostnames such as github.com or ieee.org
- rationale: one or two concise sentences
- Do not suggest piracy, credential abuse, illicit-access or disallowed sources.

Do not treat these as equivalent to peer-reviewed papers. They supplement evidence.
"""
    try:
        decision = await invoke_with_progress(
            source_decider_llm,
            prompt,
            f"Qwen3.5 source strategy: {task.name}",
            retries=0,
        )
    except Exception as exc:
        print(
            f"[SOURCE DECISION] {task.name} failed; using safe default web-source policy: {exc}",
            flush=True,
        )
        return SourceDecision(
            use_youtube=False,
            website_types=["official documentation", "technical reports", "repositories", "conference pages"],
            suggested_domains=["github.com", "arxiv.org", "ieee.org", "acm.org"],
            rationale="Fallback source policy used because the local source-selection response could not be parsed.",
        )

    # Sanitize model output before it affects web queries.
    domains = []
    for domain in decision.suggested_domains[:6]:
        d = re.sub(r"^https?://", "", str(domain).strip(), flags=re.IGNORECASE)
        d = d.split("/", 1)[0].strip().lower()
        if re.fullmatch(r"[a-z0-9][a-z0-9.-]*\.[a-z]{2,}", d) and d not in domains:
            domains.append(d)
    website_types = []
    for item in decision.website_types[:8]:
        value = re.sub(r"\s+", " ", str(item).strip())
        if value and value not in website_types:
            website_types.append(value)
    return SourceDecision(
        use_youtube=bool(decision.use_youtube),
        website_types=website_types,
        suggested_domains=domains,
        rationale=str(decision.rationale or "").strip()[:600],
    )


# -----------------------------------------------------------------------------
# Research harvesting
# -----------------------------------------------------------------------------

def relevance_score(paper: Paper, task: ResearchTask) -> float:
    haystack = f"{paper.title} {paper.abstract}".lower()
    concepts = [c.lower() for c in task.search_concepts] + [task.name.lower(), task.objective.lower()]
    hits = sum(1 for c in concepts if c and c in haystack)
    concept_score = min(hits / max(1, len(concepts)), 1.0)
    citation_score = min(paper.citation_count / 500.0, 1.0)
    recency_score = 0.0
    if paper.year:
        recency_score = max(0, min((paper.year - 2015) / 12.0, 1.0))
    return 0.55 * concept_score + 0.25 * citation_score + 0.20 * recency_score


def rerank_task_fast(db: PaperDB, task: ResearchTask) -> None:
    for p in db.all_for_task(task.name):
        db.update_rerank(p.paper_id, relevance_score(p, task))


def slug(text: str, max_len: int = 80) -> str:
    """Return a filesystem-safe, compact slug for run artifact filenames."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    value = value.strip("._-")
    return (value[:max_len] or "task")


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


async def harvest_one_query(query: str, task_name: str = "", pass_name: str = "") -> list[Paper]:
    """Harvest scholarly results plus optional technical/community sources.

    GitHub and Reddit are stored separately and never count toward the paper floor.
    """

    merged: list[Paper] = []

    if S2_RUNTIME_ENABLED:
        out: list[Paper] = []
        for page in range(MAX_SEARCH_PAGES):
            try:
                batch = await asyncio.to_thread(s2_search, query, 100, page * 100)
                out.extend(batch)
                if len(batch) < 100:
                    break
            except Exception:
                break
        merged.extend(out)

    if OPENALEX_RUNTIME_ENABLED:
        try:
            merged.extend(
                await asyncio.to_thread(openalex_search, query, 100, OPENALEX_PAGES)
            )
        except Exception as exc:
            print(f"[OpenAlex] query failed; continuing without this source: {exc}", flush=True)

    if OPENAIRE_RUNTIME_ENABLED:
        try:
            merged.extend(await asyncio.to_thread(openaire_search, query, OPENAIRE_PAGE_SIZE))
        except Exception as exc:
            print(f"[OpenAIRE] query failed; continuing without this source: {exc}", flush=True)

    if GOOGLE_SCHOLAR_RUNTIME_ENABLED:
        try:
            merged.extend(await asyncio.to_thread(google_scholar_search, query, 20))
        except Exception as exc:
            print(f"[Google Scholar] query failed; continuing without this source: {exc}", flush=True)

    if DBLP_RUNTIME_ENABLED:
        try:
            merged.extend(await asyncio.to_thread(dblp_search, query, DBLP_PAGE_SIZE))
        except Exception as exc:
            print(f"[DBLP] query failed; continuing without this source: {exc}", flush=True)

    if ARXIV_RUNTIME_ENABLED:
        try:
            merged.extend(await asyncio.to_thread(arxiv_search, query, ARXIV_PAGE_SIZE))
        except Exception as exc:
            print(f"[arXiv] query failed; continuing without this source: {exc}", flush=True)

    try:
        merged.extend(await asyncio.to_thread(crossref_search, query, 100))
    except Exception as exc:
        print(f"[Crossref] query failed after retries: {exc}", flush=True)

    if task_name:
        tech_sources: list[TechnicalSource] = []
        if GITHUB_RUNTIME_ENABLED:
            tech_sources.extend(await asyncio.to_thread(github_search, query, GITHUB_MAX_RESULTS))
        if REDDIT_RUNTIME_ENABLED:
            tech_sources.extend(await asyncio.to_thread(reddit_search, query, REDDIT_MAX_RESULTS))
        for source in tech_sources:
            TECH_SOURCE_CACHE.setdefault((task_name, pass_name), []).append(source)

    return merged


async def citation_expand(db: PaperDB, task: ResearchTask) -> None:
    hubs = db.top(task.name, MAX_CITATION_HUBS)
    async def expand(p: Paper):
        # Only Semantic Scholar records can be expanded through this endpoint.
        if not p.external_id or not p.source.startswith("semantic_scholar"):
            return
        try:
            for kind in ("references", "citations"):
                related = await asyncio.to_thread(s2_related, p.external_id, kind)
                for rel in related:
                    db.add(task.name, rel, relevance_score(rel, task), "citation_chasing")
        except Exception:
            return
    await asyncio.gather(*(expand(h) for h in hubs), return_exceptions=True)


async def harvest_task(db: PaperDB, task: ResearchTask, question: str, pass_name: str, blind: bool = False) -> dict[str, Any]:
    qs = await make_queries(task, question, blind=blind)
    # Force the requested volume of queries even if the model over/under produces.
    academic_queries = list(dict.fromkeys(qs.academic_queries))[:SEARCH_QUERY_COUNT]
    web_queries = list(dict.fromkeys(qs.web_queries))[:WEB_QUERY_COUNT]
    youtube_queries = list(dict.fromkeys(qs.youtube_queries))[:10]

    save_json(RUN_DIR / f"queries_{slug(task.name)}_{pass_name}.json", qs.model_dump())

    inserted = 0
    # Search queries in small groups. This avoids overwhelming APIs while still
    # allowing task-level parallelism.
    for start in range(0, len(academic_queries), 5):
        batch_queries = academic_queries[start:start + 5]
        print(
            f"[SEARCH] {task.name} | queries {start + 1}-{start + len(batch_queries)} "
            f"of {len(academic_queries)} | current candidates={db.count(task.name)}",
            flush=True,
        )
        results = await asyncio.gather(*(harvest_one_query(q, task.name, pass_name) for q in batch_queries), return_exceptions=True)
        for batch in results:
            if isinstance(batch, list):
                for paper in batch:
                    inserted += int(db.add(task.name, paper, relevance_score(paper, task), pass_name))
        rerank_task_fast(db, task)
        if db.count(task.name) >= DISCOVERY_TARGET:
            break

    # Citation chaining widens the graph and often finds older/less obvious papers.
    await citation_expand(db, task)
    rerank_task_fast(db, task)

    # If the requested floor is not reached, continue with alternate queries.
    expansion_round = 0
    while db.count(task.name) < MIN_CANDIDATE_PAPERS and expansion_round < 4:
        expansion_round += 1
        print(
            f"[SEARCH] {task.name} | minimum not reached "
            f"({db.count(task.name)}/{MIN_CANDIDATE_PAPERS}); expansion round {expansion_round}",
            flush=True,
        )
        extra = await make_queries(task, question, blind=blind)
        extra_queries = list(dict.fromkeys(extra.academic_queries))[:SEARCH_QUERY_COUNT]
        for q in extra_queries:
            try:
                papers = await harvest_one_query(q, task.name, pass_name)
                for paper in papers:
                    db.add(task.name, paper, relevance_score(paper, task), f"{pass_name}_exp{expansion_round}")
            except Exception:
                pass
            if db.count(task.name) >= MIN_CANDIDATE_PAPERS:
                break
        rerank_task_fast(db, task)

    source_decision = await decide_sources(task, question)
    web_hits = []
    if source_decision.website_types:
        domain_terms = " OR ".join(f"site:{d}" for d in source_decision.suggested_domains[:6])
    else:
        domain_terms = ""
    for q in web_queries:
        augmented = f"{q} {domain_terms}".strip()
        results = await asyncio.to_thread(ddgs_search, augmented, MAX_WEB_RESULTS)
        web_hits.append({"query": augmented, "results": results})
    save_json(RUN_DIR / f"web_{slug(task.name)}_{pass_name}.json", web_hits)

    yt_hits: list[dict[str, str]] = []
    if ENABLE_YOUTUBE and source_decision.use_youtube:
        for q in youtube_queries:
            try:
                yt_hits.extend(await asyncio.to_thread(youtube_search, q, 15))
            except Exception:
                pass
        save_json(RUN_DIR / f"youtube_{slug(task.name)}_{pass_name}.json", yt_hits)

    tech_key = (task.name, pass_name)
    tech_sources = TECH_SOURCE_CACHE.pop(tech_key, [])
    for ts in tech_sources:
        db.add_technical_source(task.name, ts, pass_name)
    save_json(RUN_DIR / f"technical_sources_{slug(task.name)}_{pass_name}.json",
              db.technical_sources(task.name))

    counts = {
        "task": task.name,
        "pass": pass_name,
        "candidate_count": db.count(task.name),
        "technical_source_count": db.technical_count(task.name),
        "requested_minimum": MIN_CANDIDATE_PAPERS,
        "source_counts": db.source_counts(task.name),
        "inserted_this_pass": inserted,
        "youtube_results": len(yt_hits),
        "source_decision": source_decision.model_dump(),
    }
    save_json(RUN_DIR / f"stats_{slug(task.name)}_{pass_name}.json", counts)
    return counts


# -----------------------------------------------------------------------------
# Deep paper analysis
# -----------------------------------------------------------------------------

async def analyse_paper(paper: Paper, task: ResearchTask, question: str, reference_key: str) -> Optional[EvidenceCard]:
    content = paper.abstract
    if paper.pdf_url:
        extracted = await asyncio.to_thread(fetch_pdf_text, paper.pdf_url)
        if extracted and not extracted.startswith("PDF FETCH ERROR"):
            content = extracted
    if len(content) < 1000 and paper.url:
        page_text = await asyncio.to_thread(fetch_page, paper.url)
        if page_text and not page_text.startswith("PAGE FETCH ERROR"):
            content = page_text

    prompt = f"""
You are a meticulous scientific-paper analyst.

Research question: {question}
Task: {task.name}
Task objective: {task.objective}

Reference key: {reference_key}

Metadata:
Title: {paper.title}
Year: {paper.year}
Authors: {paper.authors}
Venue: {paper.venue}
DOI: {paper.doi}
URL: {paper.url}

Paper text / abstract:
{content[:PDF_MAX_CHARS]}

Extract only information supported by the supplied content.
Do not invent numerical results, experiments, equations or claims.
Explain the control architecture or method in concrete terms.
"""
    try:
        async with LLM_SEM:
            return await invoke_with_progress(
                evidence_llm,
                prompt,
                f"Qwen3.5 paper analysis: {reference_key}",
            )
    except Exception as exc:
        print(f"[PAPER] {reference_key} failed: {exc}", flush=True)
        return None


async def deep_read_task(db: PaperDB, task: ResearchTask, question: str) -> list[EvidenceCard]:
    papers = db.top(task.name, DEEP_READ_TOP)
    cards: list[EvidenceCard] = []
    sem = asyncio.Semaphore(MAX_LLM_CONCURRENCY)

    async def one(idx: int, paper: Paper):
        reference_key = f"{slug(task.name)[:12].upper()}-{idx:03d}"
        async with sem:
            return await analyse_paper(paper, task, question, reference_key)

    results = await asyncio.gather(*(one(i + 1, p) for i, p in enumerate(papers)), return_exceptions=True)
    for card in results:
        if isinstance(card, EvidenceCard):
            cards.append(card)
    return cards


async def build_evidence_corpus(db: PaperDB, plan: ResearchPlan, question: str) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {}
    for task in plan.tasks:
        cards = await deep_read_task(db, task, question)
        evidence[task.name] = [c.model_dump() for c in cards]
    save_json(RUN_DIR / "evidence_cards.json", evidence)
    return evidence


# -----------------------------------------------------------------------------
# Research agents and hierarchical synthesis
# -----------------------------------------------------------------------------

async def web_research_agent(task: ResearchTask, question: str, queries: list[str], model_name: str = RESEARCH_MODEL) -> str:
    agent = create_agent(
        model=make_llm(model_name, 0, "web"),
        tools=BROWSER_TOOLS,
        system_prompt=(
            "You are a web evidence researcher. Use search_web first, then open relevant pages. "
            "Prefer primary/official sources. Do not fabricate citations. Return concise evidence with URLs."
        ),
    )
    prompt = f"""
Research task: {task.name}
Objective: {task.objective}
Question: {question}

Search these starting ideas, but add your own searches when useful:
{json.dumps(queries, ensure_ascii=False)}

Find evidence from websites, technical pages, official project pages, standards, repositories,
conference pages, and other credible sources that the academic databases may miss.
"""
    async with LLM_SEM:
        result = await agent_invoke_with_progress(
            agent,
            [("user", prompt)],
            f"Web research: {task.name}",
        )
    return result["messages"][-1].content


def _compact_evidence_cards(cards: list[dict[str, Any]], max_chars: int) -> str:
    """Create a bounded, synthesis-friendly digest of evidence cards.

    The digest keeps the fields most useful for literature synthesis while
    avoiding giant raw-card payloads and incomplete JSON caused by hard slicing.
    """
    compact: list[dict[str, Any]] = []
    used = 2
    for card in cards:
        item = {
            "paper_title": str(card.get("paper_title", "")),
            "year": card.get("year"),
            "authors": str(card.get("authors", ""))[:240],
            "venue": str(card.get("venue", ""))[:180],
            "doi": str(card.get("doi", "")),
            "research_problem": str(card.get("research_problem", ""))[:500],
            "method": str(card.get("method", ""))[:900],
            "platform_or_dataset": str(card.get("platform_or_dataset", ""))[:500],
            "experiment_type": str(card.get("experiment_type", ""))[:400],
            "disturbance_or_environment": str(card.get("disturbance_or_environment", ""))[:700],
            "key_findings": [str(x)[:500] for x in (card.get("key_findings") or [])[:5]],
            "limitations": [str(x)[:400] for x in (card.get("limitations") or [])[:4]],
            "relevance_to_task": str(card.get("relevance_to_task", ""))[:500],
            "evidence_strength": str(card.get("evidence_strength", ""))[:180],
            "important_equations_or_control_structure": [str(x)[:500] for x in (card.get("important_equations_or_control_structure") or [])[:3]],
        }
        encoded = json.dumps(item, ensure_ascii=False)
        extra = len(encoded) + 1
        if compact and used + extra > max_chars:
            break
        if not compact and used + extra > max_chars:
            # Keep at least the first evidence item, but never exceed the bound.
            encoded = encoded[: max(0, max_chars - used - 1)]
            compact.append({"truncated_first_evidence_record": encoded})
            break
        compact.append(item)
        used += extra
    return json.dumps(compact, ensure_ascii=False)


async def _synthesize_one_task(
    question: str,
    task: ResearchTask,
    cards: list[dict[str, Any]],
    position: int,
) -> dict[str, Any]:
    bounded = _compact_evidence_cards(cards, SYNTHESIS_TASK_INPUT_CHARS)
    prompt = f"""
You are a senior literature-review analyst responsible for one research task.

Main question:
{question}

Task {position}: {task.name}
Objective:
{task.objective}

Task scope notes:
{task.scope_notes}

Evidence cards from the deep-reading stage ({len(cards)} cards total; bounded digest below):
{bounded}

Produce a compact but technically rich task-level synthesis.
Cover:
- terminology and problem definition
- major method families and how they differ
- assumptions and system models
- disturbances/environmental effects
- sensing and estimation where relevant
- experimental/simulation evidence
- reported strengths and limitations
- contradictions or mixed findings
- important historical/seminal ideas represented in the supplied evidence
- recent/emerging directions represented in the supplied evidence
- 3-8 evidence-grounded research gaps

Do not invent papers, numbers, results, citations, or conclusions not supported by the supplied cards.
Focus on cross-paper patterns rather than writing one mini-summary per paper.
Return a standalone technical synthesis that another scientist can use as input to a global review.
"""
    model = make_llm(WRITER_MODEL, 0, "task_synthesis")
    async with LLM_SEM:
        result = await invoke_with_progress(
            model,
            prompt,
            f"Qwen3 task synthesis: {task.name}",
        )
    summary = getattr(result, "content", str(result))
    summary = str(summary).strip()
    if len(summary) > SYNTHESIS_TASK_SUMMARY_CHARS:
        summary = summary[:SYNTHESIS_TASK_SUMMARY_CHARS].rstrip() + "\n[task summary truncated]"
    return {
        "task": task.name,
        "objective": task.objective,
        "paper_count": len(cards),
        "summary": summary,
    }


async def build_task_summaries(
    question: str,
    plan: ResearchPlan,
    evidence: dict[str, list[dict[str, Any]]],
    artifact_name: str,
) -> list[dict[str, Any]]:
    """Hierarchically compress paper evidence into one bounded summary per task."""
    sem = asyncio.Semaphore(MAX_LLM_CONCURRENCY)

    async def one(idx: int, task: ResearchTask):
        async with sem:
            return await _synthesize_one_task(
                question,
                task,
                evidence.get(task.name, []),
                idx,
            )

    results = await asyncio.gather(
        *(one(i + 1, task) for i, task in enumerate(plan.tasks)),
        return_exceptions=True,
    )
    summaries: list[dict[str, Any]] = []
    for task, result in zip(plan.tasks, results):
        if isinstance(result, Exception):
            print(f"[SYNTHESIS] {task.name} task-summary failure: {result}", flush=True)
            summaries.append({
                "task": task.name,
                "objective": task.objective,
                "paper_count": len(evidence.get(task.name, [])),
                "summary": "Task-level synthesis failed; use the saved evidence cards for this task.",
            })
        else:
            summaries.append(result)
    save_json(RUN_DIR / artifact_name, summaries)
    return summaries


async def first_synthesis(
    question: str,
    plan: ResearchPlan,
    evidence: dict[str, list[dict[str, Any]]],
) -> str:
    """Create the first synthesis from bounded per-task summaries, not raw cards."""
    print("[SYNTHESIS] Building bounded task-level summaries before global synthesis...", flush=True)
    task_summaries = await build_task_summaries(
        question,
        plan,
        evidence,
        "task_summaries_pre_gap.json",
    )
    bounded = json.dumps(task_summaries, ensure_ascii=False)[:SYNTHESIS_GLOBAL_INPUT_CHARS]
    print(
        f"[SYNTHESIS] Global input: {len(bounded):,} chars | "
        f"context: {LLM_CTX_SYNTHESIS:,} | output cap: {LLM_TOKENS_SYNTHESIS:,} tokens",
        flush=True,
    )

    prompt = f"""
You are the lead scientist creating the first global evidence synthesis.

Question:
{question}

Plan:
{plan.model_dump_json(indent=2)}

Task-level evidence syntheses:
{bounded}

Create a substantial research map, not a shallow summary.
Explain terminology and the evolution of the field.
Compare methods, assumptions, disturbances, sensors, validation, metrics and limitations.
Identify consistent findings, contradictions, uncertainty and unresolved issues across tasks.
Distinguish evidence supported by the supplied task syntheses from your own integrative interpretation.
Do not invent references, numerical results, or claims absent from the supplied material.
The task syntheses are compressed evidence maps; do not pretend every candidate paper was fully read.
"""
    async with LLM_SEM:
        result = await invoke_with_progress(
            make_llm(WRITER_MODEL, 0, "synthesis"),
            prompt,
            "Qwen3 first synthesis",
        )
    text = str(getattr(result, "content", result)).strip()
    (RUN_DIR / "first_synthesis.md").write_text(text, encoding="utf-8")
    return text


async def critique(question: str, plan: ResearchPlan, draft: str) -> Critique:
    prompt = f"""
Act as an adversarial peer reviewer of the following literature synthesis.

Question:
{question}

Plan:
{plan.model_dump_json(indent=2)}

Synthesis:
{draft}

Find:
- unsupported claims
- missing method families
- contradictory findings
- missing seminal work
- missing recent work
- terminology blind spots
- geographic or venue bias
- overreliance on one source type
- places where the synthesis jumps beyond the evidence
- practical questions not answered

For every gap, create targeted search queries grouped by task.
"""
    return await invoke_with_progress(
        critic_llm,
        prompt,
        "QwQ adversarial critique",
    )


async def make_gap_plans(question: str, plan: ResearchPlan, draft: str, critique_result: Critique) -> list[GapPlan]:
    gap_plans: list[GapPlan] = []
    for task in plan.tasks:
        queries = critique_result.additional_queries_by_task.get(task.name, [])
        prompt = f"""
Design an independent second-pass search plan for this task.

Question: {question}
Task: {task.name}
Objective: {task.objective}

First synthesis summary:
{draft[:35000]}

Known critique:
{json.dumps(critique_result.model_dump(), ensure_ascii=False)[:25000]}

Existing suggested queries:
{queries}

The goal is to find evidence the first-pass vocabulary may have missed.
Seek alternate terminology, older terminology, adjacent fields, different application names,
contradictory results, negative results and technical sources.

IMPORTANT OUTPUT CONSTRAINTS:
- Return ONLY the structured GapPlan object.
- task_name MUST exactly equal the task name above.
- source_types: at most 6 concise source categories.
- queries: at most 10 concrete search queries.
- alternate_search_terms: at most 10 terms/phrases.
- overlooked_angles: at most 10 concise angles.
- adjacent_fields: at most 6 fields.
- Do NOT generate hundreds of variants or repeat near-identical phrases.
- Every query should be meaningfully different and suitable for an academic/web search engine.
"""
        try:
            gap = await invoke_with_progress(
                gap_planner_llm,
                prompt,
                f"QwQ gap plan: {task.name}",
            )
        except Exception as exc:
            # A malformed structured response should not destroy an otherwise
            # successful multi-stage research run. Fall back to the critique
            # queries for this task and continue.
            print(
                f"[GAP] {task.name} structured-output failure; using fallback queries: {exc}",
                flush=True,
            )
            gap = GapPlan(
                task_name=task.name,
                source_types=["journal articles", "conference papers", "technical reports", "web sources"],
                queries=list(dict.fromkeys(queries))[:10],
                alternate_search_terms=list(dict.fromkeys(queries))[:10],
            )

        # Normalize omissions/overproduction from local models.
        gap.task_name = task.name
        if not gap.queries:
            gap.queries = list(dict.fromkeys(gap.alternate_search_terms))[:10]
        if not gap.queries:
            gap.queries = list(dict.fromkeys(queries))[:10]
        gap.queries = list(dict.fromkeys(gap.queries))[:10]
        gap.alternate_search_terms = list(dict.fromkeys(gap.alternate_search_terms))[:10]
        gap.overlooked_angles = list(dict.fromkeys(gap.overlooked_angles))[:10]
        gap.adjacent_fields = list(dict.fromkeys(gap.adjacent_fields))[:6]
        gap.source_types = list(dict.fromkeys(gap.source_types))[:6]
        if not gap.source_types:
            gap.source_types = ["journal articles", "conference papers", "technical reports", "web sources"]

        gap_plans.append(gap)
    save_json(RUN_DIR / "gap_plans.json", [x.model_dump() for x in gap_plans])
    return gap_plans


async def targeted_gap_search(question: str, gap_plans: list[GapPlan], db: PaperDB, plan: ResearchPlan, pass_name: str) -> list[dict[str, Any]]:
    task_map = {t.name: t for t in plan.tasks}
    out = []
    for gap in gap_plans:
        task = task_map.get(gap.task_name)
        if not task:
            continue
        search_queries = list(dict.fromkeys(gap.queries + gap.alternate_search_terms))[:BLIND_QUERY_COUNT]
        found = 0
        for q in search_queries:
            try:
                papers = await harvest_one_query(q)
                for paper in papers:
                    found += int(db.add(task.name, paper, relevance_score(paper, task), pass_name))
                web = await asyncio.to_thread(ddgs_search, q, MAX_WEB_RESULTS)
                with (RUN_DIR / f"gap_web_{slug(task.name)}_{pass_name}.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"query": q, "results": web}, ensure_ascii=False) + "\n")
            except Exception:
                pass
        rerank_task_fast(db, task)
        out.append({"task": task.name, "queries": search_queries, "new_papers": found, "count": db.count(task.name)})
    save_json(RUN_DIR / f"{pass_name}_stats.json", out)
    return out


# -----------------------------------------------------------------------------
# Final review and writing
# -----------------------------------------------------------------------------

async def final_research_review(question: str, final_draft: str) -> FinalReview:
    prompt = f"""
Review this proposed final literature review before it is converted into PDF.

Question:
{question}

Draft:
{final_draft}

Check whether the report:
- explains concepts before comparing them
- distinguishes evidence from interpretation
- avoids unsupported claims
- uses consistent terminology
- identifies limitations and counter-evidence
- is clear to a technical reader
- has adequate methodology/search transparency
- does not claim that candidate papers were fully read unless they were
- contains a coherent references section

Be strict. If important issues remain, list them under required_research and set ready_for_pdf=false.
"""
    async with LLM_SEM:
        return await invoke_with_progress(
            final_review_llm,
            prompt,
            "QwQ final scientific review",
        )


def build_reference_catalog(db: PaperDB, plan: ResearchPlan) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    counter = 1
    for task in plan.tasks:
        for paper in db.top(task.name, DEEP_READ_TOP):
            key = (paper.doi or PaperDB.normalize_title(paper.title)).lower()
            if key in seen:
                continue
            seen.add(key)
            refs.append({
                "key": f"R{counter:04d}",
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "venue": paper.venue,
                "doi": paper.doi,
                "url": paper.url,
                "task": task.name,
            })
            counter += 1
    return refs


async def write_final_report(
    question: str,
    plan: ResearchPlan,
    task_summaries: list[dict[str, Any]],
    first_draft: str,
    critique_result: Critique,
    refs: list[dict[str, Any]],
) -> str:
    """Write the final report from bounded task summaries and review evidence."""
    bounded_summaries = json.dumps(task_summaries, ensure_ascii=False)[:SYNTHESIS_FINAL_INPUT_CHARS]
    compact_refs = [
        {
            "key": r.get("key", ""),
            "title": r.get("title", ""),
            "authors": r.get("authors", ""),
            "year": r.get("year", ""),
            "venue": r.get("venue", ""),
            "doi": r.get("doi", ""),
            "url": r.get("url", ""),
        }
        for r in refs
    ]
    bounded_refs = json.dumps(compact_refs, ensure_ascii=False)[:24000]
    bounded_first = first_draft[:12000]
    bounded_critique = critique_result.model_dump_json(indent=2)[:7000]

    prompt = f"""
You are the senior author writing the final deep literature review.

Research question:
{question}

Research plan:
{plan.model_dump_json(indent=2)}

First synthesis:
{bounded_first}

Adversarial critique:
{bounded_critique}

Post-gap task-level evidence syntheses:
{bounded_summaries}

Reference catalog. Use only these reference keys for citations:
{bounded_refs}

Write an extensive, properly explained research report.

Required structure:
1. Executive summary
2. Research scope and methodology
3. Terminology and problem definition
4. Historical development
5. Main technical approaches
6. Detailed evidence from the literature
7. Cross-method comparison
8. Experimental and simulation evidence
9. Disturbances, uncertainty and failure modes
10. Contradictory or mixed findings
11. Research gaps
12. Practical implications for the user's research context
13. Promising future research directions
14. Conclusions
15. References

Explain methods before comparing them. Use tables when useful.
Distinguish established evidence from emerging work and from your own synthesis.
Do not claim to have deeply read papers that were only candidate records.
Do not invent citations. Cite important factual claims using the provided Rxxxx keys, for example [R0007].
Use the post-gap task syntheses as the primary evidence basis for detailed claims.
"""
    async with LLM_SEM:
        result = await invoke_with_progress(
            make_llm(WRITER_MODEL, 0, "writer"),
            prompt,
            "Qwen3 final report writer",
        )
    report = str(getattr(result, "content", result)).strip()
    (RUN_DIR / "pre_pdf_report.md").write_text(report, encoding="utf-8")
    return report


def append_references(report: str, refs: list[dict[str, Any]]) -> str:
    body = report
    if re.search(r"^#+\s*References\s*$", report, flags=re.MULTILINE | re.IGNORECASE):
        body = re.sub(r"^#+\s*References\s*$.*", "", report, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL).rstrip()
    lines = [body, "", "## References", ""]
    cited = set(re.findall(r"\[(R\d{4})\]", report))
    # Include every deeply analysed reference, whether or not the writer happened to cite it.
    for ref in refs:
        authors = ref.get("authors") or ""
        year = ref.get("year") or "n.d."
        venue = ref.get("venue") or ""
        doi = ref.get("doi") or ""
        url = ref.get("url") or ""
        citation = f"**[{ref['key']}]** {authors} ({year}). *{ref['title']}*. {venue}."
        if doi:
            citation += f" DOI: https://doi.org/{doi}."
        elif url:
            citation += f" {url}"
        lines.append(citation)
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Markdown -> PDF
# -----------------------------------------------------------------------------

def pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=19, leading=23, spaceAfter=12))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=14.5, leading=18, spaceBefore=13, spaceAfter=7))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(name="H3x", parent=styles["Heading3"], fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.2, leading=13.4, spaceAfter=6))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.7, leading=9.5))
    styles.add(ParagraphStyle(name="Bulletx", parent=styles["BodyText"], fontSize=9.1, leading=13, leftIndent=13, firstLineIndent=-7, spaceAfter=3))
    return styles


def sanitize_pdf_text(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text


def inline_escape(text: str) -> str:
    text = sanitize_pdf_text(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(.*?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return text


def markdown_to_flowables(md: str):
    styles = pdf_styles()
    story = []
    lines = sanitize_pdf_text(md).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(inline_escape(line[2:]), styles["TitleCenter"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline_escape(line[3:]), styles["H1x"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline_escape(line[4:]), styles["H2x"]))
        elif line.startswith("#### "):
            story.append(Paragraph(inline_escape(line[5:]), styles["H3x"]))
        elif line.startswith("- ") or line.startswith("* "):
            story.append(Paragraph("&#8226; " + inline_escape(line[2:]), styles["Bulletx"]))
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tl in table_lines:
                if re.match(r"^\|\s*:?-{2,}", tl):
                    continue
                cells = [c.strip() for c in tl.strip("|").split("|")]
                rows.append([Paragraph(inline_escape(c), styles["Smallx"]) for c in cells])
            if rows:
                cols = max(len(r) for r in rows)
                for r in rows:
                    while len(r) < cols:
                        r.append(Paragraph("", styles["Smallx"]))
                widths = [doc_width := (180 * mm / cols)] * cols
                tbl = Table(rows, repeatRows=1, hAlign="LEFT", colWidths=widths)
                tbl.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDEDED")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 4))
            continue
        elif re.match(r"^\d+\.\s", line):
            story.append(Paragraph(inline_escape(line), styles["Bodyx"]))
        else:
            buf = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|[-*] |\||\d+\.\s)", lines[i]):
                buf.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline_escape(" ".join(buf)), styles["Bodyx"]))
            continue
        i += 1
    return story


def write_pdf(title: str, markdown: str, out_path: Path) -> None:
    class NumberedDocTemplate(BaseDocTemplate):
        pass

    doc = NumberedDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=title,
        author="Local Deep Research Engine",
        subject="Deep literature research report",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.2)
        canvas.drawString(15 * mm, 8 * mm, "Local Deep Research Engine")
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=footer)])
    doc.build(markdown_to_flowables(markdown))


def verify_pdf(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    doc = fitz.open(path)
    result["pages"] = doc.page_count
    result["bytes"] = path.stat().st_size
    result["text_chars"] = sum(len(p.get_text("text")) for p in doc)
    result["ok"] = doc.page_count > 0 and result["bytes"] > 1000 and result["text_chars"] > 1000
    return result


# -----------------------------------------------------------------------------
# Setup / diagnostics
# -----------------------------------------------------------------------------

def check_ollama_models() -> dict[str, Any]:
    required = sorted({PLANNER_MODEL, SEARCH_STRATEGIST_MODEL, RESEARCH_MODEL, DEEP_ANALYSER_MODEL, BLIND_RESEARCH_MODEL, CRITIC_MODEL, WRITER_MODEL})
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        return {"ok": False, "error": "ollama executable not found", "required": required}
    proc = subprocess.run([ollama_bin, "list"], capture_output=True, text=True, check=False)
    available = set()
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            available.add(parts[0])
    missing = [m for m in required if m not in available]
    return {
        "ok": not missing,
        "required": required,
        "available": sorted(available),
        "missing": missing,
        "ollama_status": proc.returncode,
    }


def smoke_test() -> int:
    print("=== Ollama model check ===")
    models = check_ollama_models()
    print(json.dumps(models, indent=2))
    if not models["ok"]:
        return 1

    print("\n=== Web search check ===")
    web = ddgs_search("site:arxiv.org underwater ROV control", max_results=3)
    print(json.dumps(web, indent=2, ensure_ascii=False)[:6000])
    print("\n=== Semantic Scholar check ===")
    if not S2_RUNTIME_ENABLED:
        print("Semantic Scholar disabled by configuration; skipping.")
    else:
        try:
            s2 = s2_search("underwater ROV control", 3, 0)
            if s2:
                print(json.dumps([p.__dict__ for p in s2], indent=2, ensure_ascii=False)[:6000])
            else:
                print("Semantic Scholar returned no records. Continuing with OpenAlex/Crossref.")
        except Exception as exc:
            print(f"Semantic Scholar unavailable: {exc}")
            print("Continuing: Semantic Scholar is optional because OpenAlex + Crossref remain available.")

    print("\n=== OpenAlex check ===")
    try:
        oa = openalex_search("underwater ROV control", 3, 1)
        print(json.dumps([p.__dict__ for p in oa], indent=2, ensure_ascii=False)[:6000])
    except Exception as exc:
        print(f"OpenAlex check failed: {exc}")
        return 1

    print("\nSmoke test passed.")
    return 0


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

async def run_research(question: str) -> dict[str, Any]:
    print("\n=== 1. PLANNING ===")
    plan = await plan_research(question)
    save_json(RUN_DIR / "research_plan.json", plan.model_dump())
    print(f"Created {len(plan.tasks)} research tasks.")
    for task in plan.tasks:
        print(f"  - {task.name}")

    db = PaperDB(RUN_DIR / "papers.sqlite")
    try:
        print("\n=== 2. PASS 1 - BROAD DISCOVERY ===")
        async def run_task(task: ResearchTask):
            async with TASK_SEM:
                stats = await harvest_task(db, task, question, "pass1", blind=False)
                print(f"[{task.name}] {stats['candidate_count']} candidates")
                return stats
        pass1 = await asyncio.gather(*(run_task(t) for t in plan.tasks))
        save_json(RUN_DIR / "pass1_summary.json", pass1)

        print("\n=== 3. PASS 2 - DEEP PAPER ANALYSIS ===")
        evidence = await build_evidence_corpus(db, plan, question)

        print("\n=== 4. PASS 3 - FIRST SYNTHESIS + ADVERSARIAL REVIEW ===")
        first = await first_synthesis(question, plan, evidence)
        review = await critique(question, plan, first)
        save_json(RUN_DIR / "critique.json", review.model_dump())

        print("\n=== 5. PASS 4 - BLIND MISSED-EVIDENCE SEARCH ===")
        gap_plans = await make_gap_plans(question, plan, first, review)
        gap_stats = await targeted_gap_search(question, gap_plans, db, plan, "blind_gap")
        save_json(RUN_DIR / "blind_gap_summary.json", gap_stats)

        # Optional additional independent web-research pass using browsing agents.
        if RUN_WEB_DEEP_READ:
            print("\n=== 6. WEB RESEARCH AGENT PASS ===")
            web_agent_outputs = []
            for task, gap in zip(plan.tasks, gap_plans):
                try:
                    out = await web_research_agent(task, question, gap.queries[:10], model_name=BLIND_RESEARCH_MODEL)
                    web_agent_outputs.append({"task": task.name, "output": out})
                except Exception as exc:
                    web_agent_outputs.append({"task": task.name, "error": str(exc)})
            save_json(RUN_DIR / "web_agent_outputs.json", web_agent_outputs)

        # Rebuild deep evidence after the blind pass.
        print("\n=== 7. POST-GAP DEEP ANALYSIS ===")
        evidence2 = await build_evidence_corpus(db, plan, question)
        save_json(RUN_DIR / "evidence_cards_post_gap.json", evidence2)

        print("\n=== 8. FINAL SYNTHESIS ===")
        final_task_summaries = await build_task_summaries(
            question,
            plan,
            evidence2,
            "task_summaries_post_gap.json",
        )
        refs = build_reference_catalog(db, plan)
        final_report = await write_final_report(question, plan, final_task_summaries, first, review, refs)
        final_review = await final_research_review(question, final_report)
        save_json(RUN_DIR / "final_review.json", final_review.model_dump())

        # One final repair pass when the reviewer finds serious problems.
        if not final_review.ready_for_pdf:
            repair_prompt = f"""
Repair this literature review based on the final reviewer findings.

Question:
{question}

Draft:
{final_report}

Final reviewer findings:
{final_review.model_dump_json(indent=2)}

Return the complete corrected report. Keep the technical explanation detailed.
Do not invent new references.
"""
            agent = create_agent(
                model=make_llm(WRITER_MODEL, 0, "repair"),
                tools=[],
                system_prompt="You are the final scientific editor.",
            )
            async with LLM_SEM:
                repaired = await agent_invoke_with_progress(
                    agent,
                    [("user", repair_prompt)],
                    "Qwen3 report repair",
                )
            final_report = repaired["messages"][-1].content
            (RUN_DIR / "repaired_report.md").write_text(final_report, encoding="utf-8")

        final_report = append_references(final_report, refs)
        final_md = RUN_DIR / "final_report.md"
        final_md.write_text(final_report, encoding="utf-8")

        print("\n=== 9. PAPER CORPUS EXPORT ===")
        corpus_csv = RUN_DIR / "paper_corpus.csv"
        db.export_csv(corpus_csv)

        print("\n=== 10. PDF GENERATION ===")
        pdf_path = ROOT / f"deep_research_{RUN_STAMP}.pdf"
        write_pdf(plan.title, final_report, pdf_path)
        pdf_check = verify_pdf(pdf_path)
        save_json(RUN_DIR / "pdf_verification.json", pdf_check)

        paper_counts = {t.name: db.count(t.name) for t in plan.tasks}
        source_counts = {t.name: db.source_counts(t.name) for t in plan.tasks}
        summary = {
            "question": question,
            "research_mode": RESEARCH_MODE,
            "title": plan.title,
            "pdf": str(pdf_path),
            "run_dir": str(RUN_DIR),
            "paper_counts": paper_counts,
            "source_counts": source_counts,
            "minimum_candidate_papers": MIN_CANDIDATE_PAPERS,
            "deep_read_per_task": DEEP_READ_TOP,
            "models": {
                "planner": PLANNER_MODEL,
                "search_strategist": SEARCH_STRATEGIST_MODEL,
                "researcher": RESEARCH_MODEL,
                "deep_analyser": DEEP_ANALYSER_MODEL,
                "blind_researcher": BLIND_RESEARCH_MODEL,
                "critic": CRITIC_MODEL,
                "gap": GAP_MODEL,
                "writer": WRITER_MODEL,
            },
            "llm_runtime": {
                "heartbeat_seconds": LLM_HEARTBEAT_SECONDS,
                "retries": LLM_RETRIES,
                "http_timeout_seconds": LLM_HTTP_TIMEOUT_SECONDS,
                "max_concurrency": MAX_LLM_CONCURRENCY,
                "task_synthesis_tokens": LLM_TOKENS_TASK_SYNTHESIS,
                "global_synthesis_tokens": LLM_TOKENS_SYNTHESIS,
                "task_synthesis_input_chars": SYNTHESIS_TASK_INPUT_CHARS,
                "global_synthesis_input_chars": SYNTHESIS_GLOBAL_INPUT_CHARS,
            },
            "pdf_verification": pdf_check,
        }
        save_json(RUN_DIR / "run_summary.json", summary)
        return summary
    finally:
        db.close()


def _validate_mode_config(mode: str, values: dict[str, int | bool]) -> None:
    if values["min_candidates"] < 1:
        raise ValueError(f"{mode} mode: candidate-paper minimum must be >= 1")
    if values["discovery_target"] < values["min_candidates"]:
        raise ValueError(f"{mode} mode: discovery target must be >= candidate-paper minimum")
    if values["deep_read"] < 1:
        raise ValueError(f"{mode} mode: deep-read count must be >= 1")
    if values["llm_concurrency"] < 1 or values["task_concurrency"] < 1:
        raise ValueError(f"{mode} mode: concurrency values must be >= 1")
    for key in ("search_queries", "web_queries", "blind_queries", "search_pages", "citation_hubs"):
        if values[key] < 1:
            raise ValueError(f"{mode} mode: {key} must be >= 1")


def apply_research_mode(mode: str) -> None:
    global RESEARCH_MODE
    global MIN_CANDIDATE_PAPERS, DISCOVERY_TARGET, DEEP_READ_TOP
    global MAX_LLM_CONCURRENCY, MAX_TASK_CONCURRENCY, SEARCH_QUERY_COUNT
    global WEB_QUERY_COUNT, BLIND_QUERY_COUNT, MAX_SEARCH_PAGES, MAX_CITATION_HUBS
    global RUN_WEB_DEEP_READ, LLM_SEM, TASK_SEM

    if mode not in {"short", "deep"}:
        raise ValueError(f"Unknown research mode: {mode}")

    if mode == "short":
        values = {
            "min_candidates": SHORT_MIN_CANDIDATE_PAPERS,
            "discovery_target": SHORT_DISCOVERY_TARGET,
            "deep_read": SHORT_PAPERS_TO_DEEP_READ,
            "llm_concurrency": SHORT_MAX_LLM_CONCURRENCY,
            "task_concurrency": SHORT_MAX_TASK_CONCURRENCY,
            "search_queries": SHORT_SEARCH_QUERY_COUNT,
            "web_queries": SHORT_WEB_QUERY_COUNT,
            "blind_queries": SHORT_BLIND_QUERY_COUNT,
            "search_pages": SHORT_MAX_SEARCH_PAGES,
            "citation_hubs": SHORT_MAX_CITATION_HUBS,
            "max_web_results": SHORT_MAX_WEB_RESULTS,
            "web_deep_read": SHORT_RUN_WEB_DEEP_READ,
        }
    else:
        values = {
            "min_candidates": DEEP_MIN_CANDIDATE_PAPERS,
            "discovery_target": DEEP_DISCOVERY_TARGET,
            "deep_read": DEEP_PAPERS_TO_DEEP_READ,
            "llm_concurrency": DEEP_MAX_LLM_CONCURRENCY,
            "task_concurrency": DEEP_MAX_TASK_CONCURRENCY,
            "search_queries": DEEP_SEARCH_QUERY_COUNT,
            "web_queries": DEEP_WEB_QUERY_COUNT,
            "blind_queries": DEEP_BLIND_QUERY_COUNT,
            "search_pages": DEEP_MAX_SEARCH_PAGES,
            "citation_hubs": DEEP_MAX_CITATION_HUBS,
            "max_web_results": DEEP_MAX_WEB_RESULTS,
            "web_deep_read": DEEP_RUN_WEB_DEEP_READ,
        }

    _validate_mode_config(mode, values)
    RESEARCH_MODE = mode
    MIN_CANDIDATE_PAPERS = values["min_candidates"]
    DISCOVERY_TARGET = values["discovery_target"]
    DEEP_READ_TOP = values["deep_read"]
    MAX_LLM_CONCURRENCY = values["llm_concurrency"]
    MAX_TASK_CONCURRENCY = values["task_concurrency"]
    SEARCH_QUERY_COUNT = values["search_queries"]
    WEB_QUERY_COUNT = values["web_queries"]
    BLIND_QUERY_COUNT = values["blind_queries"]
    MAX_SEARCH_PAGES = values["search_pages"]
    MAX_CITATION_HUBS = values["citation_hubs"]
    MAX_WEB_RESULTS = values["max_web_results"]
    RUN_WEB_DEEP_READ = values["web_deep_read"]

    # Mode is selected before research starts, so recreate the semaphores with the selected limits.
    LLM_SEM = asyncio.Semaphore(MAX_LLM_CONCURRENCY)
    TASK_SEM = asyncio.Semaphore(MAX_TASK_CONCURRENCY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local multi-pass research engine using Ollama.")
    parser.add_argument("question", nargs="*", help="Research question")
    parser.add_argument("--smoke-test", action="store_true", help="Check Ollama models and research APIs")
    parser.add_argument(
        "--mode", choices=("short", "deep"), default=None,
        help="Research mode. Default: deep. Short is a compact evidence scan; deep is the full workflow.",
    )
    parser.add_argument(
        "--short-research", action="store_true",
        help="Alias for --mode short.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        raise SystemExit(smoke_test())
    if args.short_research and args.mode:
        raise SystemExit("Use either --short-research or --mode, not both.")
    if not args.question:
        print('Usage: python deep_research_engine.py [--mode short|deep] "your research question"')
        print('       python deep_research_engine.py --short-research "your research question"')
        raise SystemExit(2)
    selected_mode = "short" if args.short_research else (args.mode or "deep")
    apply_research_mode(selected_mode)
    print(
        f"\n*** {RESEARCH_MODE.upper()} RESEARCH MODE *** "
        f"| minimum papers/task={MIN_CANDIDATE_PAPERS} "
        f"| discovery target/task={DISCOVERY_TARGET} "
        f"| deep reads/task={DEEP_READ_TOP} "
        f"| LLM concurrency={MAX_LLM_CONCURRENCY} "
        f"| task concurrency={MAX_TASK_CONCURRENCY} ***\n"
    )
    result = asyncio.run(run_research(" ".join(args.question).strip()))
    print("\n=== COMPLETE ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

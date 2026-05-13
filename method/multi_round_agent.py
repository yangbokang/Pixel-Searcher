#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-round ReAct-style reasoning agent for grounding & choice tasks.

Design:
    1.  Decompose multi-hop question into sub-questions.
    2.  Iterative SEARCH / THINK / ANSWER loop (≤ MAX_SEARCH_ROUNDS rounds).
    3.  After resolving, execute grounding or choice scoring.
"""

from __future__ import annotations

import base64
import html
import http.client
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen

from config import (
    Box,
    Config,
    call_llm_text,
    call_llm_vision,
    compute_iou,
    extract_json,
    extract_llm_text,
    file_to_data_url,
    with_retry,
)
from saliency_filter import (
    CandidateBBox,
    crop_candidate,
    draw_candidates_on_image,
    render_highlight,
    run_saliency_pipeline,
)


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class SearchHit:
    query: str
    title: str
    url: str
    snippet: str
    rank: int


@dataclass
class RoundTrace:
    """Record of one reasoning round."""
    round_num: int
    action: str          # SEARCH / AUTO_SEARCH / THINK / ANSWER / FORCE_ANSWER
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedEntity:
    """Final resolved entity after multi-round reasoning."""
    entity_name: str
    visual_category: str
    entity_type: str
    key_cues: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class AgentResult:
    """Full output of the multi-round agent."""
    resolved: Optional[ResolvedEntity] = None
    rounds_used: int = 0
    traces: List[RoundTrace] = field(default_factory=list)
    sub_questions: List[str] = field(default_factory=list)
    raw_evidence: str = ""


# ===================================================================
# Web search
# ===================================================================

def _clean_html(text: str) -> str:
    val = re.sub(r"<.*?>", " ", text, flags=re.S)
    val = html.unescape(val)
    return re.sub(r"\s+", " ", val).strip()


def _decode_bing_url(raw_url: str) -> str:
    """Unwrap Bing redirect URLs when possible."""
    val = html.unescape(raw_url).strip()
    parsed = urlparse(val)
    qs = parse_qs(parsed.query)
    if "u" in qs and qs["u"]:
        token = qs["u"][0]
        # Bing commonly prefixes wrapped URLs with 'a1'
        if token.startswith("a1"):
            token = token[2:]
        padding = "=" * (-len(token) % 4)
        try:
            decoded = base64.urlsafe_b64decode((token + padding).encode("ascii"))
            decoded_text = decoded.decode("utf-8", errors="ignore").strip()
            if decoded_text.startswith("http"):
                return decoded_text
        except Exception:
            pass
    return val


def _bing_search(query: str, cfg: Config) -> List[SearchHit]:
    """Fallback HTML search using Bing."""
    url = f"https://www.bing.com/search?q={quote(query)}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=cfg.timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    blocks = re.findall(r'<li class="b_algo".*?</li>', body, flags=re.I | re.S)
    hits: List[SearchHit] = []
    for block in blocks:
        title_match = re.search(
            r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>',
            block,
            flags=re.I | re.S,
        )
        if not title_match:
            continue
        raw_u, raw_t = title_match.group(1), title_match.group(2)
        snippet_match = re.search(
            r'<div class="b_caption"[^>]*>\s*<p[^>]*>(.*?)</p>',
            block,
            flags=re.I | re.S,
        )
        hits.append(SearchHit(
            query=query,
            title=_clean_html(raw_t),
            url=_decode_bing_url(raw_u),
            snippet=_clean_html(snippet_match.group(1)) if snippet_match else "",
            rank=len(hits) + 1,
        ))
        if len(hits) >= cfg.search_results_per_query:
            break
    return hits


def _serper_request(
    cfg: Config,
    path: str,
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Call Serper if an API key is configured."""
    api_key = (cfg.serper_api_key or "").strip()
    if not api_key:
        return None
    conn = http.client.HTTPSConnection("google.serper.dev", timeout=int(cfg.timeout))
    body = json.dumps(payload)
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    try:
        conn.request("POST", path, body, headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="ignore")
        if resp.status >= 400:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _serper_search(query: str, cfg: Config) -> List[SearchHit]:
    """Search via Serper Google Search API."""
    data = _serper_request(cfg, "/search", {"q": query})
    if not data:
        return []
    organic = data.get("organic", [])
    hits: List[SearchHit] = []
    if isinstance(organic, list):
        for item in organic:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            link = str(item.get("link", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if not title and not link:
                continue
            hits.append(SearchHit(
                query=query,
                title=title,
                url=link,
                snippet=snippet,
                rank=len(hits) + 1,
            ))
            if len(hits) >= cfg.search_results_per_query:
                break
    return hits


def _serper_image_search(query: str, cfg: Config, max_results: int = 5) -> List[str]:
    """Image search via Serper Google Images API."""
    data = _serper_request(cfg, "/images", {"q": query})
    if not data:
        return []
    items = data.get("images", [])
    urls: List[str] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("imageUrl", "thumbnailUrl", "imageUrlLow", "thumbnail"):
                val = str(item.get(key, "")).strip()
                if val.startswith("http") and val not in urls:
                    urls.append(val)
                    break
            if len(urls) >= max_results:
                break
    return urls


def web_search(query: str, cfg: Config) -> List[SearchHit]:
    """Run web search and return parsed results."""
    serper_hits = _serper_search(query, cfg)
    if serper_hits:
        return serper_hits
    return _bing_search(query, cfg)


def image_search(query: str, cfg: Config, max_results: int = 5) -> List[str]:
    """Run image search and return image URLs."""
    return _serper_image_search(query, cfg, max_results=max_results)


def download_reference_images(
    urls: List[str],
    cache_dir: Path,
    max_images: int = 3,
    timeout: int = 10,
) -> List[Path]:
    """Download reference images to local cache. Returns list of saved paths."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for i, url in enumerate(urls[:max_images * 2]):
        if len(saved) >= max_images:
            break
        out_path = cache_dir / f"ref_{i}.jpg"
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp:
                content = resp.read()
            # Basic validation: must be >1KB and look like an image
            if len(content) < 1024:
                continue
            out_path.write_bytes(content)
            # Verify with PIL
            from PIL import Image
            with Image.open(out_path) as img:
                img.verify()
            saved.append(out_path)
        except Exception:
            out_path.unlink(missing_ok=True)
            continue
    return saved


# ===================================================================
# Prompt templates
# ===================================================================

_DECOMPOSE_PROMPT = """\
You are decomposing a multi-hop visual grounding question into simpler \
atomic sub-questions that can each be answered by a single web search.

Question: {question}

Return strict JSON only:
{{"sub_questions": ["sub-question 1", "sub-question 2", ...]}}

Rules:
1. 1-3 sub-questions, ordered by reasoning dependency.
2. Each sub-question should target one hop of reasoning.
3. If the question is already simple, return it as the only sub-question.
4. Preserve the final target of the original question. If the question asks \
about the item/person in the image, the last sub-question must still ask about \
that final target, not about an intermediate clue.
5. Do not let a year, event, or historical clue replace the final grounded \
entity. Intermediate clues are for resolving the target, not for becoming the \
target.
6. Return only JSON.
"""

_REACT_PROMPT = """\
You are a multi-round reasoning agent for visual grounding.  Your goal is to \
identify the exact entity described by the question so it can be located in \
an image.

Original question: {question}
Sub-questions: {sub_questions}

Accumulated evidence so far:
{evidence}

Interaction round {round_num} of {max_rounds}.  Only SEARCH / ANSWER consume \
an interaction round. THINK does not.

Return strict JSON with ONE of these forms:
1. {{"action": "SEARCH", "query": "your web search query"}}
2. {{"action": "THINK", "reasoning": "your reasoning based on evidence so far"}}
3. {{"action": "ANSWER", "entity_name": "resolved entity", \
"visual_category": "phone/person/car/...", \
"entity_type": "device/person/character/vehicle/object", \
"key_cues": ["cue1", "cue2"], "confidence": 0.0-1.0}}

Guidelines:
- Use SEARCH to gather information you don't have yet.
- Use THINK only to briefly consolidate evidence before the next action.
- If evidence is still missing or ambiguous, prefer SEARCH over repeated THINK.
- You may use at most one THINK before you must SEARCH with a different query \
or ANSWER.
- Do not repeat the same reasoning across rounds.
- Use ANSWER only when you are confident about the entity.
- If ambiguity remains, issue a more targeted SEARCH instead of restating the \
same conclusion.
- If this is interaction round {max_rounds}, you MUST use ANSWER.
- Return only JSON.
"""

_FORCE_ANSWER_PROMPT = """\
You have gathered the following evidence about this question.  You must now \
give your best answer.

Question: {question}
Evidence: {evidence}

Return strict JSON:
{{"entity_name": "best guess entity", \
"visual_category": "phone/person/car/...", \
"entity_type": "device/person/character/vehicle/object", \
"key_cues": ["cue1", "cue2"], "confidence": 0.0-1.0, \
"remaining_ambiguities": ["ambiguity1", "ambiguity2"]}}

Rules:
- Return only JSON.
- Before answering, identify unresolved ambiguities from the evidence.
- If the evidence already resolves the entity, remaining_ambiguities can be [].
- Give your best guess even if uncertain.
"""

_VISUAL_RESEARCH_PROMPT = """\
Given search results about the appearance of "{entity_name}", extract a \
concise visual description focusing on shape, color, size, logos, and \
distinguishing physical features.

Search results:
{search_evidence}

Return strict JSON:
{{"visual_description": "1-3 sentence description of how it looks", \
"shape": "compact/tall/flat/...", "color": "primary color(s)", \
"distinctive_features": ["feature1", "feature2"]}}

Rules: Return only JSON.  Focus on external appearance.
"""

_SCORE_CANDIDATE_PROMPT = """\
You are scoring whether a highlighted visual candidate matches a text \
hypothesis.  You will see two target images first: the full image with the \
candidate highlighted in yellow, and a zoomed crop of the candidate region. \
You may also see additional web reference images of the entity after that.

Reference text: {reference_text}
Entity name: {entity_name}
Visual category: {visual_category}
Key cues: {key_cues}
Candidate id: {candidate_id}

Visual appearance from web search:
{visual_description}

Return strict JSON:
{{"support_score": 0-5, "contradiction_score": 0-5, \
"confidence": 0.0-1.0, "reason": "short reason"}}

Rules:
1. support_score: how much the visual evidence supports this being the entity.
2. contradiction_score: how much the visual evidence contradicts it.
3. Compare the candidate crop with the visual description and any attached \
reference images carefully.
4. Pay attention to shape, color, logos, text, layout, and distinctive \
features.
5. Broad boxes, multi-object boxes, and mostly-background boxes should receive \
lower support even if the right object is somewhere inside.
6. Only mention details that are actually visible or explicitly stated in the \
reference text. Do not invent hidden labels, numbers, or logos.
7. Return only JSON.
"""

_SCORE_OPTION_PROMPT = """\
You see an image with one object highlighted in yellow.  A cropped view of \
that object is also provided.  Decide which option best describes this object.

Options:
{options_text}

Entity info from search:
{entity_info}

Return strict JSON:
{{"selected_index": 0, "confidence": 0.0-1.0, "reason": "short reason"}}

Rules:
1. selected_index is 0-based index of the best matching option.
2. Use visible evidence from the highlighted object.
3. Return only JSON.
"""

_REFERENCE_MATCH_PROMPT = """\
You see two images:
1. A REFERENCE image of "{entity_name}" found on the web
2. A CANDIDATE crop from the target image

Does the candidate show the same type/model of object as the reference?

Return strict JSON:
{{"match_score": 0-5, "reason": "short reason"}}

Rules:
1. match_score 5 = definitely the same object type/model.
2. match_score 0 = completely different objects.
3. Focus on shape, color, brand logos, and distinctive features.
4. Ignore background differences — only compare the objects themselves.
5. Return only JSON.
"""

_MINI_RESOLVE_PROMPT = """\
Given this description of an entity, identify what it refers to.

Description: {text}

Return strict JSON:
{{"entity_name": "identified entity", \
"visual_category": "phone/person/car/...", \
"entity_type": "device/person/character/vehicle/object", \
"key_cues": ["cue1", "cue2"], "confidence": 0.0-1.0}}

Rules: Return only JSON.
"""

_ENTITY_VERIFY_PROMPT = """\
You are checking whether a resolved entity is actually consistent with a \
visual grounding question and the gathered evidence.

Question: {question}
Proposed entity: {entity_name}
Visual category: {visual_category}
Entity type: {entity_type}
Key cues: {key_cues}

Evidence:
{evidence}

Return strict JSON:
{{"is_consistent": true,
"consistency_score": 0.0-5.0,
"issues": ["issue 1", "issue 2"],
"followup_queries": ["query 1", "query 2"]}}

Rules:
1. Mark is_consistent false if the proposed entity seems to be the wrong \
product/person/character/model, too generic, unsupported by evidence, or an \
intermediate clue rather than the final visible target in the image.
2. consistency_score 5 means the entity is well supported and specific.
3. If inconsistent, provide 1-2 targeted followup_queries to resolve the \
remaining ambiguity.
4. For model-level answers, exact evidence matters. Do not mark an entity \
consistent unless the evidence explicitly supports that exact model/person, not \
just a nearby series, sibling model, platform, or speculative variant.
5. Return only JSON.
"""

_ENTITY_REPAIR_PROMPT = """\
The current resolved entity for a visual grounding question appears unreliable.

Question: {question}
Current entity: {entity_name}
Known issues with the current entity:
{issues}

Evidence:
{evidence}

Return strict JSON:
{{"entity_name": "better entity",
"visual_category": "phone/person/car/object/...",
"entity_type": "device/person/character/vehicle/object",
"key_cues": ["cue1", "cue2"],
"confidence": 0.0-1.0}}

Rules:
1. Re-resolve the entity from the evidence. Do not stick to the current entity \
if it is unsupported.
2. Prefer the most concrete model/person/character/entity actually supported \
by the evidence.
3. If the question asks about the item/person in the image, answer that final \
visible target, not an intermediate clue used to identify it.
4. Prefer an exact model/person only when it is explicitly supported by the \
evidence; otherwise step back to the best supported visible target.
5. If evidence is insufficient, still return the best alternative guess.
6. Return only JSON.
"""

_FINAL_TARGET_RESOLVE_PROMPT = """\
Resolve the FINAL visible target of this grounding question.

Question: {question}
Evidence:
{evidence}

Return strict JSON:
{{"entity_name": "final visible target",
"visual_category": "phone/person/car/object/...",
"entity_type": "device/person/character/vehicle/object",
"key_cues": ["cue1", "cue2"],
"confidence": 0.0-1.0}}

Rules:
1. Answer the actual item/person that should be located in the image.
2. Do not answer with an intermediate clue entity, historical reference, \
designer, event, or source article unless that is also the visible target.
3. Prefer the concrete visible model/person/character over a generic series \
or franchise name.
4. Only return an exact model/person if the evidence explicitly supports that \
exact target; otherwise return the best supported visible target.
5. Return only JSON.
"""

_VISUAL_ENTITY_REPAIR_PROMPT = """\
The current resolved entity does not visually match the objects found in the \
image. Re-resolve the target entity using both the question and the visible \
candidate summary.

Question: {question}
Current entity: {entity_name}
Current visual category: {visual_category}

Visible candidates in the image:
{candidate_summary}

Scoring evidence:
{score_summary}

Return strict JSON:
{{"entity_name": "better entity",
"visual_category": "phone/person/car/object/...",
"entity_type": "device/person/character/vehicle/object",
"key_cues": ["cue1", "cue2"],
"confidence": 0.0-1.0}}

Rules:
1. Use the visible candidates as a hard constraint: the answer should be an \
entity that could plausibly appear in this image.
2. Prefer specific model-level entities when the question implies them.
3. If the current entity is impossible given the visible candidates, replace it.
4. Return only JSON.
"""

_DIRECT_GROUND_PROMPT = """\
You are locating a specific entity in the FIRST image.

The FIRST image is the target scene. Any additional images are web reference \
images of the target entity.

Question: {reference_text}
Entity name: {entity_name}
Visual category: {visual_category}
Key cues: {key_cues}

Visual appearance summary:
{visual_description}

Return strict JSON in one of these forms:
{{"bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0, "reason": "short reason"}}
{{"bbox": null, "confidence": 0.0-1.0, "reason": "short reason"}}

Rules:
1. bbox must use absolute pixel coordinates in the FIRST image only.
2. Use the attached reference images to find the same object/person/icon/model.
3. If several similar instances exist, choose the one best matching the cues.
4. Return a tight box around one concrete instance only. Avoid broad boxes that \
cover multiple objects, large empty regions, or the center area between objects.
5. If no plausible match exists, return bbox null.
6. Return only JSON.
"""

_JOINT_RANK_CANDIDATES_PROMPT = """\
You are selecting the best matching candidate in the FIRST image.

The FIRST image is the full scene with all candidate boxes labeled.
The next candidate images are crops in this order:
{candidate_order}

Any remaining images are web reference images for the target entity.

Question: {reference_text}
Entity name: {entity_name}
Visual category: {visual_category}
Key cues: {key_cues}

Visual appearance summary:
{visual_description}

Candidates:
{candidate_lines}

Return strict JSON:
{{"best_candidate_id": "candidate_x", "runner_up_candidate_id": "candidate_y or ''", \
"confidence": 0.0-1.0, "reason": "short reason"}}

Rules:
1. Compare the labeled boxes in the overview image and the candidate crops jointly.
2. Prefer exact instance-level matches, not just same coarse category.
3. Use reference images when available.
4. Prefer the tighter candidate when two boxes cover the same object but one is \
broader or contains more background.
5. The reason must only cite visible evidence or explicit web evidence. Do not \
invent jersey numbers, logos, text, colors, or details that are not clearly visible.
6. If two candidates are similar, explicitly choose the better one.
7. Return only JSON.
"""


# ===================================================================
# Multi-round ReAct loop
# ===================================================================

def decompose_question(client, cfg: Config, question: str) -> List[str]:
    """Decompose a multi-hop question into sub-questions."""
    prompt = _DECOMPOSE_PROMPT.format(question=question)
    try:
        raw = with_retry(
            lambda: call_llm_text(client, cfg, prompt, max_tokens=512),
            retries=cfg.retry_times,
        )
        if cfg.print_raw:
            print("=== DECOMPOSE RAW ===")
            print(raw)
            print()
        payload = extract_json(raw)
        if isinstance(payload, dict):
            subs = payload.get("sub_questions", [])
        elif isinstance(payload, list):
            subs = payload
        else:
            subs = []
        if isinstance(subs, list) and subs:
            return [str(s).strip() for s in subs if str(s).strip()]
    except Exception as exc:
        if cfg.print_raw:
            print(f"  [decompose] error: {exc}")
    return [question]


def _build_followup_query(
    question: str,
    sub_questions: List[str],
    traces: List[RoundTrace],
    latest_reasoning: str = "",
) -> str:
    """Pick a concrete follow-up query instead of looping in THINK."""
    domain_queries = _build_domain_queries(question, sub_questions)
    search_queries = [
        str(t.detail.get("query", "")).strip()
        for t in traces if t.action in {"SEARCH", "AUTO_SEARCH"}
    ]
    used = {q.lower() for q in search_queries if q}

    for query in domain_queries:
        if query and query.lower() not in used:
            return query

    for sub in sub_questions:
        candidate = sub.strip()
        if candidate and candidate.lower() not in used:
            return candidate

    reasoning = re.sub(r"\s+", " ", latest_reasoning).strip()
    if reasoning:
        if search_queries:
            return f"{question} {reasoning[:120]}"
        return reasoning[:160]

    if search_queries:
        return f"{question} official identity"
    return question


def _build_domain_queries(question: str, sub_questions: List[str]) -> List[str]:
    """Build targeted follow-up searches for hard domains."""
    q = re.sub(r"\s+", " ", question).strip()
    lowered = q.lower()
    queries: List[str] = []

    def add(val: str) -> None:
        val = re.sub(r"\s+", " ", val).strip()
        if val and val not in queries:
            queries.append(val)

    for sub in sub_questions:
        add(sub)

    if re.search(r"voice actor|voiced|seiyuu", lowered):
        add(f"{q} voice actor")
        add(f"{q} character voiced by")
        add(f"{q} site:wikipedia.org")
    if re.search(r"medal|record|finisher|olympics|ski|snowboard|football|race|event", lowered):
        add(f"{q} official results")
        add(f"{q} site:olympics.com")
        add(f"{q} wikipedia result")
    if re.search(r"icon|app|logo|film and television ip|franchise", lowered):
        add(f"{q} app icon")
        add(f"{q} franchise logo")
        add(f"{q} official icon")
    if re.search(r"released|announced|launched|specification|router|phone|laptop|mouse|camera|earphones|car", lowered):
        add(f"{q} official specs")
        add(f"{q} product photo")
        add(f"{q} model name")
    if re.search(r"person|athlete|player|actor|singer|character", lowered):
        add(f"{q} wikipedia")
        add(f"{q} profile")

    add(q)
    return queries[:12]


def _used_queries(traces: List[RoundTrace]) -> List[str]:
    return [
        str(t.detail.get("query", "")).strip()
        for t in traces if t.action in {"SEARCH", "AUTO_SEARCH"}
    ]


def _count_interaction_rounds(traces: List[RoundTrace]) -> int:
    """Only interactive actions count as rounds."""
    return sum(
        1 for t in traces
        if t.action in {"SEARCH", "AUTO_SEARCH", "ANSWER"}
    )


def _run_search_round(
    cfg: Config,
    query: str,
    round_num: int,
    evidence_parts: List[str],
    traces: List[RoundTrace],
    action: str = "SEARCH",
    trigger: str = "",
) -> None:
    """Execute one search step and store evidence/traces."""
    hits = web_search(query, cfg)
    hit_text = "\n".join(
        f"  [{h.rank}] {h.title} | {h.snippet}" for h in hits
    ) or "  (no results)"
    evidence_parts.append(
        f"[Round {round_num} {action} query=\"{query}\"]\n{hit_text}"
    )
    detail: Dict[str, Any] = {
        "query": query,
        "hits": [{"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits],
    }
    if trigger:
        detail["trigger"] = trigger
    traces.append(RoundTrace(round_num, action, detail))


def run_react_loop(client, cfg: Config, question: str) -> AgentResult:
    """Execute the multi-round ReAct reasoning loop."""
    sub_questions = decompose_question(client, cfg, question)
    evidence_parts: List[str] = []
    traces: List[RoundTrace] = []
    resolved: Optional[ResolvedEntity] = None

    max_rounds = cfg.max_search_rounds
    interaction_round = 1
    think_used = False
    step_guard = max_rounds * 4 + 4

    while interaction_round <= max_rounds and resolved is None and step_guard > 0:
        step_guard -= 1
        evidence_text = "\n".join(evidence_parts) if evidence_parts else "(none)"
        is_last = (interaction_round == max_rounds)

        try:
            prompt = _REACT_PROMPT.format(
                question=question,
                sub_questions=json.dumps(sub_questions, ensure_ascii=False),
                evidence=evidence_text,
                round_num=interaction_round,
                max_rounds=max_rounds,
            )

            raw = with_retry(
                lambda: call_llm_text(client, cfg, prompt, max_tokens=768),
                retries=cfg.retry_times,
            )
            if cfg.print_raw:
                print(f"=== REACT ROUND {interaction_round} RAW ===")
                print(raw)
                print()

            try:
                payload = extract_json(raw)
            except ValueError:
                payload = {"action": "THINK", "reasoning": raw[:500]}

            # handle list-type response (take first dict)
            if isinstance(payload, list):
                payload = next((item for item in payload if isinstance(item, dict)), {})

            action = str(payload.get("action", "")).upper().strip()
            if not action and isinstance(payload, dict) and payload.get("entity_name"):
                action = "ANSWER"

            if action == "SEARCH":
                query = str(payload.get("query", "")).strip()
                used_queries = {q.lower() for q in _used_queries(traces) if q}
                if not query or query.lower() in used_queries:
                    query = _build_followup_query(
                        question, sub_questions, traces,
                        str(payload.get("reasoning", "")).strip(),
                    )
                _run_search_round(
                    cfg, query, interaction_round, evidence_parts, traces,
                    action="SEARCH",
                )
                interaction_round += 1
                think_used = False
                continue

            elif action == "THINK":
                reasoning = str(payload.get("reasoning", "")).strip()
                evidence_parts.append(f"[Round {interaction_round} THINK] {reasoning}")
                traces.append(RoundTrace(
                    interaction_round, "THINK", {"reasoning": reasoning}
                ))
                if think_used:
                    if interaction_round < max_rounds:
                        followup_query = _build_followup_query(
                            question, sub_questions, traces, reasoning,
                        )
                        _run_search_round(
                            cfg, followup_query, interaction_round, evidence_parts, traces,
                            action="AUTO_SEARCH", trigger="think_limit",
                        )
                        interaction_round += 1
                        think_used = False
                    else:
                        resolved = _force_answer(client, cfg, question, evidence_parts)
                        traces.append(RoundTrace(interaction_round, "FORCE_ANSWER", {
                            "entity_name": resolved.entity_name if resolved else "",
                        }))
                    continue
                think_used = True
                if is_last:
                    resolved = _force_answer(client, cfg, question, evidence_parts)
                    traces.append(RoundTrace(interaction_round, "FORCE_ANSWER", {
                        "entity_name": resolved.entity_name if resolved else "",
                    }))
                continue

            elif action == "ANSWER":
                resolved = _parse_answer_payload(payload)
                traces.append(RoundTrace(interaction_round, "ANSWER", {
                    "entity_name": resolved.entity_name if resolved else "",
                    "remaining_ambiguities": payload.get("remaining_ambiguities", []),
                }))
                interaction_round += 1
                think_used = False
                if resolved:
                    break
                continue

            else:
                resolved = _parse_answer_payload(payload)
                if resolved:
                    traces.append(RoundTrace(interaction_round, "ANSWER", {
                        "entity_name": resolved.entity_name,
                    }))
                    interaction_round += 1
                    break
                evidence_parts.append(f"[Round {interaction_round} UNKNOWN] {raw[:300]}")
                traces.append(RoundTrace(interaction_round, "THINK", {"raw": raw[:300]}))
                if think_used and interaction_round < max_rounds:
                    followup_query = _build_followup_query(
                        question, sub_questions, traces, raw[:160],
                    )
                    _run_search_round(
                        cfg, followup_query, interaction_round, evidence_parts, traces,
                        action="AUTO_SEARCH", trigger="unknown_action",
                    )
                    interaction_round += 1
                    think_used = False
                else:
                    think_used = True
                continue

        except Exception as exc:
            if cfg.print_raw:
                print(f"  [react] round {interaction_round} error: {exc}")
            evidence_parts.append(f"[Round {interaction_round} ERROR] {str(exc)[:200]}")
            traces.append(RoundTrace(interaction_round, "THINK", {"error": str(exc)[:200]}))
            if interaction_round < max_rounds:
                followup_query = _build_followup_query(
                    question, sub_questions, traces, str(exc)[:120],
                )
                _run_search_round(
                    cfg, followup_query, interaction_round, evidence_parts, traces,
                    action="AUTO_SEARCH", trigger="error_recovery",
                )
                interaction_round += 1
                think_used = False
            else:
                resolved = _force_answer(client, cfg, question, evidence_parts)
                traces.append(RoundTrace(interaction_round, "FORCE_ANSWER", {
                    "entity_name": resolved.entity_name if resolved else "",
                }))

    if resolved is None:
        resolved = _force_answer(client, cfg, question, evidence_parts)
        traces.append(RoundTrace(min(interaction_round, max_rounds), "FORCE_ANSWER", {
            "entity_name": resolved.entity_name if resolved else "",
        }))

    return AgentResult(
        resolved=resolved,
        rounds_used=_count_interaction_rounds(traces),
        traces=traces,
        sub_questions=sub_questions,
        raw_evidence="\n".join(evidence_parts),
    )


def _parse_answer_payload(payload: Any) -> Optional[ResolvedEntity]:
    """Parse an ANSWER action payload into a ResolvedEntity."""
    # handle list-type: take first dict
    if isinstance(payload, list):
        payload = next((item for item in payload if isinstance(item, dict)), {})
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("entity_name", "")).strip()
    if not name or name.lower() in {"unknown", "string", "resolved entity",
                                      "best guess entity"}:
        return None
    cat = str(payload.get("visual_category", "object")).strip().lower()
    etype = str(payload.get("entity_type", "object")).strip().lower()
    cues = payload.get("key_cues", [])
    if isinstance(cues, list):
        cues = [str(c).strip() for c in cues if str(c).strip()][:5]
    else:
        cues = []
    conf = payload.get("confidence", 0.5)
    if isinstance(conf, (int, float)):
        conf = max(0.0, min(1.0, float(conf)))
    else:
        conf = 0.5
    return ResolvedEntity(
        entity_name=name,
        visual_category=cat,
        entity_type=etype,
        key_cues=cues,
        confidence=conf,
    )


def _force_answer(client, cfg: Config, question: str,
                   evidence_parts: List[str]) -> Optional[ResolvedEntity]:
    """Force a structured answer when max rounds exhausted."""
    evidence_text = "\n".join(evidence_parts) if evidence_parts else "(none)"
    prompt = _FORCE_ANSWER_PROMPT.format(question=question, evidence=evidence_text)
    raw = with_retry(
        lambda: call_llm_text(client, cfg, prompt, max_tokens=512),
        retries=cfg.retry_times,
    )
    if cfg.print_raw:
        print("=== FORCE ANSWER RAW ===")
        print(raw)
        print()
    try:
        payload = extract_json(raw)
        return _parse_answer_payload(payload)
    except Exception:
        return ResolvedEntity(
            entity_name="unknown",
            visual_category="object",
            entity_type="object",
            confidence=0.1,
        )


# ===================================================================
# Entity verification / repair
# ===================================================================

def _question_targets_visible_entity(question: str) -> bool:
    """Heuristic for questions whose final answer should be the visible target."""
    lowered = re.sub(r"\s+", " ", question).lower()
    return bool(re.search(
        r"in the image|in this image|visible in the image|visible in this image|"
        r"locate|find the|identify the|which .* is shown|which .* pictured|"
        r"what is the name of .* in the image|what is the .* in the image|"
        r"which .* in the image",
        lowered,
    ))


def _normalize_entity_text(text: str) -> str:
    """Normalize entity text for exact-support checks."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _entity_support_queries(entity: ResolvedEntity) -> List[str]:
    """Build targeted exact-match follow-up queries for entity verification."""
    cat = (entity.visual_category or entity.entity_type or "object").strip()
    return [
        f"\"{entity.entity_name}\" official {cat}".strip(),
        f"\"{entity.entity_name}\" exact model".strip(),
    ]


def _evidence_supports_exact_entity(entity: ResolvedEntity, evidence_text: str) -> bool:
    """Require explicit evidence support for exact model-level entities."""
    norm_entity = _normalize_entity_text(entity.entity_name)
    norm_evidence = _normalize_entity_text(evidence_text)
    if not norm_entity or not norm_evidence:
        return False
    if norm_entity in norm_evidence:
        return True

    tokens = [
        tok for tok in norm_entity.split()
        if len(tok) >= 2 and tok not in {"the", "and", "edition", "model", "series"}
    ]
    if not tokens:
        return False

    # Names with digits or several distinctive tokens need stronger support.
    requires_exact = bool(re.search(r"\d", norm_entity)) or len(tokens) >= 3
    if requires_exact:
        return all(tok in norm_evidence for tok in tokens)

    # Simpler names can pass if most distinctive tokens are present.
    present = sum(tok in norm_evidence for tok in tokens)
    return present >= max(1, len(tokens) - 1)


def _resolve_final_target_from_evidence(
    client,
    cfg: Config,
    question: str,
    evidence_text: str,
) -> Optional[ResolvedEntity]:
    """Ask explicitly for the final visible target instead of an intermediate clue."""
    prompt = _FINAL_TARGET_RESOLVE_PROMPT.format(
        question=question,
        evidence=evidence_text[:6000] if evidence_text else "(none)",
    )
    raw = with_retry(
        lambda: call_llm_text(client, cfg, prompt, max_tokens=384),
        retries=cfg.retry_times,
    )
    if cfg.print_raw:
        print("=== FINAL TARGET RESOLVE RAW ===")
        print(raw)
        print()
    return _parse_answer_payload(extract_json(raw))

def _verify_resolved_entity(
    client,
    cfg: Config,
    question: str,
    entity: ResolvedEntity,
    evidence_text: str,
) -> Dict[str, Any]:
    """Check whether the resolved entity is actually supported."""
    prompt = _ENTITY_VERIFY_PROMPT.format(
        question=question,
        entity_name=entity.entity_name,
        visual_category=entity.visual_category or "object",
        entity_type=entity.entity_type or "object",
        key_cues="; ".join(entity.key_cues) if entity.key_cues else "none",
        evidence=evidence_text[:5000] if evidence_text else "(none)",
    )
    raw = with_retry(
        lambda: call_llm_text(client, cfg, prompt, max_tokens=384),
        retries=cfg.retry_times,
    )
    if cfg.print_raw:
        print("=== ENTITY VERIFY RAW ===")
        print(raw)
        print()

    payload = extract_json(raw)
    if not isinstance(payload, dict):
        return {
            "is_consistent": True,
            "consistency_score": 3.0,
            "issues": [],
            "followup_queries": [],
        }

    issues = payload.get("issues", [])
    if isinstance(issues, list):
        issues = [str(v).strip() for v in issues if str(v).strip()][:4]
    else:
        issues = []

    queries = payload.get("followup_queries", [])
    if isinstance(queries, list):
        queries = [str(v).strip() for v in queries if str(v).strip()][:2]
    else:
        queries = []

    try:
        score = float(payload.get("consistency_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    is_consistent = bool(payload.get("is_consistent", False))
    if not _evidence_supports_exact_entity(entity, evidence_text):
        exact_issue = "exact_entity_not_explicitly_supported_by_evidence"
        if exact_issue not in issues:
            issues.append(exact_issue)
        if not queries:
            queries = _entity_support_queries(entity)
        is_consistent = False
        score = min(score, 2.5)

    return {
        "is_consistent": is_consistent,
        "consistency_score": max(0.0, min(5.0, score)),
        "issues": issues,
        "followup_queries": queries,
    }


def _repair_entity_if_needed(
    client,
    cfg: Config,
    question: str,
    agent_result: AgentResult,
) -> Tuple[AgentResult, Dict[str, Any]]:
    """Run a verification gate and repair the entity when it looks wrong."""
    if agent_result.resolved is None:
        return agent_result, {
            "is_consistent": False,
            "consistency_score": 0.0,
            "issues": ["no_resolved_entity"],
            "followup_queries": [],
            "repaired": False,
        }

    verification = _verify_resolved_entity(
        client,
        cfg,
        question,
        agent_result.resolved,
        agent_result.raw_evidence,
    )
    if _question_targets_visible_entity(question):
        try:
            visible_target = _resolve_final_target_from_evidence(
                client, cfg, question, agent_result.raw_evidence,
            )
        except Exception:
            visible_target = None
        if (
            visible_target is not None
            and visible_target.entity_name.strip()
            and visible_target.entity_name.strip().lower()
            != agent_result.resolved.entity_name.strip().lower()
        ):
            visible_check = _verify_resolved_entity(
                client,
                cfg,
                question,
                visible_target,
                agent_result.raw_evidence,
            )
            if (
                float(visible_check.get("consistency_score", 0.0) or 0.0)
                >= float(verification.get("consistency_score", 0.0) or 0.0) + 0.25
                or (
                    visible_check.get("is_consistent", False)
                    and not verification.get("is_consistent", False)
                )
            ):
                verification = dict(visible_check)
                verification["repaired"] = True
                verification["original_entity_name"] = agent_result.resolved.entity_name
                agent_result = AgentResult(
                    resolved=visible_target,
                    rounds_used=agent_result.rounds_used,
                    traces=agent_result.traces,
                    sub_questions=agent_result.sub_questions,
                    raw_evidence=agent_result.raw_evidence,
                )
    needs_repair = (
        not verification.get("is_consistent", False)
        or float(verification.get("consistency_score", 0.0) or 0.0) < 3.5
    )
    if not needs_repair:
        verification["repaired"] = False
        return agent_result, verification

    traces = list(agent_result.traces)
    evidence_parts = [agent_result.raw_evidence] if agent_result.raw_evidence else []
    if verification.get("issues"):
        evidence_parts.append(
            "[Entity verification issues] " + "; ".join(verification["issues"])
        )

    next_round = (max((t.round_num for t in traces), default=0) + 1)
    followups = verification.get("followup_queries", [])[:2]
    if not followups:
        followups = [
            _build_followup_query(
                question,
                agent_result.sub_questions,
                traces,
                "; ".join(verification.get("issues", [])),
            )
        ]

    for query in followups:
        _run_search_round(
            cfg,
            query,
            next_round,
            evidence_parts,
            traces,
            action="AUTO_SEARCH",
            trigger="entity_verification",
        )
        next_round += 1

    repaired = None
    if _question_targets_visible_entity(question):
        try:
            repaired = _resolve_final_target_from_evidence(
                client, cfg, question, "\n".join(evidence_parts),
            )
        except Exception:
            repaired = None

    prompt_repaired = None
    try:
        repair_prompt = _ENTITY_REPAIR_PROMPT.format(
            question=question,
            entity_name=agent_result.resolved.entity_name,
            issues="\n".join(f"- {item}" for item in verification.get("issues", [])) or "- unsupported current entity",
            evidence="\n".join(evidence_parts)[:6000],
        )
        raw = with_retry(
            lambda: call_llm_text(client, cfg, repair_prompt, max_tokens=384),
            retries=cfg.retry_times,
        )
        if cfg.print_raw:
            print("=== ENTITY REPAIR RAW ===")
            print(raw)
            print()
        prompt_repaired = _parse_answer_payload(extract_json(raw))
    except Exception:
        prompt_repaired = None

    if repaired is None:
        repaired = prompt_repaired
    elif (
        prompt_repaired is not None
        and prompt_repaired.entity_name.strip().lower() != repaired.entity_name.strip().lower()
    ):
        try:
            repaired_check = _verify_resolved_entity(
                client,
                cfg,
                question,
                repaired,
                "\n".join(evidence_parts),
            )
            prompt_check = _verify_resolved_entity(
                client,
                cfg,
                question,
                prompt_repaired,
                "\n".join(evidence_parts),
            )
            if float(prompt_check.get("consistency_score", 0.0) or 0.0) > float(
                repaired_check.get("consistency_score", 0.0) or 0.0
            ):
                repaired = prompt_repaired
        except Exception:
            pass

    if repaired is None:
        repaired = _force_answer(client, cfg, question, evidence_parts)

    if repaired:
        repair_check = _verify_resolved_entity(
            client,
            cfg,
            question,
            repaired,
            "\n".join(evidence_parts),
        )
        if (
            float(repair_check.get("consistency_score", 0.0) or 0.0)
            >= float(verification.get("consistency_score", 0.0) or 0.0)
            and (
                repair_check.get("is_consistent", False)
                or repaired.entity_name.strip().lower() != agent_result.resolved.entity_name.strip().lower()
            )
        ):
            traces.append(RoundTrace(next_round, "FORCE_ANSWER", {
                "entity_name": repaired.entity_name,
                "trigger": "entity_verification",
            }))
            repaired_result = AgentResult(
                resolved=repaired,
                rounds_used=_count_interaction_rounds(traces),
                traces=traces,
                sub_questions=agent_result.sub_questions,
                raw_evidence="\n".join(evidence_parts),
            )
            repair_check["repaired"] = True
            repair_check["original_entity_name"] = agent_result.resolved.entity_name
            return repaired_result, repair_check

    verification["repaired"] = False
    return agent_result, verification


# ===================================================================
# Mini resolve (for choice option texts)
# ===================================================================

def mini_resolve_text(client, cfg: Config, text: str) -> ResolvedEntity:
    """Quick 1-shot resolution of a descriptive text into an entity."""
    # check if search might help
    needs_search = bool(re.search(
        r"\b(20\d{2}|released|announced|launched|predecessor|successor|"
        r"co-founded|codename|architect|voice actor|born|debut)\b",
        text, flags=re.I,
    ))

    if needs_search:
        # do one quick search
        compact = re.sub(r"\b(the|device|in the image|that|which|is a)\b",
                         " ", text, flags=re.I)
        compact = re.sub(r"\s+", " ", compact).strip()[:120]
        hits = web_search(compact, cfg)
        if hits:
            evidence = "\n".join(f"  {h.title}: {h.snippet}" for h in hits[:3])
            text_for_resolve = f"{text}\n\nSearch evidence:\n{evidence}"
        else:
            text_for_resolve = text
    else:
        text_for_resolve = text

    prompt = _MINI_RESOLVE_PROMPT.format(text=text_for_resolve)
    raw = with_retry(
        lambda: call_llm_text(client, cfg, prompt, max_tokens=512),
        retries=cfg.retry_times,
    )
    try:
        payload = extract_json(raw)
        ent = _parse_answer_payload(payload)
        if ent:
            return ent
    except Exception:
        pass
    return ResolvedEntity(
        entity_name="unknown", visual_category="object",
        entity_type="object", confidence=0.1,
    )


# ===================================================================
# Visual research — search for entity appearance
# ===================================================================

def _search_visual_appearance(client, cfg: Config,
                                entity: ResolvedEntity) -> str:
    """Search for the entity's visual appearance and return a description."""
    # Build search queries targeting appearance
    queries = [
        f"{entity.entity_name} appearance design shape color",
        f"{entity.entity_name} what does it look like",
    ]

    all_snippets: List[str] = []
    for q in queries:
        hits = web_search(q, cfg)
        for h in hits[:3]:
            if h.snippet:
                all_snippets.append(f"- {h.title}: {h.snippet}")
        if len(all_snippets) >= 6:
            break

    if not all_snippets:
        return "(no visual information found)"

    search_evidence = "\n".join(all_snippets[:8])

    # Use LLM to summarize visual features
    prompt = _VISUAL_RESEARCH_PROMPT.format(
        entity_name=entity.entity_name,
        search_evidence=search_evidence,
    )
    try:
        raw = with_retry(
            lambda: call_llm_text(client, cfg, prompt, max_tokens=512),
            retries=cfg.retry_times,
        )
        if cfg.print_raw:
            print("=== VISUAL RESEARCH RAW ===")
            print(raw)
            print()
        payload = extract_json(raw)
        if isinstance(payload, dict):
            desc = str(payload.get("visual_description", "")).strip()
            shape = str(payload.get("shape", "")).strip()
            color = str(payload.get("color", "")).strip()
            feats = payload.get("distinctive_features", [])
            if isinstance(feats, list):
                feats = [str(f).strip() for f in feats if str(f).strip()][:5]
            else:
                feats = []
            parts = []
            if desc:
                parts.append(desc)
            if shape:
                parts.append(f"Shape: {shape}")
            if color:
                parts.append(f"Color: {color}")
            if feats:
                parts.append(f"Features: {'; '.join(feats)}")
            return "\n".join(parts) if parts else "(could not summarize)"
    except Exception as exc:
        if cfg.print_raw:
            print(f"  [visual_research] summarize failed: {exc}")

    # Fallback: just return raw snippets
    return search_evidence[:500]


# ===================================================================
# Task 1: Grounding — run_grounding
# ===================================================================

def _build_reference_queries(entity: ResolvedEntity) -> List[str]:
    """Construct image-search queries for appearance references."""
    cue_text = " ".join(entity.key_cues[:3]).strip()
    suffix = cue_text if cue_text else entity.visual_category
    cat = (entity.visual_category or "").lower()
    etype = (entity.entity_type or "").lower()
    queries = [
        f"\"{entity.entity_name}\" official image".strip(),
        f"\"{entity.entity_name}\" press image".strip(),
        f"\"{entity.entity_name}\" photo".strip(),
    ]
    if cat in {"device", "router", "phone", "mouse", "earphones", "car"} or etype == "device":
        manufacturer = ""
        upper_name = entity.entity_name.upper()
        if "ASUS" in upper_name or "ROG" in upper_name:
            manufacturer = "asus.com"
        elif "APPLE" in upper_name or "IPHONE" in upper_name or "AIRPODS" in upper_name:
            manufacturer = "apple.com"
        elif "GOPRO" in upper_name or "HERO" in upper_name:
            manufacturer = "gopro.com"
        elif "DJI" in upper_name:
            manufacturer = "dji.com"
        if manufacturer:
            queries.insert(0, f"site:{manufacturer} \"{entity.entity_name}\"".strip())
    if cat in {"person", "character"} or etype in {"person", "character"}:
        queries.append(f"\"{entity.entity_name}\" portrait".strip())
    if cat in {"icon", "logo"}:
        queries.append(f"\"{entity.entity_name}\" logo".strip())
        queries.append(f"\"{entity.entity_name}\" icon".strip())
    queries.append(f"\"{entity.entity_name}\" {suffix}".strip())
    return [q for q in queries if q][:8]


def _fetch_reference_images(
    cfg: Config,
    entity: ResolvedEntity,
    artifact_dir: Path,
    max_images: int = 3,
) -> List[Path]:
    """Search and download reference images before candidate scoring."""
    ref_dir = artifact_dir / "reference_images"
    seen_urls: List[str] = []
    for query in _build_reference_queries(entity):
        for url in image_search(query, cfg, max_results=max_images):
            if url not in seen_urls:
                seen_urls.append(url)
        if len(seen_urls) >= max_images * 2:
            break
    if cfg.print_raw:
        print(f"  [grounding] reference image urls={len(seen_urls)}")
    return download_reference_images(seen_urls, ref_dir, max_images=max_images)


def _parse_bbox_payload(payload: Any, image_path: Path) -> Optional[List[int]]:
    """Parse a bbox payload and clip it into image bounds."""
    if not isinstance(payload, dict):
        return None
    bbox_raw = payload.get("bbox")
    if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        return None
    try:
        coords = [float(v) for v in bbox_raw]
    except (TypeError, ValueError):
        return None
    from PIL import Image
    with Image.open(image_path) as img:
        w, h = img.size
    box = Box(*coords).normalize()
    if all(0.0 <= c <= 1.0 for c in coords):
        box = Box(
            coords[0] * w, coords[1] * h, coords[2] * w, coords[3] * h,
        ).normalize()
    elif max(coords) <= 1000 and (w > 1000 or h > 1000):
        box = Box(
            coords[0] / 1000.0 * w,
            coords[1] / 1000.0 * h,
            coords[2] / 1000.0 * w,
            coords[3] / 1000.0 * h,
        ).normalize()
    box = box.clip(w, h)
    if box.area <= 0:
        return None
    return box.to_int_list()


def _box_area_value(box: Any) -> float:
    """Compute raw pixel area for an xyxy box."""
    if not isinstance(box, list) or len(box) != 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _box_area_ratio(box: Any, image_path: Path) -> float:
    """Compute image-area ratio for an xyxy box."""
    area = _box_area_value(box)
    if area <= 0:
        return 0.0
    from PIL import Image
    with Image.open(image_path) as img:
        image_area = float(img.size[0] * img.size[1])
    return area / image_area if image_area > 0 else 0.0


def _ground_entity_direct(
    client,
    cfg: Config,
    image_path: Path,
    hypothesis: ResolvedEntity,
    reference_text: str,
    visual_description: str,
    ref_images: List[Path],
) -> Optional[Dict[str, Any]]:
    """Directly locate the entity on the full image using references."""
    prompt = _DIRECT_GROUND_PROMPT.format(
        reference_text=reference_text,
        entity_name=hypothesis.entity_name,
        visual_category=hypothesis.visual_category,
        key_cues="; ".join(hypothesis.key_cues) if hypothesis.key_cues else "none",
        visual_description=visual_description or "(no visual info)",
    )
    images = [image_path] + list(ref_images[:3])
    try:
        raw = with_retry(
            lambda: call_llm_vision(client, cfg, prompt, images, max_tokens=512),
            retries=cfg.retry_times,
        )
        if cfg.print_raw:
            print("=== DIRECT GROUND RAW ===")
            print(raw)
            print()
        payload = extract_json(raw)
        bbox = _parse_bbox_payload(payload, image_path)
        if bbox is None:
            return None
        conf = float(payload.get("confidence", 0.0) or 0.0)
        return {
            "bbox_xyxy": bbox,
            "confidence": max(0.0, min(1.0, conf)),
            "reason": str(payload.get("reason", "")).strip(),
        }
    except Exception as exc:
        if cfg.print_raw:
            print(f"  [direct_ground] failed: {exc}")
        return None


def _append_distinct_candidate(
    candidates: List[CandidateBBox],
    bbox_xyxy: List[int],
    label: str,
    candidate_id: str,
) -> None:
    """Add a synthetic candidate unless it duplicates an existing one."""
    new_box = Box(*[float(v) for v in bbox_xyxy]).normalize()
    for cand in candidates:
        old_box = Box(*[float(v) for v in cand.bbox_xyxy]).normalize()
        if compute_iou(new_box, old_box) >= 0.7:
            return
    candidates.append(CandidateBBox(
        bbox_xyxy=bbox_xyxy,
        label=label,
        saliency_score=1.0,
        candidate_id=candidate_id,
    ))


def _needs_dense_visual_fallback(question_text: str,
                                 hypothesis: ResolvedEntity) -> bool:
    """Identify cases where small-object fallback should be enabled."""
    q = question_text.lower()
    cat = (hypothesis.visual_category or "").lower()
    return bool(re.search(
        r"icon|logo|app|earphones|earbuds|mouse|headshot|avatar|person|athlete|"
        r"player|character|figure|poster|cover",
        q,
    )) or cat in {"person", "icon", "logo", "mouse", "earphones", "character"}


def _needs_candidate_competition(question_text: str,
                                 hypothesis: ResolvedEntity) -> bool:
    """Trigger fallback candidates for repeatable or ambiguous instance types."""
    q = question_text.lower()
    cat = (hypothesis.visual_category or "").lower()
    etype = (hypothesis.entity_type or "").lower()
    return bool(re.search(
        r"which .* image|which one|locate|find the|identify the",
        q,
    )) or cat in {
        "device", "router", "phone", "car", "person", "object", "icon",
        "mouse", "earphones", "character",
    } or etype in {"device", "person", "vehicle", "character", "object"}


def _collect_entity_conditioned_candidates(
    client,
    cfg: Config,
    image_path: Path,
    question_text: str,
    hypothesis: ResolvedEntity,
    visual_desc: str,
    ref_images: List[Path],
    direct_ground: Optional[Dict[str, Any]],
) -> List[CandidateBBox]:
    """Build candidate list with direct grounding first and saliency only as fallback."""
    candidates: List[CandidateBBox] = []
    if direct_ground:
        _append_distinct_candidate(
            candidates,
            direct_ground["bbox_xyxy"],
            hypothesis.visual_category or "direct_ground",
            "direct_ground_1",
        )

    need_dense_fallback = (
        direct_ground is None
        or float(direct_ground.get("confidence", 0.0) or 0.0) < 0.97
        or _needs_dense_visual_fallback(question_text, hypothesis)
        or _needs_candidate_competition(question_text, hypothesis)
    )
    if not need_dense_fallback:
        return candidates

    fallback_candidates = run_saliency_pipeline(
        client,
        cfg,
        image_path,
        max_boxes=max(cfg.max_boxes * 2, 40),
        top_k=max(cfg.saliency_top_k * 2, 12),
        min_area_ratio=min(cfg.min_box_area_ratio, 0.002),
        max_area_ratio=max(cfg.max_box_area_ratio, 0.98),
        use_tiling=True,
        tile_grid_size=2,
    )
    for cand in fallback_candidates:
        _append_distinct_candidate(
            candidates, cand.bbox_xyxy, cand.label, cand.candidate_id or cand.label,
        )
    return candidates


def _joint_select_candidate(
    client,
    cfg: Config,
    image_path: Path,
    artifact_dir: Path,
    hypothesis: ResolvedEntity,
    reference_text: str,
    visual_description: str,
    ref_images: List[Path],
    candidate_scores: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Jointly rank top candidates using a labeled overview plus crops."""
    if len(candidate_scores) <= 1:
        return candidate_scores[0] if candidate_scores else None

    sorted_scores = sorted(
        candidate_scores, key=lambda s: s["final_score"], reverse=True
    )
    shortlisted = list(sorted_scores[:4])
    direct_sc = next(
        (sc for sc in candidate_scores if str(sc.get("candidate_id", "")).startswith("direct_ground")),
        None,
    )
    if direct_sc is not None and all(
        sc["candidate_id"] != direct_sc["candidate_id"] for sc in shortlisted
    ):
        shortlisted = shortlisted[:3] + [direct_sc]
    candidate_boxes: List[CandidateBBox] = []
    candidate_order: List[str] = []
    candidate_lines: List[str] = []
    crop_paths: List[Path] = []
    crops_dir = artifact_dir / "candidates"

    for sc in shortlisted:
        candidate_order.append(sc["candidate_id"])
        area_ratio = round(_box_area_ratio(sc.get("bbox_xyxy"), image_path), 4)
        candidate_lines.append(
            f"- {sc['candidate_id']}: label={sc.get('label', '')}, "
            f"bbox={sc['bbox_xyxy']}, area_ratio={area_ratio}, "
            f"final_score={round(float(sc.get('final_score', 0.0)), 3)}"
        )
        candidate_boxes.append(CandidateBBox(
            bbox_xyxy=sc["bbox_xyxy"],
            label=sc.get("label", ""),
            candidate_id=sc["candidate_id"],
        ))
        crop_path = crops_dir / f"{sc['candidate_id']}_crop.png"
        if crop_path.exists():
            crop_paths.append(crop_path)

    overview_path = artifact_dir / "candidates" / "joint_overview.png"
    draw_candidates_on_image(image_path, candidate_boxes, overview_path, color="yellow")

    prompt = _JOINT_RANK_CANDIDATES_PROMPT.format(
        reference_text=reference_text,
        entity_name=hypothesis.entity_name,
        visual_category=hypothesis.visual_category,
        key_cues="; ".join(hypothesis.key_cues) if hypothesis.key_cues else "none",
        visual_description=visual_description or "(no visual info)",
        candidate_order=", ".join(candidate_order),
        candidate_lines="\n".join(candidate_lines),
    )
    images = [overview_path] + crop_paths + list(ref_images[:2])
    try:
        raw = with_retry(
            lambda: call_llm_vision(client, cfg, prompt, images, max_tokens=512),
            retries=cfg.retry_times,
        )
        if cfg.print_raw:
            print("=== JOINT RANK RAW ===")
            print(raw)
            print()
        payload = extract_json(raw)
        best_id = str(payload.get("best_candidate_id", "")).strip()
        if best_id:
            chosen = next((sc for sc in shortlisted if sc["candidate_id"] == best_id), None)
            if chosen is not None:
                chosen = dict(chosen)
                chosen["joint_reason"] = str(payload.get("reason", "")).strip()
                chosen["joint_confidence"] = max(
                    0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0))
                )
                chosen["runner_up_candidate_id"] = str(
                    payload.get("runner_up_candidate_id", "")
                ).strip()
                return chosen
    except Exception as exc:
        if cfg.print_raw:
            print(f"  [joint-rank] failed: {exc}")
    return shortlisted[0]


def _select_best_grounding_candidate(
    client,
    cfg: Config,
    image_path: Path,
    artifact_dir: Path,
    hypothesis: ResolvedEntity,
    reference_text: str,
    visual_description: str,
    ref_images: List[Path],
    candidate_scores: List[Dict[str, Any]],
    direct_ground: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pick the final candidate while protecting strong direct-ground results."""
    joint_best = _joint_select_candidate(
        client,
        cfg,
        image_path,
        artifact_dir,
        hypothesis,
        reference_text,
        visual_description,
        ref_images,
        candidate_scores,
    ) or max(candidate_scores, key=lambda s: s["final_score"])

    direct_sc = next(
        (sc for sc in candidate_scores if str(sc.get("candidate_id", "")).startswith("direct_ground")),
        None,
    )
    if direct_sc is None or not direct_ground:
        return joint_best

    direct_conf = max(0.0, min(1.0, float(direct_ground.get("confidence", 0.0) or 0.0)))
    joint_margin = float(joint_best.get("final_score", 0.0)) - float(direct_sc.get("final_score", 0.0))
    direct_support = float(direct_sc.get("support_score", 0.0)) + float(direct_sc.get("ref_match_score", 0.0)) * 0.5
    joint_support = float(joint_best.get("support_score", 0.0)) + float(joint_best.get("ref_match_score", 0.0)) * 0.5
    direct_area = _box_area_value(direct_sc.get("bbox_xyxy"))
    direct_area_ratio = _box_area_ratio(direct_sc.get("bbox_xyxy"), image_path)
    direct_is_broad = direct_area_ratio >= 0.22

    non_direct_scores = [
        sc for sc in candidate_scores
        if sc.get("candidate_id") != direct_sc.get("candidate_id")
    ]
    top_non_direct = max(
        non_direct_scores, key=lambda s: float(s.get("final_score", 0.0)), default=None
    )

    if joint_best["candidate_id"] == direct_sc["candidate_id"]:
        if top_non_direct is not None:
            alt_area = _box_area_value(top_non_direct.get("bbox_xyxy"))
            alt_gap = float(direct_sc.get("final_score", 0.0)) - float(top_non_direct.get("final_score", 0.0))
            alt_support = float(top_non_direct.get("support_score", 0.0)) + float(top_non_direct.get("ref_match_score", 0.0)) * 0.5
            if (
                alt_gap <= 0.6
                and alt_support >= direct_support - 0.35
                and alt_area > 0
                and (
                    direct_is_broad
                    or direct_area > alt_area * 1.2
                )
            ):
                chosen = dict(top_non_direct)
                chosen["selection_mode"] = "direct_demoted_to_precise"
                chosen["guarded_over"] = direct_sc["candidate_id"]
                return chosen
        chosen = dict(joint_best)
        chosen["selection_mode"] = "joint_direct"
        return chosen

    if (
        not direct_is_broad
        and direct_conf >= 0.95
        and joint_margin <= 0.1
        and direct_support >= joint_support + 0.15
        and (
            _box_area_value(joint_best.get("bbox_xyxy")) <= 0.0
            or direct_area <= _box_area_value(joint_best.get("bbox_xyxy")) * 1.05
        )
    ):
        chosen = dict(direct_sc)
        chosen["selection_mode"] = "direct_guard"
        chosen["guarded_over"] = joint_best["candidate_id"]
        return chosen

    chosen = dict(joint_best)
    chosen["selection_mode"] = "joint_rank"
    return chosen


def _should_trigger_visual_entity_repair(
    candidate_scores: List[Dict[str, Any]],
) -> bool:
    """Detect cases where the resolved entity conflicts with all visible candidates."""
    if not candidate_scores:
        return False
    top_score = max(float(sc.get("final_score", 0.0) or 0.0) for sc in candidate_scores)
    supportive = sum(float(sc.get("support_score", 0.0) or 0.0) >= 2.0 for sc in candidate_scores)
    mostly_contradict = all(
        float(sc.get("support_score", 0.0) or 0.0) <= 1.0
        and float(sc.get("contradiction_score", 0.0) or 0.0) >= 4.0
        for sc in candidate_scores[: min(len(candidate_scores), 6)]
    )
    return top_score < 0.75 or (supportive == 0 and mostly_contradict)


def _repair_entity_with_visual_candidates(
    client,
    cfg: Config,
    question_text: str,
    hypothesis: ResolvedEntity,
    candidate_scores: List[Dict[str, Any]],
) -> Optional[ResolvedEntity]:
    """Use visible candidate labels and score contradictions to re-resolve the entity."""
    shortlisted = sorted(
        candidate_scores, key=lambda s: s.get("final_score", 0.0), reverse=True
    )[:6]
    candidate_summary = "\n".join(
        f"- {sc['candidate_id']}: label={sc.get('label', '')}, bbox={sc.get('bbox_xyxy')}"
        for sc in shortlisted
    ) or "(none)"
    score_summary = "\n".join(
        f"- {sc['candidate_id']}: support={sc.get('support_score', 0)}, "
        f"contradiction={sc.get('contradiction_score', 0)}, "
        f"reason={str(sc.get('reason', '')).strip()[:180]}"
        for sc in shortlisted
    ) or "(none)"
    prompt = _VISUAL_ENTITY_REPAIR_PROMPT.format(
        question=question_text,
        entity_name=hypothesis.entity_name,
        visual_category=hypothesis.visual_category or "object",
        candidate_summary=candidate_summary,
        score_summary=score_summary,
    )
    try:
        raw = with_retry(
            lambda: call_llm_text(client, cfg, prompt, max_tokens=384),
            retries=cfg.retry_times,
        )
        if cfg.print_raw:
            print("=== VISUAL ENTITY REPAIR RAW ===")
            print(raw)
            print()
        return _parse_answer_payload(extract_json(raw))
    except Exception:
        return None


def _score_candidate_vs_references(
    client, cfg: Config,
    ref_images: List[Path],
    crop_path: Path,
    entity_name: str,
) -> float:
    """Score a candidate crop against reference images. Returns avg match score."""
    if not ref_images:
        return 0.0
    total = 0.0
    count = 0
    prompt = _REFERENCE_MATCH_PROMPT.format(entity_name=entity_name)
    for ref_path in ref_images:
        try:
            raw = with_retry(
                lambda rp=ref_path: call_llm_vision(
                    client, cfg, prompt, [rp, crop_path], max_tokens=256
                ),
                retries=cfg.retry_times,
            )
            payload = extract_json(raw)
            sc = max(0.0, min(5.0, float(payload.get("match_score", 0))))
            total += sc
            count += 1
            if cfg.print_raw:
                reason = str(payload.get("reason", ""))[:80]
                print(f"    [ref-match] {ref_path.name} → score={sc:.1f} ({reason})")
        except Exception:
            continue
    return total / count if count > 0 else 0.0


def run_grounding(
    client,
    cfg: Config,
    image_path: Path,
    question_text: str,
    artifact_dir: Path,
) -> Dict[str, Any]:
    """
    Full grounding pipeline:
      1. Multi-round ReAct → resolve entity
      2. Visual research → search for entity appearance
      3. Saliency pipeline → candidate bboxes
      4. Score candidates vs hypothesis + visual description
      5. Reference-image tie-breaking for top candidates
      6. Return best bbox
    """
    # Phase 1: resolve entity via ReAct
    agent_result = run_react_loop(client, cfg, question_text)
    if cfg.print_raw:
        ent = agent_result.resolved
        print(f"  [grounding] ReAct done: rounds={agent_result.rounds_used}, "
              f"entity={ent.entity_name if ent else 'None'}")

    if agent_result.resolved is None:
        return _grounding_empty(agent_result)

    agent_result, entity_verification = _repair_entity_if_needed(
        client, cfg, question_text, agent_result
    )
    hypothesis = agent_result.resolved
    if hypothesis is None:
        return _grounding_empty(agent_result)

    # Phase 2: visual research — search for what the entity looks like
    try:
        visual_desc = _search_visual_appearance(client, cfg, hypothesis)
    except Exception as exc:
        if cfg.print_raw:
            print(f"  [grounding] visual research failed: {exc}")
        visual_desc = "(no visual information available)"
    if cfg.print_raw:
        print(f"  [grounding] visual_desc: {visual_desc[:200]}")

    # Phase 3: fetch reference images early and try direct grounding
    try:
        ref_images = _fetch_reference_images(cfg, hypothesis, artifact_dir, max_images=3)
    except Exception as exc:
        if cfg.print_raw:
            print(f"  [grounding] reference image search failed: {exc}")
        ref_images = []
    if cfg.print_raw:
        print(f"  [grounding] downloaded {len(ref_images)} reference images")

    direct_ground = _ground_entity_direct(
        client, cfg, image_path, hypothesis, question_text, visual_desc, ref_images,
    )

    # Phase 4: entity-conditioned grounding is primary; saliency is fallback only
    try:
        candidates = _collect_entity_conditioned_candidates(
            client, cfg, image_path, question_text, hypothesis,
            visual_desc, ref_images, direct_ground,
        )
    except Exception as exc:
        if cfg.print_raw:
            print(f"  [grounding] candidate collection failed: {exc}")
        candidates = []

    if cfg.print_raw and direct_ground:
        print(f"  [grounding] direct bbox={direct_ground['bbox_xyxy']}")

    if not candidates:
        # Final dense fallback for small or difficult targets.
        try:
            dense_candidates = run_saliency_pipeline(
                client,
                cfg,
                image_path,
                max_boxes=max(cfg.max_boxes * 3, 60),
                top_k=max(cfg.saliency_top_k * 2, 14),
                min_area_ratio=0.001,
                max_area_ratio=0.99,
                use_tiling=True,
                tile_grid_size=3,
            )
            for cand in dense_candidates:
                _append_distinct_candidate(
                    candidates, cand.bbox_xyxy, cand.label, cand.candidate_id or cand.label,
                )
        except Exception as exc:
            if cfg.print_raw:
                print(f"  [grounding] dense fallback failed: {exc}")
        if not candidates:
            return _grounding_empty(agent_result)

    # Phase 5: score each candidate with visual description + reference images
    scores: List[Dict[str, Any]] = []
    for cand in candidates:
        try:
            sc = _score_one_candidate(
                client, cfg, image_path, cand, hypothesis,
                question_text, artifact_dir, visual_desc, ref_images,
            )
            if direct_ground:
                direct_box = Box(*[float(v) for v in direct_ground["bbox_xyxy"]]).normalize()
                cand_box = Box(*[float(v) for v in cand.bbox_xyxy]).normalize()
                overlap = compute_iou(direct_box, cand_box)
                direct_ratio = _box_area_ratio(direct_ground["bbox_xyxy"], image_path)
                sc["direct_ground_iou"] = round(overlap, 3)
                if overlap >= 0.5 and direct_ratio < 0.22:
                    sc["final_score"] += 0.25
                elif overlap >= 0.2 and direct_ratio < 0.16:
                    sc["final_score"] += 0.1
            scores.append(sc)
        except Exception as exc:
            if cfg.print_raw:
                print(f"  [grounding] score {cand.candidate_id} failed: {exc}")

    if not scores:
        return _grounding_empty(agent_result)

    if _should_trigger_visual_entity_repair(scores):
        repaired_hypothesis = _repair_entity_with_visual_candidates(
            client, cfg, question_text, hypothesis, scores,
        )
        if (
            repaired_hypothesis is not None
            and repaired_hypothesis.entity_name.strip()
            and repaired_hypothesis.entity_name.strip().lower()
            != hypothesis.entity_name.strip().lower()
        ):
            hypothesis = repaired_hypothesis
            entity_verification = {
                **(entity_verification or {}),
                "visual_repair": True,
                "visual_repair_entity_name": hypothesis.entity_name,
            }
            try:
                visual_desc = _search_visual_appearance(client, cfg, hypothesis)
            except Exception:
                visual_desc = visual_desc or "(no visual information available)"
            try:
                ref_images = _fetch_reference_images(cfg, hypothesis, artifact_dir, max_images=3)
            except Exception:
                ref_images = []
            direct_ground = _ground_entity_direct(
                client, cfg, image_path, hypothesis, question_text, visual_desc, ref_images,
            )
            candidates = _collect_entity_conditioned_candidates(
                client, cfg, image_path, question_text, hypothesis,
                visual_desc, ref_images, direct_ground,
            )
            if not candidates:
                return _grounding_empty(agent_result)
            scores = []
            for cand in candidates:
                try:
                    sc = _score_one_candidate(
                        client, cfg, image_path, cand, hypothesis,
                        question_text, artifact_dir, visual_desc, ref_images,
                    )
                    if direct_ground:
                        direct_box = Box(*[float(v) for v in direct_ground["bbox_xyxy"]]).normalize()
                        cand_box = Box(*[float(v) for v in cand.bbox_xyxy]).normalize()
                        overlap = compute_iou(direct_box, cand_box)
                        direct_ratio = _box_area_ratio(direct_ground["bbox_xyxy"], image_path)
                        sc["direct_ground_iou"] = round(overlap, 3)
                        if overlap >= 0.5 and direct_ratio < 0.22:
                            sc["final_score"] += 0.25
                        elif overlap >= 0.2 and direct_ratio < 0.16:
                            sc["final_score"] += 0.1
                    scores.append(sc)
                except Exception as exc:
                    if cfg.print_raw:
                        print(f"  [grounding] repaired score {cand.candidate_id} failed: {exc}")
            if not scores:
                return _grounding_empty(agent_result)

    # Phase 6: joint ranking plus direct-ground protection
    best = _select_best_grounding_candidate(
        client, cfg, image_path, artifact_dir, hypothesis,
        question_text, visual_desc, ref_images, scores, direct_ground,
    )
    pred_bbox = best["bbox_xyxy"]

    return {
        "task_type": "ground_from_question",
        "predicted_bbox": pred_bbox,
        "selected_candidate_id": best["candidate_id"],
        "confidence": best.get("confidence", 0.0),
        "resolved_entity": _safe_asdict(hypothesis),
        "visual_description": visual_desc,
        "candidate_scores": scores,
        "entity_verification": entity_verification,
        "joint_selection": {
            "candidate_id": best.get("candidate_id"),
            "joint_reason": best.get("joint_reason", ""),
            "joint_confidence": best.get("joint_confidence", 0.0),
            "runner_up_candidate_id": best.get("runner_up_candidate_id", ""),
            "selection_mode": best.get("selection_mode", ""),
            "guarded_over": best.get("guarded_over", ""),
        },
        "rounds_used": agent_result.rounds_used,
        "sub_questions": agent_result.sub_questions,
        "traces": [_safe_asdict(t) for t in agent_result.traces],
        "raw_evidence": agent_result.raw_evidence,
        "direct_grounding": direct_ground,
        "num_candidates": len(candidates),
        "num_ref_images": len(ref_images),
    }


def _safe_asdict(obj) -> Any:
    """Safe asdict that handles non-dataclass objects."""
    try:
        return asdict(obj)
    except Exception:
        return str(obj)


def _grounding_empty(agent_result: AgentResult) -> Dict[str, Any]:
    return {
        "task_type": "ground_from_question",
        "predicted_bbox": None,
        "selected_candidate_id": None,
        "confidence": 0.0,
        "resolved_entity": _safe_asdict(agent_result.resolved) if agent_result.resolved else None,
        "visual_description": "(no visual info)",
        "candidate_scores": [],
        "entity_verification": None,
        "joint_selection": None,
        "rounds_used": agent_result.rounds_used,
        "sub_questions": agent_result.sub_questions,
        "traces": [_safe_asdict(t) for t in agent_result.traces],
        "raw_evidence": agent_result.raw_evidence,
        "direct_grounding": None,
        "num_candidates": 0,
    }


def _score_one_candidate(
    client, cfg: Config, image_path: Path,
    cand: CandidateBBox, hypothesis: ResolvedEntity,
    reference_text: str, artifact_dir: Path,
    visual_description: str = "",
    ref_images: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Score a single visual candidate against the hypothesis."""
    cid = cand.candidate_id
    vis_dir = artifact_dir / "candidates"

    # render highlight and crop
    highlight_path = render_highlight(
        image_path, cand.bbox_xyxy, cid, vis_dir / f"{cid}_full.png"
    )
    crop_path = crop_candidate(
        image_path, cand.bbox_xyxy, vis_dir / f"{cid}_crop.png"
    )

    cues_str = "; ".join(hypothesis.key_cues) if hypothesis.key_cues else "none"
    prompt = _SCORE_CANDIDATE_PROMPT.format(
        reference_text=reference_text,
        entity_name=hypothesis.entity_name,
        visual_category=hypothesis.visual_category,
        key_cues=cues_str,
        candidate_id=cid,
        visual_description=visual_description or "(no visual info)",
    )

    score_images = [highlight_path, crop_path] + list(ref_images or [])[:2]
    raw = with_retry(
        lambda: call_llm_vision(
            client, cfg, prompt, score_images, max_tokens=512
        ),
        retries=cfg.retry_times,
    )
    if cfg.print_raw:
        print(f"=== SCORE {cid} RAW ===")
        print(raw)
        print()

    support, contra, conf, reason = 0.0, 0.0, 0.0, "parse_error"
    try:
        payload = extract_json(raw)
        support = max(0.0, min(5.0, float(payload.get("support_score", 0))))
        contra = max(0.0, min(5.0, float(payload.get("contradiction_score", 0))))
        conf = max(0.0, min(1.0, float(payload.get("confidence", 0))))
        reason = str(payload.get("reason", "")).strip()
    except Exception:
        # regex fallback
        sm = re.search(r"support_score[\":\s]+(\d+(?:\.\d+)?)", raw, re.I)
        cm = re.search(r"contradiction_score[\":\s]+(\d+(?:\.\d+)?)", raw, re.I)
        if sm:
            support = max(0.0, min(5.0, float(sm.group(1))))
        if cm:
            contra = max(0.0, min(5.0, float(cm.group(1))))

    ref_score = 0.0
    if ref_images:
        try:
            ref_score = _score_candidate_vs_references(
                client, cfg, ref_images, crop_path, hypothesis.entity_name,
            )
        except Exception as exc:
            if cfg.print_raw:
                print(f"  [grounding] ref score {cid} failed: {exc}")

    final_score = support - contra
    if ref_images:
        final_score += (ref_score - 2.5) * 0.45
        if ref_score <= 1.0:
            final_score -= 0.4

    return {
        "candidate_id": cid,
        "bbox_xyxy": cand.bbox_xyxy,
        "label": cand.label,
        "support_score": support,
        "contradiction_score": contra,
        "ref_match_score": round(ref_score, 3),
        "final_score": round(final_score, 4),
        "confidence": conf,
        "reason": reason,
    }


# ===================================================================
# Task 2: Choice — run_choice
# ===================================================================

def run_choice(
    client,
    cfg: Config,
    image_path: Path,
    bbox_xyxy: List[int],
    options: List[str],
    artifact_dir: Path,
) -> Dict[str, Any]:
    """
    Choice pipeline:
      1. For each option, mini-resolve to understand entity
      2. Render highlight + crop of given bbox
      3. LLM scores all options against the visual crop
      4. Return best option index
    """
    # Render views of the target bbox
    vis_dir = artifact_dir / "choice_vis"
    highlight_path = render_highlight(
        image_path, bbox_xyxy, "target", vis_dir / "target_full.png"
    )
    crop_path = crop_candidate(
        image_path, bbox_xyxy, vis_dir / "target_crop.png"
    )

    # Mini-resolve each option
    option_entities: List[ResolvedEntity] = []
    for opt in options:
        ent = mini_resolve_text(client, cfg, opt)
        option_entities.append(ent)

    # Build entity info summary
    entity_lines = []
    for i, (opt, ent) in enumerate(zip(options, option_entities)):
        entity_lines.append(
            f"Option {i}: entity={ent.entity_name}, "
            f"category={ent.visual_category}, cues={ent.key_cues}"
        )
    entity_info = "\n".join(entity_lines)

    # Build numbered options text
    options_text = "\n".join(
        f"  [{i}] {opt}" for i, opt in enumerate(options)
    )

    prompt = _SCORE_OPTION_PROMPT.format(
        options_text=options_text,
        entity_info=entity_info,
    )

    raw = with_retry(
        lambda: call_llm_vision(
            client, cfg, prompt, [highlight_path, crop_path], max_tokens=512
        ),
        retries=cfg.retry_times,
    )
    if cfg.print_raw:
        print("=== CHOICE SCORE RAW ===")
        print(raw)
        print()

    selected_index = None
    confidence = 0.0
    reason = ""
    parsed_result: Dict[str, Any] = {}
    try:
        payload = extract_json(raw)
        if isinstance(payload, dict):
            parsed_result = payload
        idx = payload.get("selected_index")
        if isinstance(idx, int) and 0 <= idx < len(options):
            selected_index = idx
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0))))
        reason = str(payload.get("reason", "")).strip()
    except Exception:
        # regex fallback
        m = re.search(r"selected_index[\":\s]+(\d+)", raw, re.I)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(options):
                selected_index = idx
        parsed_result = {
            "selected_index": selected_index,
            "confidence": confidence,
            "reason": reason,
        }

    return {
        "task_type": "choose_option_for_bbox",
        "selected_index": selected_index,
        "confidence": confidence,
        "reason": reason,
        "option_entities": [_safe_asdict(e) for e in option_entities],
        "num_options": len(options),
        "raw_output": raw,
        "result_payload": parsed_result,
    }

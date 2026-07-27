"""Quora fetch + parse.

The original spec was wrong about how this page works, in a way that would have cost
a Playwright build:

    "The content is server-rendered and sits in the DOM behind the modal... You do
     not need to defeat the modal, you need to read the DOM before it matters."

There is no DOM to read. Anonymous Quora ships ``<div id="root"></div>`` -- empty --
plus a ``<noscript>`` asking for JS. The content arrives as inline ``.push()`` calls
into ``window.ansFrontendGlobals.data.inlineQueryResults.results``, so there is no
modal to race and no browser needed. Plain httpx gets everything a headless Chrome
would.

The payload is **triple** JSON-encoded, which is the single most likely silent
breakage point in this codebase:

    1. The argument to .push(...) is a JSON *string*.
    2. Parsing it yields an object whose `data` holds the GraphQL result.
    3. The `title` / `content` fields inside are *themselves* JSON strings holding
       Quora's rich-text document.

Miss the third and you get a JSON blob scored as if it were prose. Hence
``tests/test_quora_parser.py`` and a checked-in fixture.

**Completeness.** A logged-out page carries roughly 3-4 answers regardless of how
many exist, and it returns HTTP 200 while doing so. Status code cannot tell you that
you fetched a third of the page. ``answer_count`` from og:description is the oracle:
without it, fetch success measures the wrong thing entirely.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

log = logging.getLogger(__name__)

# The .push() argument: a double-quoted JSON string literal, escapes included.
_PUSH_ARG = re.compile(r"\.push\(\s*(\"(?:[^\"\\]|\\.)*\")\s*\)")

# og:description reads "Answer (1 of 47): ..." on a question page. That 47 is the
# only signal of how much the logged-out gate is hiding from us.
_ANSWER_COUNT = re.compile(r"Answer\s+\(\d+\s+of\s+([\d,]+)\)", re.I)
_OG_DESCRIPTION = re.compile(
    r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
    re.I | re.S,
)
_TITLE_TAG = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


@dataclass
class QuoraAnswer:
    text: str
    upvotes: int = 0
    views: int = 0
    created_at: datetime | None = None
    is_machine_generated: bool = False


@dataclass
class QuoraPage:
    url: str
    question: str | None = None
    answers: list[QuoraAnswer] = field(default_factory=list)
    # How many answers Quora says exist, from og:description. None means we could not
    # read the oracle -- which is itself worth knowing, not a zero.
    answer_count: int | None = None

    @property
    def answers_seen(self) -> int:
        return len(self.answers)

    @property
    def completeness(self) -> float | None:
        """Fraction of existing answers we actually got. None when the oracle is unreadable.

        This is the number to watch, not fetch success rate. A 200 with 3 of 47
        answers is a successful fetch of an unrepresentative page.
        """
        if not self.answer_count:
            return None
        return min(1.0, self.answers_seen / self.answer_count)

    @property
    def body(self) -> str:
        """Question plus answers. For the scorer, which benefits from the full thread."""
        parts = [self.question] if self.question else []
        parts.extend(a.text for a in self.answers if a.text)
        return "\n\n".join(p for p in parts if p)

    @property
    def embed_text(self) -> str:
        """What the prefilter embeds. The QUESTION, not the body -- deliberately.

        On a real page the question is ~35 chars and the answers are ~13,800. Embedding
        ``body`` would produce a vector that is ~99.7% other people's *answers* and
        match leaves against solutions rather than against the pain. The question is
        the intent statement; that is the thing we are looking for.

        A lead-in of answer text is included only as disambiguating context for very
        short questions, capped so it can never dominate the vector.
        """
        if not self.question:
            return self.body[:2_000]
        context = ""
        if len(self.question) < 120 and self.answers:
            context = "\n\n" + self.answers[0].text[:600]
        return self.question + context

    @property
    def is_saturated(self) -> bool:
        """910 answers means the question is answered to death -- a dead lead.

        The spec wanted ``actionable`` to mean "recent and thinly answered, so there is
        still room to be useful". ``answer_count`` gives us the thinly-answered half
        directly, and for free, from the same og:description we already parse for the
        completeness oracle.
        """
        return bool(self.answer_count and self.answer_count > 20)

    @property
    def posted_at(self) -> datetime | None:
        """Earliest answer time, used as an upper bound on the question's age.

        Quora exposes no question creation time to logged-out clients. The oldest
        answer is the tightest bound available: the question is at least that old.
        Questions with zero answers -- the purest buying signal on the platform --
        are undateable entirely, which is a known and accepted gap.
        """
        times = [a.created_at for a in self.answers if a.created_at]
        return min(times) if times else None

    @property
    def engagement(self) -> int:
        return sum(a.upvotes for a in self.answers)


def is_challenge(html: str, status_code: int, headers: Mapping[str, str] | None = None) -> bool:
    """Cloudflare managed challenge rather than a real page.

    Measured against live Quora: challenge bodies ran 3.2-5.7KB with a
    ``cf-mitigated: challenge`` response header. A real question page is 220-260KB.
    The size gap is three orders of magnitude, so the classification is not delicate
    -- but check the header first, since it is the only signal Cloudflare states
    explicitly rather than one we infer.
    """
    if headers and headers.get("cf-mitigated") == "challenge":
        return True
    # Any non-200 is not a page, full stop. This used to enumerate `== 403`, which meant
    # 429 fell through to the size heuristic below -- and a 429 body is not always small.
    # Observed live: 429s at ~6KB (correctly rejected) and 429s large enough to clear the
    # 20KB bar (silently accepted, parsed, and saved as a real post with an empty body).
    # Enumerating statuses is the bug; the invariant is that a parseable page is a 200.
    if status_code != 200:
        return True
    # A real page is >100KB. Anything small enough to be an interstitial, that also
    # lacks the payload global, is not a page we can parse.
    if len(html) < 20_000 and "ansFrontendGlobals" not in html:
        return True
    return False


def fetch(url: str, timeout: float = 30.0, proxy: str | None = None) -> tuple[int, str, Mapping[str, str]]:
    """Fetch a Quora page with a real Chrome TLS fingerprint.

    ``curl_cffi``, not httpx/requests, and this is not interchangeable. Measured from
    one IP, back to back:

        httpx (OpenSSL TLS):      0 / 6 -> 403, every one cf-mitigated: challenge
        curl_cffi (Chrome TLS):   3 / 3 -> 200, full 220-260KB payload

    Same IP, same headers. The only variable is the TLS/HTTP2 fingerprint: httpx and
    requests delegate the handshake to OpenSSL and emit a JA3/JA4 shared by millions
    of bots, which Cloudflare scores directly. curl_cffi replays Chrome's actual
    handshake.

    The practical consequence is that Quora's gate is **fingerprint, not IP
    reputation**, so rotating IPs would not have fixed it and a residential proxy is
    not required here. ``proxy`` stays as an escape hatch in case that changes.
    """
    from curl_cffi import requests as cffi

    response = cffi.get(
        url,
        impersonate="chrome",
        timeout=timeout,
        proxies={"https": proxy, "http": proxy} if proxy else None,
    )
    return response.status_code, response.text, response.headers


def parse(html: str, url: str) -> QuoraPage:
    page = QuoraPage(url=url)
    page.answer_count = _extract_answer_count(html)
    page.question = _extract_question_from_title(html)

    seen: set[str] = set()
    for payload in _iter_inline_payloads(html):
        for node in _walk(payload):
            typename = node.get("__typename")
            if typename == "Question" and not page.question:
                text = _decode_rich_text(node.get("title"))
                if text:
                    page.question = text
            elif typename == "Answer":
                answer = _parse_answer(node)
                if answer and answer.text and answer.text not in seen:
                    seen.add(answer.text)
                    page.answers.append(answer)

    if page.answer_count and page.answers_seen > page.answer_count:
        # The oracle is a lower bound in practice; trust the parse but log the drift.
        log.debug("quora %s: parsed %d answers but og says %d", url, page.answers_seen, page.answer_count)
    return page


def _iter_inline_payloads(html: str) -> Iterator[dict[str, Any]]:
    """Yield decoded .push() payloads. Decode levels 1 and 2 of the three."""
    for match in _PUSH_ARG.finditer(html):
        raw = match.group(1)
        try:
            inner = json.loads(raw)          # level 1: the string literal
        except json.JSONDecodeError:
            continue
        if not isinstance(inner, str):
            continue
        try:
            payload = json.loads(inner)      # level 2: the object it holds
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    """Depth-first over every dict. The GraphQL shape nests unpredictably, so we do
    not hardcode a path -- a path would break on any backend refactor, silently."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _parse_answer(node: dict[str, Any]) -> QuoraAnswer | None:
    text = _decode_rich_text(node.get("content"))
    if not text:
        return None
    return QuoraAnswer(
        text=text,
        upvotes=_as_int(node.get("numUpvotes")),
        views=_as_int(node.get("numViews")),
        created_at=_as_datetime(node.get("creationTime")),
        # Quora's own flag. Semantics are inferred from their help-center wording and
        # not documented anywhere, so this is a ranking signal, not a hard filter --
        # see the plan's open questions.
        is_machine_generated=bool(node.get("isMachineAnswer", False)),
    )


def _decode_rich_text(value: Any) -> str | None:
    """Level 3: the field is itself a JSON string holding a rich-text document.

    Shape: {"sections": [{"spans": [{"text": "..."}]}]}. Concatenating spans is what
    turns it back into prose. If it is already plain text (Quora is not perfectly
    consistent), pass it through rather than dropping it.
    """
    if not value:
        return None
    if isinstance(value, dict):
        doc = value
    elif isinstance(value, str):
        try:
            doc = json.loads(value)
        except json.JSONDecodeError:
            stripped = value.strip()
            return stripped or None
        if not isinstance(doc, dict):
            return None
    else:
        return None

    parts: list[str] = []
    for section in doc.get("sections", []):
        if not isinstance(section, dict):
            continue
        spans = section.get("spans", [])
        line = "".join(s.get("text", "") for s in spans if isinstance(s, dict))
        if line.strip():
            parts.append(line.strip())
    return "\n".join(parts) or None


def _extract_answer_count(html: str) -> int | None:
    match = _OG_DESCRIPTION.search(html)
    if not match:
        return None
    count = _ANSWER_COUNT.search(match.group(1))
    if not count:
        return None
    return int(count.group(1).replace(",", ""))


def _extract_question_from_title(html: str) -> str | None:
    match = _TITLE_TAG.search(html)
    if not match:
        return None
    title = re.sub(r"\s*-\s*Quora\s*$", "", match.group(1).strip(), flags=re.I)
    return title or None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_datetime(value: Any) -> datetime | None:
    """Quora timestamps are microseconds since epoch."""
    if not value:
        return None
    try:
        micros = int(value)
    except (TypeError, ValueError):
        return None
    if micros <= 0:
        return None
    try:
        return datetime.fromtimestamp(micros / 1_000_000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None

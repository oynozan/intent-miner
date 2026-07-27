"""Quora parser tests, against real fetched HTML.

``quora_question.html`` is a live capture of https://www.quora.com/How-do-I-learn-Python
(256KB, curl_cffi + Chrome TLS). ``quora_challenge.html`` is a live Cloudflare
challenge from the same URL via httpx. Both are ground truth, not hand-written
approximations of what Quora might return -- the whole point of the fixture is that
the triple JSON encoding is too fiddly to mock convincingly.

When Quora changes their payload shape, these tests are how you find out. Re-capture
the fixture rather than loosening the assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scrape.quora import QuoraPage, _decode_rich_text, is_challenge, parse

FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://www.quora.com/How-do-I-learn-Python"


@pytest.fixture(scope="module")
def real_html() -> str:
    return (FIXTURES / "quora_question.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page(real_html: str) -> QuoraPage:
    return parse(real_html, URL)


def test_extracts_question(page: QuoraPage) -> None:
    assert page.question == "How should I start learning Python?"


def test_extracts_answers_with_prose_not_json(page: QuoraPage) -> None:
    """The failure this guards: miss the third decode level and every 'answer' is a
    JSON blob that embeds and scores as gibberish while looking populated."""
    assert page.answers, "no answers parsed from a real 256KB page"
    for answer in page.answers:
        assert not answer.text.lstrip().startswith(("{", "[")), "answer text is still JSON"
        assert '"sections"' not in answer.text
        assert '"spans"' not in answer.text
    assert len(page.body) > 5_000


def test_answer_metadata_parsed(page: QuoraPage) -> None:
    assert page.engagement > 0, "no upvotes parsed"
    assert any(a.views > 0 for a in page.answers)
    assert any(a.created_at is not None for a in page.answers)


def test_machine_answer_flag_is_read(page: QuoraPage) -> None:
    """isMachineAnswer is undocumented but present on live payloads. It is a ranking
    signal, not a filter -- so this asserts we read it, not that any given page has one."""
    assert all(isinstance(a.is_machine_generated, bool) for a in page.answers)


def test_posted_at_is_oldest_answer(page: QuoraPage) -> None:
    """Quora gives logged-out clients no question creation time. The oldest answer is
    the tightest available bound: the question is at least that old."""
    times = [a.created_at for a in page.answers if a.created_at]
    assert page.posted_at == min(times)


# --- The completeness oracle: the reason this parser reports two numbers ---

def test_answer_count_oracle_read_from_og_description(page: QuoraPage) -> None:
    """og:description reads 'Answer (1 of 910)'. Without this we cannot tell a
    complete fetch from a 0.4% fetch -- both are HTTP 200."""
    assert page.answer_count == 910


def test_logged_out_gate_truncates_massively(page: QuoraPage) -> None:
    """Measured reality: 4 answers of 910 on a real page.

    This is the finding that matters most for product framing. Quora serves logged-out
    clients a handful of answers regardless of how many exist, and returns 200 while
    doing it. Fetch success rate is therefore NOT a measure of whether we got the page.

    This test asserts the gate exists rather than pinning an exact ratio, because the
    ratio is Quora's to change.
    """
    assert page.answers_seen < 10
    assert page.answer_count is not None and page.answer_count > 100
    assert page.completeness is not None and page.completeness < 0.10


def test_embed_text_is_the_question_not_the_answer_pile(page: QuoraPage) -> None:
    """Guards the vector against being swamped by other people's answers.

    On this real page: question is 35 chars, body is ~13,800. Embedding the body would
    make the vector ~99.7% answers and match leaves against solutions instead of pain.
    """
    assert page.question is not None
    assert page.embed_text.startswith(page.question)
    assert len(page.embed_text) < 1_000
    assert len(page.embed_text) < len(page.body) / 5


def test_saturated_question_flagged_as_dead_lead(page: QuoraPage) -> None:
    """910 answers = no room to be useful. This is the 'thinly answered' half of the
    spec's `actionable`, available for free from the oracle we already parse."""
    assert page.is_saturated is True


def test_thinly_answered_question_not_saturated() -> None:
    html = '<html><meta property="og:description" content="Answer (1 of 2): foo" /></html>'
    assert parse(html, URL).is_saturated is False


def test_completeness_is_none_when_oracle_unreadable() -> None:
    """An unreadable oracle must be None, not 0.0 -- 'we don't know' and 'we got
    nothing' call for different handling."""
    page = parse("<html><title>Q - Quora</title></html>", URL)
    assert page.answer_count is None
    assert page.completeness is None


# --- Challenge detection ---

def test_real_cloudflare_challenge_detected() -> None:
    html = (FIXTURES / "quora_challenge.html").read_text(encoding="utf-8")
    assert is_challenge(html, 403, {"cf-mitigated": "challenge"})
    assert is_challenge(html, 403, None), "must detect on status alone"
    # 5744 bytes: the naive `len < 5000` heuristic would have missed this one.
    assert len(html) > 5_000
    assert is_challenge(html, 200, None), "must detect on shape when status lies"


def test_real_page_not_flagged_as_challenge(real_html: str) -> None:
    assert not is_challenge(real_html, 200, {"content-type": "text/html"})


def test_429_with_a_full_size_body_is_still_a_challenge() -> None:
    """is_challenge enumerated `status_code == 403`, so 429 fell through to the size
    heuristic -- and Quora's 429 page is NOT small. Measured live: 176,365 bytes AND
    containing `ansFrontendGlobals`, clearing both guards.

    It was then parsed into a real-looking post whose question was the literal string
    'Error 429 (Too Many Requests)', saved as fetch_status='ok', embedded and scored.
    29 such rows landed in one run; only the prefilter's semantic kill stopped them
    reaching the output, which is luck rather than a guard.

    Enumerating statuses was the bug. A parseable page is a 200.
    """
    big_body = "ansFrontendGlobals" + ("x" * 200_000)
    assert is_challenge(big_body, 429, {}), "429 is never a page, however large the body"
    assert is_challenge(big_body, 503, {})
    assert is_challenge(big_body, 302, {})
    assert not is_challenge(big_body, 200, {}), "a genuine 200 of this shape must pass"


def test_429_body_does_not_parse_into_a_usable_post() -> None:
    """Belt and braces: even if a 429 reached parse(), the result must not look like a
    real question. This is the shape that got saved 29 times."""
    html = "<html><title>Error 429 (Too Many Requests)</title>ansFrontendGlobals</html>"
    page = parse(html, "https://quora.com/x")
    assert page.answers_seen == 0
    assert not page.answer_count


# --- The triple decode, unit level ---

def test_decode_rich_text_walks_sections_and_spans() -> None:
    doc = json.dumps({"sections": [{"spans": [{"text": "Hello "}, {"text": "world"}]},
                                   {"spans": [{"text": "second line"}]}]})
    assert _decode_rich_text(doc) == "Hello world\nsecond line"


def test_decode_rich_text_passes_through_plain_text() -> None:
    """Quora is not perfectly consistent; a plain string must survive rather than vanish."""
    assert _decode_rich_text("just prose") == "just prose"


def test_decode_rich_text_handles_empty_and_none() -> None:
    assert _decode_rich_text(None) is None
    assert _decode_rich_text("") is None
    assert _decode_rich_text(json.dumps({"sections": []})) is None

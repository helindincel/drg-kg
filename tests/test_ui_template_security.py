from __future__ import annotations

from pathlib import Path


def test_query_results_escape_user_and_response_text_before_inner_html():
    template = Path("drg/api/templates/index.html").read_text(encoding="utf-8")

    assert "${query}</span>" not in template
    assert "${answerText}</p>" not in template
    assert "${error.message}" not in template
    assert "${escapeHtml(query)}</span>" in template
    assert "${escapeHtml(answerText)}</p>" in template
    assert "${escapeHtml(error.message)}" in template

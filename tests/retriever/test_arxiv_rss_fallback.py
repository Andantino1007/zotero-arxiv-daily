"""Offline regression tests for unavailable arXiv API metadata."""

from copy import deepcopy
import arxiv
import pytest
import requests
from omegaconf import OmegaConf

import zotero_arxiv_daily.retriever.arxiv_retriever as module


@pytest.fixture(autouse=True)
def neutral_selection(config):
    source = config.source.arxiv
    OmegaConf.set_struct(source, False)
    for key in ("priority_keywords", "strong_keywords", "keywords", "required_keywords", "exclude_keywords", "fallback_categories"):
        source[key] = []
    source.min_paper_num = 0
    source.include_cross_list = False


@pytest.fixture(autouse=True)
def no_network_or_delays(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("Regression tests must not use the network")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_request)
    monkeypatch.setattr(module, "sleep", lambda _: None)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)


def _http_error(status):
    return arxiv.HTTPError("https://export.arxiv.org/api/query", 0, status)


def _install_client(monkeypatch, responses):
    calls = []
    responses = iter(responses)

    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {"num_retries": 0, "delay_seconds": 10}

        def results(self, search):
            calls.append(search.id_list)
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return iter(response)

    monkeypatch.setattr(module.arxiv, "Client", Client)
    return calls


def _entries(feed):
    return [entry for entry in feed.entries if entry.arxiv_announce_type == "new"]


def _many_entries(feed, count):
    template = _entries(feed)[0]
    feed["entries"] = []
    for index in range(count):
        entry = deepcopy(template)
        entry["id"] = f"oai:arXiv.org:2609.{30000 + index}v1"
        feed.entries.append(entry)


def test_rss_metadata_preserves_abstract_authors_categories_and_links(mock_feedparser):
    result = module._result_from_rss(_entries(mock_feedparser)[0])
    assert result.title == "Neural Architecture Search for Efficient Transformers"
    assert result.summary == "We propose a neural architecture search method for efficient transformers."
    assert [author.name for author in result.authors] == ["Alice Smith", "Bob Jones"]
    assert result.categories == ["cs.AI", "cs.LG"]
    assert result.primary_category == "cs.AI"
    assert result.entry_id == "https://arxiv.org/abs/2508.14001v1"
    assert result.pdf_url == "https://arxiv.org/pdf/2508.14001v1"
    assert result.source_url() == "https://arxiv.org/src/2508.14001v1"
    assert result.published.year == 2025


def test_html_summary_is_plain_text(mock_feedparser):
    entry = _entries(mock_feedparser)[0]
    entry["summary_detail"]["type"] = "text/html"
    entry["summary"] = "<p>arXiv:2508.14001v1 Announce Type: new</p><p>Abstract: A &amp; B <b>method</b>.</p>"
    result = module._result_from_rss(entry)
    assert "A & B" in result.summary
    assert "<b>" not in result.summary
    assert "Announce Type" not in result.summary


@pytest.mark.parametrize("field", ["id", "title", "summary"])
def test_incomplete_rss_is_not_silently_dropped(mock_feedparser, field):
    entry = _entries(mock_feedparser)[0]
    entry[field] = ""
    with pytest.raises(ValueError, match="Incomplete arXiv RSS metadata"):
        module._result_from_rss(entry)


def test_406_switches_all_remaining_batches_to_rss(config, mock_feedparser, monkeypatch):
    _many_entries(mock_feedparser, 45)
    config.source.arxiv.extract_full_text = False
    calls = _install_client(monkeypatch, [_http_error(406)])
    papers = module.ArxivRetriever(config).retrieve_papers()
    assert len(papers) == 45
    assert len({paper.url for paper in papers}) == 45
    assert len(calls) == 1
    assert all(paper.authors == ["Alice Smith", "Bob Jones"] for paper in papers)
    assert all(paper.abstract and paper.pdf_url and paper.full_text is None for paper in papers)


def test_later_failure_keeps_successful_api_batch(config, mock_feedparser, monkeypatch):
    _many_entries(mock_feedparser, 45)
    first = [module._result_from_rss(entry) for entry in mock_feedparser.entries[:20]]
    first[0].title = "Enriched API title"
    calls = _install_client(monkeypatch, [first, _http_error(406)])
    papers = module.ArxivRetriever(config)._retrieve_raw_papers()
    assert len(papers) == 45
    assert len({paper.entry_id for paper in papers}) == 45
    assert papers[0].title == "Enriched API title"
    assert len(calls) == 2


@pytest.mark.parametrize("include_cross_list", [False, True])
def test_fallback_keeps_announcement_selection(config, mock_feedparser, monkeypatch, include_cross_list):
    config.source.arxiv.include_cross_list = include_cross_list
    _install_client(monkeypatch, [_http_error(406)])
    papers = module.ArxivRetriever(config)._retrieve_raw_papers()
    expected = mock_feedparser.entries if include_cross_list else _entries(mock_feedparser)
    assert {paper.title for paper in papers} == {entry.title for entry in expected}


def test_fallback_keeps_debug_limit(config, mock_feedparser, monkeypatch):
    _many_entries(mock_feedparser, 45)
    config.executor.debug = True
    calls = _install_client(monkeypatch, [_http_error(406)])
    assert len(module.ArxivRetriever(config)._retrieve_raw_papers()) == 10
    assert len(calls[0]) == 10


def test_fallback_keeps_keyword_and_sparse_result_filters(config, mock_feedparser, monkeypatch):
    source = config.source.arxiv
    OmegaConf.set_struct(source, False)
    source.keywords = ["neural"]
    source.required_keywords = ["transformers", "multi-agent"]
    source.min_paper_num = 2
    source.exclude_keywords = ["reward"]
    _install_client(monkeypatch, [_http_error(406)])
    papers = module.ArxivRetriever(config)._retrieve_raw_papers()
    assert [paper.title for paper in papers] == [_entries(mock_feedparser)[0].title]
    source.exclude_keywords = []
    _install_client(monkeypatch, [_http_error(406)])
    papers = module.ArxivRetriever(config)._retrieve_raw_papers()
    assert len(papers) == 2


@pytest.mark.parametrize("error", [_http_error(429), _http_error(503), requests.Timeout("timeout")])
def test_transient_failure_has_bounded_retries(config, mock_feedparser, monkeypatch, error):
    calls = _install_client(monkeypatch, [error, error, error])
    waits = []
    monkeypatch.setattr(module, "sleep", waits.append)
    papers = module.ArxivRetriever(config)._retrieve_raw_papers()
    assert len(papers) == len(_entries(mock_feedparser))
    assert len(calls) == 3
    assert waits == [30, 60]


def test_transient_failure_can_recover(config, mock_feedparser, monkeypatch):
    expected = [module._result_from_rss(entry) for entry in _entries(mock_feedparser)]
    calls = _install_client(monkeypatch, [_http_error(429), expected])
    papers = module.ArxivRetriever(config)._retrieve_raw_papers()
    assert papers == expected
    assert len(calls) == 2


def test_unexpected_http_errors_are_not_hidden(config, mock_feedparser, monkeypatch):
    calls = _install_client(monkeypatch, [_http_error(400)])
    with pytest.raises(arxiv.HTTPError):
        module.ArxivRetriever(config)._retrieve_raw_papers()
    assert len(calls) == 1


def test_empty_api_response_uses_rss(config, mock_feedparser, monkeypatch):
    _install_client(monkeypatch, [[]])
    assert len(module.ArxivRetriever(config)._retrieve_raw_papers()) == len(_entries(mock_feedparser))


def test_partial_api_response_does_not_lose_papers(config, mock_feedparser, monkeypatch):
    entries = _entries(mock_feedparser)
    _install_client(monkeypatch, [[module._result_from_rss(entries[0])]])
    papers = module.ArxivRetriever(config)._retrieve_raw_papers()
    assert {paper.title for paper in papers} == {entry.title for entry in entries}


@pytest.mark.parametrize("failure", [{"status": 503}, {"bozo": True}, {"feed": {}}])
def test_broken_feed_is_not_reported_as_no_papers(config, mock_feedparser, failure):
    mock_feedparser.update(failure)
    with pytest.raises(RuntimeError, match="Failed to read arXiv RSS"):
        module.ArxivRetriever(config)._retrieve_raw_papers()


def test_valid_empty_feed_does_not_call_api(config, mock_feedparser, monkeypatch):
    mock_feedparser["entries"] = []
    calls = _install_client(monkeypatch, [])
    assert module.ArxivRetriever(config)._retrieve_raw_papers() == []
    assert calls == []

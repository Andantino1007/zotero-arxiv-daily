from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
from calendar import timegm
from datetime import datetime, timezone
from html.parser import HTMLParser
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def _normalized_keywords(values: Any) -> list[str]:
    if values is None:
        return []
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


class _RSSPlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _result_from_rss(entry: Any) -> ArxivResult:
    paper_id = entry.get("id", "").removeprefix("oai:arXiv.org:")
    title = entry.get("title", "").strip()
    summary = entry.get("summary", "").strip()
    if entry.get("summary_detail", {}).get("type") in {"text/html", "application/xhtml+xml"}:
        parser = _RSSPlainText()
        parser.feed(summary)
        summary = " ".join(parser.parts).strip()
    # RSS summaries prepend the identifier and announcement type to the abstract.
    if summary.startswith("arXiv:") and "Abstract:" in summary:
        summary = summary.partition("Abstract:")[2].strip()
    if not paper_id or not title or not summary:
        raise ValueError(f"Incomplete arXiv RSS metadata for {paper_id or 'unknown paper'}")
    categories = [tag["term"] for tag in entry.get("tags", []) if tag.get("term")]
    authors = [
        ArxivResult.Author(name.strip())
        for author in entry.get("authors", [])
        for name in author.get("name", "").split(",")
        if name.strip()
    ]
    dates = {
        field: datetime.fromtimestamp(timegm(entry[f"{field}_parsed"]), timezone.utc)
        for field in ("published", "updated")
        if entry.get(f"{field}_parsed")
    }
    return ArxivResult(
        entry_id=f"https://arxiv.org/abs/{paper_id}",
        title=title,
        summary=summary,
        authors=authors,
        categories=categories,
        primary_category=categories[0] if categories else "",
        links=[ArxivResult.Link(
            href=f"https://arxiv.org/pdf/{paper_id}", title="pdf",
            rel="related", content_type="application/pdf",
        )],
        **dates,
    )


def _retrieve_api_batch(client: arxiv.Client, paper_ids: list[str]) -> list[ArxivResult] | None:
    search = arxiv.Search(id_list=paper_ids, max_results=len(paper_ids))
    for attempt in range(3):
        try:
            batch = list(client.results(search))
            if len(batch) != len(paper_ids):
                logger.warning(
                    f"arXiv API returned {len(batch)}/{len(paper_ids)} papers; using RSS metadata"
                )
                return None
            return batch
        except (arxiv.HTTPError, arxiv.UnexpectedEmptyPageError, requests.exceptions.RequestException) as exc:
            status = exc.status if isinstance(exc, arxiv.HTTPError) else None
            if status is not None and status not in {406, 429} and not 500 <= status < 600:
                raise
            logger.warning(f"arXiv API batch failed (attempt {attempt + 1}/3): {exc}")
            # A rejected request should not be repeated for every batch in the feed.
            if status == 406 or attempt == 2:
                return None
            sleep(30 * (2 ** attempt))
    return None


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
        self.extract_full_text = self.config.source.arxiv.get("extract_full_text", True)
        self.priority_keywords = _normalized_keywords(self.config.source.arxiv.get("priority_keywords", []))
        self.strong_keywords = _normalized_keywords(self.config.source.arxiv.get("strong_keywords", []))
        self.keywords = _normalized_keywords(self.config.source.arxiv.get("keywords", []))
        self.required_keywords = _normalized_keywords(self.config.source.arxiv.get("required_keywords", []))
        self.exclude_keywords = _normalized_keywords(self.config.source.arxiv.get("exclude_keywords", []))
        self.fallback_categories = _normalized_keywords(self.config.source.arxiv.get("fallback_categories", []))
        self.min_paper_num = int(self.config.source.arxiv.get("min_paper_num", 0) or 0)

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        # Retry here only, rather than multiplying client retries by batch retries.
        client = arxiv.Client(num_retries=0, delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if feed.get("status", 200) >= 400 or feed.get("bozo") or not feed.feed.get("title"):
            raise RuntimeError(f"Failed to read arXiv RSS feed for {query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        fallback_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        entries = [
            i
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            entries = entries[:10]

        # Get full information of each paper from arxiv api
        use_rss = False
        with tqdm(total=len(entries)) as bar:
            for i in range(0, len(entries), 20):
                batch_entries = entries[i:i + 20]
                batch = None
                if not use_rss:
                    batch = _retrieve_api_batch(
                        client, [entry.id.removeprefix("oai:arXiv.org:") for entry in batch_entries]
                    )
                    if batch is None:
                        use_rss = True
                        logger.warning(
                            f"Using arXiv RSS metadata for the remaining {len(entries) - i} papers "
                            "because the API is unavailable"
                        )
                if use_rss:
                    batch = [_result_from_rss(entry) for entry in batch_entries]
                bar.update(len(batch))
                filtered_batch = [paper for paper in batch if self._matches_keywords(paper)]
                fallback_papers.extend([paper for paper in batch if self._matches_fallback_candidate(paper)])
                if self.priority_keywords or self.strong_keywords or self.keywords or self.required_keywords or self.exclude_keywords:
                    logger.info(
                        f"Keyword filter kept {len(filtered_batch)}/{len(batch)} arXiv papers "
                        f"in batch {i // 20}"
                    )
                raw_papers.extend(filtered_batch)
                if not use_rss and i + 20 < len(entries):
                    sleep(3)

        raw_papers = self._expand_sparse_results(raw_papers, fallback_papers)
        return raw_papers

    def _expand_sparse_results(self, raw_papers: list[ArxivResult], fallback_papers: list[ArxivResult]) -> list[ArxivResult]:
        if self.min_paper_num <= 0 or len(raw_papers) >= self.min_paper_num:
            return raw_papers

        before_count = len(raw_papers)
        seen_ids = {paper.entry_id for paper in raw_papers}
        fallback_papers = sorted(fallback_papers, key=self._fallback_score, reverse=True)
        for paper in fallback_papers:
            if paper.entry_id in seen_ids:
                continue
            raw_papers.append(paper)
            seen_ids.add(paper.entry_id)
            if len(raw_papers) >= self.min_paper_num:
                break

        if len(raw_papers) > before_count:
            logger.info(
                f"Expanded sparse keyword results from {before_count} to {len(raw_papers)} "
                f"papers using mechanics fallback candidates before reranking"
            )
        return raw_papers

    def _matches_keywords(self, raw_paper: ArxivResult) -> bool:
        searchable_text = self._searchable_text(raw_paper)
        if self.exclude_keywords and _contains_any(searchable_text, self.exclude_keywords):
            return False

        has_required = self.required_keywords and _contains_any(searchable_text, self.required_keywords)
        has_priority = self.priority_keywords and _contains_any(searchable_text, self.priority_keywords)
        has_strong = self.strong_keywords and _contains_any(searchable_text, self.strong_keywords)
        has_keyword = (self.keywords and _contains_any(searchable_text, self.keywords)) or has_priority

        # Tight gate: even strong AI/SciML terms need a mechanics anchor.
        if has_required and (has_priority or has_strong):
            return True
        if self.keywords and not has_keyword:
            return False
        if self.required_keywords and not has_required:
            return False

        return bool(has_keyword and has_required) or not (self.priority_keywords or self.strong_keywords or self.keywords or self.required_keywords)

    def _matches_fallback_candidate(self, raw_paper: ArxivResult) -> bool:
        searchable_text = self._searchable_text(raw_paper)
        if self.exclude_keywords and _contains_any(searchable_text, self.exclude_keywords):
            return False

        # Fallback is stricter than before: category-only papers no longer pass.
        # They need an actual mechanics anchor in title/abstract/category text.
        if not self.required_keywords or not _contains_any(searchable_text, self.required_keywords):
            return False

        paper_categories = self._paper_categories(raw_paper)
        if not self.fallback_categories:
            return True
        return any(category in paper_categories for category in self.fallback_categories) or self._fallback_score(raw_paper) > 0

    def _fallback_score(self, raw_paper: ArxivResult) -> int:
        searchable_text = self._searchable_text(raw_paper)
        score = 0
        score += 80 * sum(1 for keyword in self.priority_keywords if keyword in searchable_text)
        score += 25 * sum(1 for keyword in self.strong_keywords if keyword in searchable_text)
        score += 18 * sum(1 for keyword in self.required_keywords if keyword in searchable_text)
        score += 6 * sum(1 for keyword in self.keywords if keyword in searchable_text)
        paper_categories = self._paper_categories(raw_paper)
        if any(category in paper_categories for category in self.fallback_categories):
            score += 1
        return score

    def _paper_categories(self, raw_paper: ArxivResult) -> set[str]:
        paper_categories = {category.lower() for category in (getattr(raw_paper, "categories", []) or [])}
        primary_category = (getattr(raw_paper, "primary_category", "") or "").lower()
        if primary_category:
            paper_categories.add(primary_category)
        return paper_categories

    def _searchable_text(self, raw_paper: ArxivResult) -> str:
        categories = " ".join(getattr(raw_paper, "categories", []) or [])
        primary_category = getattr(raw_paper, "primary_category", "") or ""
        return "\n".join(
            [
                raw_paper.title or "",
                raw_paper.summary or "",
                primary_category,
                categories,
            ]
        ).lower()

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = None
        if self.extract_full_text:
            full_text = extract_text_from_tar(raw_paper)
            if full_text is None:
                full_text = extract_text_from_html(raw_paper)
            if full_text is None:
                full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )

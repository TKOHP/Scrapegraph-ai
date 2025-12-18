"""
多源论文检索与PDF抓取脚本
面向关键词“fintech survey”，在 arxiv / Google Scholar / Google 三个来源中检索，
自动提取并下载 PDF 到以时间戳命名的本地文件夹。
"""

import os
import re
import tempfile
from typing import List, Optional, Iterable, Tuple
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime

import requests
from scrapegraphai.graphs import SearchGraph
from langchain_openai import ChatOpenAI


def build_llm() -> ChatOpenAI:
    """
    构建兼容 ChatOpenAI 接口的 LLM 客户端

    默认使用阿里 DashScope 兼容接口（Qwen-* 系列），避免硬编码密钥。
    请在系统环境中配置 `DASHSCOPE_API_KEY` 或自行替换为其他兼容提供商。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-cd7b54e0eaf5444ea29c71dc2cea3731")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=base_url,
        model=model,
    )


def _extract_hrefs(html: str) -> List[str]:
    """
    从 HTML 文本中提取所有 href 链接（简易正则）

    返回值为原始链接列表，不做去重与补全。
    """
    return re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)


def _is_pdf_url(url: str) -> bool:
    """
    粗略判断链接是否为 PDF（根据路径后缀与常见参数形式）
    """
    return bool(re.search(r"\.pdf($|[#?])", url, flags=re.IGNORECASE))


def _fetch_pdf_links_from_page(page_url: str, timeout: int = 20) -> List[str]:
    """
    请求页面并尽力提取 PDF 链接：
    1) 若响应为 PDF（Content-Type 包含 application/pdf），直接认定为 PDF；
    2) 否则解析 HTML 中的 href，筛选出以 .pdf 结尾的链接，并做相对路径的补全。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/pdf,application/octet-stream,*/*",
    }
    try:
        r = requests.get(page_url, timeout=timeout, headers=headers, allow_redirects=True)
        ct = r.headers.get("Content-Type", "")
        if "application/pdf" in ct:
            return [page_url]
        if not r.text:
            return []
        pdfs = []
        for href in set(_extract_hrefs(r.text)):
            if _is_pdf_url(href):
                pdfs.append(urljoin(page_url, href))
        return list(dict.fromkeys(pdfs))
    except Exception:
        return []


def _run_search_graph(prompt: str, *, engine: str, max_results: int, llm: ChatOpenAI) -> List[str]:
    """
    使用 SearchGraph 执行检索并返回图运行期间考虑的候选 URL 列表。

    通过传入 `model_instance` 与 `model_tokens` 复用外部 LLM 客户端，避免硬编码密钥。
    """
    graph_config = {
        "llm": {
            "model_instance": llm,
            "model_tokens": 8192,
        },
        "search_engine": engine,
        "max_results": max_results,
        "verbose": False,
        "headless": True,
        # 关键：为 ChromiumLoader 传入稳定抓取参数，减少“页面正在导航”错误
        "loader_kwargs": {
            "timeout": 45,
            "retry_limit": 3,
            "load_state": "networkidle",
            "requires_js_support": True,
            "browser_name": "chromium",
        },
    }
    sg = SearchGraph(prompt=prompt, config=graph_config)
    sg.run()
    return sg.get_considered_urls()


def _merge_unique(url_lists: Iterable[Iterable[str]]) -> List[str]:
    """
    合并多个 URL 列表并保持顺序去重。
    """
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def search_fintech_survey_sources(llm: ChatOpenAI, max_results: int = 20) -> List[str]:
    """
    面向关键词“fintech survey”，在三类来源上检索候选页面：
    - arxiv: 使用 `site:arxiv.org` 约束；
    - Google Scholar: 使用 `site:scholar.google.com` 约束；
    - Google: 使用 `filetype:pdf` 强化检索；

    为提升覆盖面，分别调用 DuckDuckGo 与 Bing 两种搜索引擎，并做结果去重。
    """
    prompt = build_search_prompt()
    engines = ["duckduckgo", "bing"]

    collected: List[List[str]] = []
    for eng in engines:
        try:
            urls = _run_search_graph(prompt, engine=eng, max_results=max_results, llm=llm)
            collected.append(urls)
        except Exception:
            pass

    direct = direct_search_sources("fintech survey", max_results=max_results)
    collected.append(direct)
    return _merge_unique(collected)


def build_search_prompt() -> str:
    """
    构建用于 SearchGraph 的检索提示词。

    目标来源包括 arxiv、Google Scholar 与 Google 普通检索，并偏好直接返回 PDF 或可到达 PDF 的详情页。
    """
    return (
        "请在公开网络中检索与“fintech survey”主题直接相关的论文页面，并优先返回 PDF 链接或指向 PDF 的详情页。\n"
        "目标来源：arxiv.org（论文页或PDF）、scholar.google.com（论文详情）、Google（使用 filetype:pdf 约束）。\n"
        "关键词建议：fintech survey; financial technology; literature review; overview。\n"
        "输出：请返回若干最具代表性的页面 URL，用于后续抓取与下载。"
    )


def direct_search_sources(query: str, max_results: int = 20) -> List[str]:
    """
    直接分别在 arxiv、Google Scholar、Google 网站进行搜索并汇总返回 URL 列表。
    """
    urls: List[str] = []
    try:
        urls += search_arxiv_pdfs(query, max_results=max_results)
    except Exception:
        pass
    try:
        urls += search_scholar_pdfs(query, max_results=max_results)
    except Exception:
        pass
    try:
        urls += search_google_pdfs(query, max_results=max_results)
    except Exception:
        pass
    return list(dict.fromkeys(urls))


def search_arxiv_pdfs(query: str, max_results: int = 20) -> List[str]:
    """
    在 arxiv 网站检索并提取 PDF 链接。
    """
    base = "https://arxiv.org"
    q = requests.utils.quote(query)
    url = f"{base}/search/?query={q}&searchtype=all&abstracts=show&order=-announced_date_first&size={max_results}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,*/*",
    }
    resp = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
    resp.raise_for_status()
    html = resp.text
    pdfs: List[str] = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if href.startswith("/pdf/") and href.lower().endswith(".pdf"):
                pdfs.append(urljoin(base, href))
    except Exception:
        for href in set(_extract_hrefs(html)):
            if href.startswith("/pdf/") and href.lower().endswith(".pdf"):
                pdfs.append(urljoin(base, href))
    return pdfs[:max_results]


def search_scholar_pdfs(query: str, max_results: int = 20) -> List[str]:
    """
    在 Google Scholar 检索并提取页面中的 PDF 链接。
    """
    q = requests.utils.quote(query + " literature review")
    url = f"https://scholar.google.com/scholar?q={q}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,*/*",
    }
    resp = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
    resp.raise_for_status()
    html = resp.text
    pdfs: List[str] = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("div.gs_or_ggsm a[href]"):
            href = a.get("href", "")
            if _is_pdf_url(href):
                pdfs.append(href)
        if not pdfs:
            for a in soup.select("a[href]"):
                href = a.get("href", "")
                if _is_pdf_url(href):
                    pdfs.append(href)
    except Exception:
        for href in set(_extract_hrefs(html)):
            if _is_pdf_url(href):
                pdfs.append(href)
    return pdfs[:max_results]


def search_google_pdfs(query: str, max_results: int = 20) -> List[str]:
    """
    在 Google 网站检索 `filetype:pdf` 并解析出 PDF 链接。
    若存在 `SERPER_API_KEY`，优先使用 Serper 接口以提高稳定性。
    """
    api_key = os.getenv("SERPER_API_KEY")
    if api_key:
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        data = {"q": f"{query} filetype:pdf", "num": max_results}
        resp = requests.post("https://google.serper.dev/search", json=data, headers=headers, timeout=30)
        resp.raise_for_status()
        js = resp.json()
        urls = [item.get("link") for item in js.get("organic", []) if item.get("link", "").lower().endswith(".pdf")]
        return urls[:max_results]

    q = requests.utils.quote(query + " filetype:pdf")
    url = f"https://www.google.com/search?q={q}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,*/*",
    }
    resp = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
    resp.raise_for_status()
    html = resp.text
    pdfs: List[str] = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if href.startswith("/url?"):
                qs = parse_qs(urlparse(href).query)
                target = qs.get("q", [""])[0]
                if _is_pdf_url(target):
                    pdfs.append(target)
            elif _is_pdf_url(href):
                pdfs.append(href)
    except Exception:
        for href in set(_extract_hrefs(html)):
            if href.startswith("/url?"):
                qs = parse_qs(urlparse(href).query)
                target = qs.get("q", [""])[0]
                if _is_pdf_url(target):
                    pdfs.append(target)
            elif _is_pdf_url(href):
                pdfs.append(href)
    return list(dict.fromkeys(pdfs))[:max_results]


def harvest_pdf_links(candidates: Iterable[str], timeout: int = 20) -> List[str]:
    """
    依据候选页面集合提取 PDF 链接：
    - 直接检查每个候选是否为 PDF（响应头或后缀）；
    - 如为普通 HTML 页面，则在页面中解析并抽取 PDF 链接。
    """
    pdfs: List[str] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*",
    }
    for url in candidates:
        # 先尝试通过请求头快速判定
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True, headers=headers)
            ct = r.headers.get("Content-Type", "")
            if "application/pdf" in ct:
                pdfs.append(url)
                continue
        except Exception:
            pass

        # 后缀判定或页面解析
        if _is_pdf_url(url):
            pdfs.append(url)
            continue

        extracted = _fetch_pdf_links_from_page(url, timeout=timeout)
        pdfs.extend(extracted)

    # 去重保持顺序
    return list(dict.fromkeys(pdfs))


def create_timestamp_folder(base_dir: str) -> str:
    """
    在指定基准目录下创建以时间戳命名的文件夹并返回路径。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, ts)
    os.makedirs(out_dir, exist_ok=False)
    return out_dir


def download_pdf_to_dir(url: str, out_dir: str, filename: Optional[str] = None) -> str:
    """
    下载单个 PDF 到指定目录并返回保存路径。

    参数：
    - url: PDF 链接
    - out_dir: 目标保存目录（需存在）
    - filename: 可选文件名，若未提供则根据 URL 自动生成
    """
    os.makedirs(out_dir, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*",
    }
    resp = requests.get(url, timeout=45, headers=headers, allow_redirects=True)
    resp.raise_for_status()

    if not filename:
        path_part = urlparse(url).path
        base = os.path.basename(path_part) or "paper.pdf"
        if not base.lower().endswith(".pdf"):
            base += ".pdf"
        filename = base

    save_path = os.path.join(out_dir, filename)
    if os.path.exists(save_path):
        name, ext = os.path.splitext(filename)
        idx = 2
        while True:
            candidate = os.path.join(out_dir, f"{name}_{idx}{ext}")
            if not os.path.exists(candidate):
                save_path = candidate
                break
            idx += 1

    with open(save_path, "wb") as f:
        f.write(resp.content)
    return save_path


def download_pdfs(urls: List[str], out_dir: str, max_count: int) -> List[str]:
    """
    批量下载前 `max_count` 个 PDF 到 `out_dir` 并返回本地路径列表。
    文件名使用序号前缀以提高可读性与稳定性。
    """
    saved: List[str] = []
    for i, u in enumerate(urls[:max_count], start=1):
        path_part = urlparse(u).path
        base = os.path.basename(path_part)
        if base and base.lower().endswith(".pdf"):
            fname = f"{i:03d}_{base}"
        else:
            fname = f"{i:03d}_paper.pdf"
        try:
            saved_path = download_pdf_to_dir(u, out_dir, fname)
            saved.append(saved_path)
        except Exception:
            continue
    return saved


def collect_fintech_survey_pdfs(n: int, base_dir: str) -> Tuple[str, List[str]]:
    """
    核心入口：检索“fintech survey”并下载前 N 个 PDF。

    过程：
    1) 构建 LLM；
    2) 多源检索候选页面；
    3) 从候选中提取 PDF 链接；
    4) 创建时间戳目录并下载文件。
    返回：时间戳目录与成功下载的本地路径列表。
    """
    llm = build_llm()
    candidates = search_fintech_survey_sources(llm, max_results=max(20, n * 4))
    pdf_links = harvest_pdf_links(candidates, timeout=25)
    if not pdf_links:
        # 回退策略：增强查询词
        more_candidates = search_fintech_survey_sources(
            llm, max_results=max(30, n * 5)
        )
        pdf_links = harvest_pdf_links(more_candidates, timeout=25)
    if not pdf_links:
        raise RuntimeError("未检索到 PDF 链接")

    out_dir = create_timestamp_folder(base_dir)
    saved_paths = download_pdfs(pdf_links, out_dir, max_count=n)
    return out_dir, saved_paths


def main() -> None:
    """
    示例入口：执行检索并抓取指定数量的 PDF 到时间戳目录。
    可根据需求调整数量与保存目录。
    """
    n = 5
    base_dir = r"d:\workproject\Weekly\Scrapegraph-ai\examples\mycode"
    out_dir, saved = collect_fintech_survey_pdfs(n, base_dir)
    print(f"已创建时间戳目录: {out_dir}")
    print(f"成功下载 {len(saved)} 个PDF")
    for p in saved:
        print(f"- {p}")


if __name__ == "__main__":
    main()

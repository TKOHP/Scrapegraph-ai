"""
仅从 Google Scholar 检索“fintech survey”并下载 PDF 的示例脚本
核心使用 SearchGraph 获取候选 Scholar 链接，然后解析并下载 PDF 到以时间戳命名的目录。
"""

import os
import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urljoin, parse_qs
from datetime import datetime

import requests
from scrapegraphai.graphs import SearchGraph, SearchLinkGraph
from scrapegraphai.docloaders import ChromiumLoader
from langchain_openai import ChatOpenAI


def build_llm() -> ChatOpenAI:
    """
    构建兼容 ChatOpenAI 接口的 LLM 客户端

    默认使用阿里 DashScope 兼容接口（Qwen-* 系列），通过环境变量读取密钥：
    - DASHSCOPE_API_KEY
    - DASHSCOPE_MODEL（可选，默认 qwen-plus）
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-cd7b54e0eaf5444ea29c71dc2cea3731")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=base_url,
        model=model,
    )


def build_scholar_prompt(query: str) -> str:
    """
    构建 Google Scholar 定向检索提示词

    强调仅在 `scholar.google.com` 检索与 query 相关的文献页面，并优先返回包含 PDF 的链接。
    """
    return (
        f"请仅在 Google Scholar (scholar.google.com) 检索与“{query}”主题直接相关的文献页面，"
        "优先返回包含 PDF 的外链或条目详情链接。\n"
        "请避免返回任何非 Scholar 域名的链接，严格限定域为 scholar.google.com。\n"
        "查询格式建议：site:scholar.google.com fintech survey literature review。\n"
        "输出：请返回若干 Scholar 结果页面的 URL（以用于后续解析与下载 PDF）。"
    )


def build_scholar_search_url(query: str) -> str:
    """
    构造 Google Scholar 的检索页面 URL。
    使用 `literature review` 强化检索相关性。
    """
    q = requests.utils.quote(f"{query} literature review")
    return f"https://scholar.google.com/scholar?q={q}"


def run_scholar_link_graph(llm: ChatOpenAI, source_url: str, max_results: int = 20) -> List[str]:
    """
    使用 SearchLinkGraph 抓取 `source_url`（Scholar 检索页），并返回同域（scholar.google.*）的链接列表。

    通过 `filter_links=True` 与默认 `diff_domain_filter=True` 实现链接域限定；
    通过 `loader_kwargs` 配置浏览器行为以减少导航错误。
    """
    graph_config = {
        "llm": {
            "model_instance": llm,
            "model_tokens": 8192,
        },
        "verbose": False,
        "headless": True,
        "filter_config": {},
        "loader_kwargs": {
            "timeout": 45,
            "retry_limit": 3,
            "load_state": "networkidle",
            "requires_js_support": True,
            "browser_name": "chromium",
        },
    }
    graph = SearchLinkGraph(source=source_url, config=graph_config)
    links = graph.run()
    # SearchLinkGraph 返回的 "parsed_doc" 为链接列表；后置确保仅保留 Scholar 域
    return [u for u in links if "scholar.google." in u]


def is_pdf_url(url: str) -> bool:
    """
    判断链接是否为 PDF（根据后缀与常见参数形式）。
    """
    u = url.lower()
    return bool(
        re.search(r"\.pdf($|[#?])", u, flags=re.IGNORECASE)
        or "/pdf" in u
        or re.search(r"(type|format)=pdf", u, flags=re.IGNORECASE)
    )


def extract_pdf_links_from_scholar_page(page_url: str, timeout: int = 25, snapshot_dir: Optional[str] = None) -> List[str]:
    """
    解析 Google Scholar 页面中的 PDF 链接。

    解析规则：在 HTML 中查找所有 href 含有 "pdf" 的 a 标签，
    其内的 href 解析为 PDF 地址；如为 /url?q= 跳转，则提取 q 参数作为真实 PDF 链接。
    在解析开始时会保存该页面的 HTML 快照以供审查。
    """
    print(f"[ScholarPDF] step=fetch_page url={page_url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,*/*",
    }
    html = None
    # 优先使用浏览器渲染抓取，以提升对反爬虫与动态内容的兼容性
    try:
        html = fetch_html_with_browser(page_url, timeout=timeout, snapshot_dir=snapshot_dir)
    except Exception:
        html = None
    if not html:
        try:
            resp = requests.get(page_url, timeout=timeout, headers=headers, allow_redirects=True)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            html = ""
    if not html:
        print(f"[ScholarPDF] step=fetch_empty url={page_url}")
        return []

    snap_path = save_html_snapshot(page_url, html, snapshot_dir=snapshot_dir)
    print(f"[ScholarPDF] step=save_html url={page_url} path={snap_path}")

    pdfs: List[str] = []
    print(f"[ScholarPDF] step=parse_html url={page_url}")
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        anchors = soup.select("a[href]")
        for a in anchors:
            href = a.get("href", "")
            lh = href.lower()
            if "pdf" not in lh:
                continue
            if href.startswith("/url?"):
                qs = parse_qs(urlparse(href).query)
                target = qs.get("q", [""])[0]
                if target:
                    pdfs.append(target)
            else:
                pdfs.append(urljoin(page_url, href))
    except Exception:
        for href in re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
            lh = href.lower()
            if "pdf" not in lh:
                continue
            if href.startswith("/url?"):
                qs = parse_qs(urlparse(href).query)
                target = qs.get("q", [""])[0]
                if target:
                    pdfs.append(target)
            else:
                pdfs.append(urljoin(page_url, href))
    print(f"[ScholarPDF] step=extracted count={len(pdfs)} url={page_url}")
    return list(dict.fromkeys(pdfs))


def save_html_snapshot(page_url: str, html: str, snapshot_dir: Optional[str] = None) -> str:
    """
    将指定页面的 HTML 内容保存到本地快照文件并返回保存路径。

    保存目录为当前脚本同级的 `scholar_html` 子目录，文件名包含时间戳与 URL 摘要。
    """
    base_dir = snapshot_dir or os.path.join(os.path.dirname(__file__), "scholar_html")
    os.makedirs(base_dir, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", page_url)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{safe[:80]}.html"
    path = os.path.join(base_dir, name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        pass
    return path


def fetch_html_with_browser(page_url: str, timeout: int = 45, snapshot_dir: Optional[str] = None) -> str:
    """
    使用 ChromiumLoader（Playwright）抓取并渲染页面，返回完整 HTML。

    通过增强的 loader 配置（超时、重试、networkidle、JS支持）提高稳定性。
    如提供 `snapshot_dir`，则使用该目录下的 `storage_state.json` 以持久化会话（可缓解部分反爬行为）。
    """
    storage_state = None
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)
        storage_state = os.path.join(snapshot_dir, "storage_state.json")

    loader = ChromiumLoader(
        [page_url],
        headless=True,
        storage_state=storage_state,
        timeout=timeout,
        retry_limit=3,
        load_state="networkidle",
        requires_js_support=True,
        browser_name="chromium",
    )
    docs = loader.load()
    if not docs:
        return ""
    return docs[0].page_content or ""


def validate_pdf_url(url: str, timeout: int = 25) -> bool:
    """
    校验链接是否为有效 PDF：
    1) 通过 HEAD 检查 `Content-Type` 是否包含 `application/pdf`；
    2) 若不确定，则 GET 前若干字节，判断是否以 `%PDF-` 开头。
    默认携带 `Referer: https://scholar.google.com` 以提升跨站兼容性。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://scholar.google.com",
    }
    try:
        r = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        ct = r.headers.get("Content-Type", "")
        if "application/pdf" in ct.lower():
            return True
    except Exception:
        pass

    try:
        with requests.get(url, timeout=timeout, headers=headers, stream=True) as resp:
            resp.raise_for_status()
            chunk = next(resp.iter_content(chunk_size=4096))
            return chunk.startswith(b"%PDF-")
    except Exception:
        return False


def harvest_pdfs_from_scholar(urls: List[str], timeout: int = 25, snapshot_dir: Optional[str] = None) -> List[str]:
    """
    针对一组 Google Scholar 结果页面，收集其中的 PDF 外链并去重返回。
    """
    results: List[str] = []
    for u in urls:
        if "scholar.google." not in u:
            continue
        print(f"[ScholarPDF] step=harvest_page url={u}")
        candidates = extract_pdf_links_from_scholar_page(u, timeout=timeout, snapshot_dir=snapshot_dir)
        valid_count = 0
        for c in candidates:
            ok = validate_pdf_url(c, timeout=timeout)
            if ok:
                results.append(c)
                valid_count += 1
        print(f"[ScholarPDF] step=harvest_done url={u} candidates={len(candidates)} valid={valid_count}")
    return list(dict.fromkeys(results))


def create_timestamp_folder(base_dir: str, suffix: str = "_scholar") -> str:
    """
    在指定基准目录下创建以时间戳命名的文件夹并返回路径。

    目录命名规则：`<YYYYMMDD_HHMMSS>_scholar`，可通过 `suffix` 自定义后缀。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"{ts}{suffix}")
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
        "Referer": "https://scholar.google.com",
    }

    # 先进行有效性校验，避免保存 HTML 或错误内容
    if not validate_pdf_url(url, timeout=30):
        raise ValueError("链接非有效 PDF 或被拒绝访问")

    # 使用流式下载确保大文件稳定写入
    resp = requests.get(url, timeout=45, headers=headers, allow_redirects=True, stream=True)
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
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
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
        fname = f"{i:03d}_{base if base.lower().endswith('.pdf') else 'paper.pdf'}"
        try:
            saved_path = download_pdf_to_dir(u, out_dir, fname)
            saved.append(saved_path)
        except Exception:
            continue
    return saved


def collect_scholar_pdfs(query: str, n: int, base_dir: str) -> Tuple[str, List[str]]:
    """
    根据查询词在 Google Scholar 上检索并下载前 N 个 PDF。

    过程：
    1) 构建 LLM；
    2) 运行 SearchGraph 获取 Scholar 结果页 URL；
    3) 从结果页解析 PDF 外链并去重；
    4) 创建时间戳目录并下载文件。
    返回：时间戳目录与成功下载的本地路径列表。
    """
    llm = build_llm()
    source_url = build_scholar_search_url(query)
    scholar_pages = run_scholar_link_graph(llm, source_url, max_results=max(20, n * 4))
    # 先创建时间戳目录，用于统一保存 HTML 快照与 PDF 文件
    out_dir = create_timestamp_folder(base_dir)
    pdf_links = harvest_pdfs_from_scholar(scholar_pages, timeout=25, snapshot_dir=out_dir)
    if not pdf_links:
        # 回退：直接构造 Scholar 查询并解析 PDF 链接
        q = f"{query} literature review"
        fallback_page = f"https://scholar.google.com/scholar?q={requests.utils.quote(q)}"
        pdf_links = extract_pdf_links_from_scholar_page(fallback_page, timeout=25, snapshot_dir=out_dir)
    if not pdf_links:
        raise RuntimeError("未检索到 PDF 链接")
    saved_paths = download_pdfs(pdf_links, out_dir, max_count=n)
    return out_dir, saved_paths


def main() -> None:
    """
    示例入口：在 Google Scholar 检索并抓取指定数量的 PDF 到时间戳目录。
    可根据需求调整数量与保存目录。
    """
    query = "fintech survey"
    n = 10
    base_dir = r"d:\workproject\Weekly\Scrapegraph-ai\examples\mycode"
    out_dir, saved = collect_scholar_pdfs(query, n, base_dir)
    print(f"已创建时间戳目录: {out_dir}")
    print(f"成功下载 {len(saved)} 个PDF")
    for p in saved:
        print(f"- {p}")


if __name__ == "__main__":
    main()

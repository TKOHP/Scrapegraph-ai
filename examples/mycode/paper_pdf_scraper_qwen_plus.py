"""
论文检索与PDF抓取示例脚本
使用搜索节点检索主题相关的 PDF 链接，批量下载并保存到以时间戳命名的文件夹
"""

import os
import tempfile
import requests
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime

from scrapegraphai.nodes import (
    SearchInternetNode,
)
from langchain_openai import ChatOpenAI


def build_llm() -> ChatOpenAI:
    """构建 Qwen-Plus 的 LLM 客户端（DashScope 兼容接口）"""
    llm_cfg = {
        "api_key": "sk-cd7b54e0eaf5444ea29c71dc2cea3731",
        "model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    return ChatOpenAI(
        openai_api_key=llm_cfg["api_key"],
        openai_api_base=llm_cfg["base_url"],
        model=llm_cfg["model"],
    )


def search_pdf_links(llm: ChatOpenAI, query: str, max_results: int = 10) -> List[str]:
    """使用 SearchInternetNode 检索并筛选 PDF 链接；若无直接PDF，尝试在页面中提取PDF链接"""
    search_node = SearchInternetNode(
        input="user_input",
        output=["search_results"],
        node_config={
            "llm_model": llm,
            "search_engine": "duckduckgo",
            "max_results": max_results,
            "verbose": False,
        },
    )
    state = {"user_input": query}
    result = search_node.execute(state)
    urls = result.get("search_results", [])
    pdfs = [u for u in urls if u.lower().endswith(".pdf")]
    if pdfs:
        return pdfs

    # 二次策略：对候选页进行解析，提取页面中的PDF链接
    extracted: List[str] = []
    for u in urls:
        try:
            r = requests.get(u, timeout=15, allow_redirects=True)
            ct = r.headers.get("Content-Type", "")
            if "application/pdf" in ct:
                extracted.append(u)
                continue
            if not r.text:
                continue
            for href in set(_extract_hrefs(r.text)):
                if href.lower().endswith(".pdf"):
                    extracted.append(urljoin(u, href))
        except Exception:
            continue
    return list(dict.fromkeys(extracted))


def _extract_hrefs(html: str) -> List[str]:
    """从HTML中提取所有href链接（简易正则）"""
    import re
    return re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)


def download_pdf(url: str) -> str:
    """下载 PDF 到临时文件并返回本地路径

    该函数保留以兼容原有用法，若需指定保存目录与文件名，使用 `download_pdf_to_dir`。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/pdf,application/octet-stream,*/*",
    }
    resp = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path


def download_pdf_to_dir(url: str, out_dir: str, filename: Optional[str] = None) -> str:
    """下载单个 PDF 到指定目录并返回保存路径

    参数：
    - url: PDF 链接
    - out_dir: 目标保存目录（需存在）
    - filename: 可选文件名，若未提供则根据 URL 自动生成
    """
    os.makedirs(out_dir, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
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
    # 如存在同名文件，追加序号避免覆盖
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


def create_timestamp_folder(base_dir: str) -> str:
    """在指定基准目录下创建以时间戳命名的文件夹并返回路径"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, ts)
    os.makedirs(out_dir, exist_ok=False)
    return out_dir


def download_pdfs(urls: List[str], out_dir: str, max_count: int) -> List[str]:
    """批量下载前 `max_count` 个 PDF 到 `out_dir` 并返回本地路径列表"""
    saved: List[str] = []
    for i, u in enumerate(urls[:max_count], start=1):
        path_part = urlparse(u).path
        base = os.path.basename(path_part)
        # 构造规范文件名：序号_原名或序号_paper.pdf
        if base and base.lower().endswith(".pdf"):
            fname = f"{i:03d}_{base}"
        else:
            fname = f"{i:03d}_paper.pdf"
        try:
            saved_path = download_pdf_to_dir(u, out_dir, fname)
            saved.append(saved_path)
        except Exception:
            # 单个失败不影响整体流程
            continue
    return saved


def collect_pdfs(query: str, n: int, base_dir: str) -> str:
    """根据查询检索 PDF 链接，下载前 N 个到时间戳文件夹并返回文件夹路径

    下载策略包含回退：在初次检索为空时自动添加 `filetype:pdf` 强化查询。
    """
    llm = build_llm()
    pdf_links = search_pdf_links(llm, query, max_results=max(20, n * 3))
    if not pdf_links:
        pdf_links = search_pdf_links(llm, f"{query} filetype:pdf", max_results=max(30, n * 4))
    if not pdf_links:
        raise RuntimeError("未检索到 PDF 链接")

    out_dir = create_timestamp_folder(base_dir)
    saved_paths = download_pdfs(pdf_links, out_dir, max_count=n)
    print(f"已创建时间戳目录: {out_dir}")
    print(f"成功下载 {len(saved_paths)} 个PDF")
    for p in saved_paths:
        print(f"- {p}")
    return out_dir


def main() -> None:
    """示例入口：执行查询并抓取指定数量的 PDF 到时间戳目录"""
    query = "Fintech survey PDF site:arxiv.org"
    n = 5  # 目标抓取数量，可根据需要调整
    base_dir = r"d:\workproject\Weekly\Scrapegraph-ai\examples\mycode"
    collect_pdfs(query, n, base_dir)


if __name__ == "__main__":
    main()

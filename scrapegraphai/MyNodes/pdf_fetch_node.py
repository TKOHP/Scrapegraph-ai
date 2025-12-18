"""
PDF 获取节点

遍历 `AIPaper` 对象，根据 `urlLink` 获取 PDF 链接并下载保存，更新数据库 `pdfLink` 字段。
"""

import os
import re
import time
import hashlib
from typing import List, Optional
from urllib.parse import urljoin, urlparse, parse_qs

import requests

from ..utils import get_logger
from ..nodes.base_node import BaseNode
from .db_manager import DatabaseManager, AIPaper


def _is_pdf_url(url: str) -> bool:
    """判断链接是否可能为 PDF"""
    u = (url or "").lower()
    return bool(
        re.search(r"\.pdf($|[#?])", u)
        or "/pdf" in u
        or re.search(r"(type|format)=pdf", u)
    )


class PdfFetchNode(BaseNode):
    """
    PDF 下载节点
    
    根据论文网页地址尝试解析或验证 PDF 链接，下载到指定目录并更新数据库。
    node_config 支持：
    - db_path: 数据库文件路径
    - download_dir: PDF 保存目录，默认 `data/papers`
    - timeout: 请求超时秒数，默认 30
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "PdfFetch",
    ):
        """
        初始化节点
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger(__name__)
        cfg = self.node_config or {}
        self.db_path = cfg.get("db_path", "data/google_scholar_papers.db")
        self.download_dir = cfg.get("download_dir", os.path.join("data", "papers"))
        self.timeout = int(cfg.get("timeout", 30))
        self.db = DatabaseManager(self.db_path)
        os.makedirs(self.download_dir, exist_ok=True)

    def _safe_filename(self, base_url: str, ext: str = ".pdf") -> str:
        """
        生成唯一可用的文件名
        """
        parsed = urlparse(base_url)
        name = os.path.basename(parsed.path) or parsed.netloc or "paper"
        if not name.lower().endswith(ext):
            name = f"{name}{ext}"
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
        if len(safe) < 3:
            safe = hashlib.sha1(base_url.encode("utf-8")).hexdigest()[:12] + ext
        return safe

    def _validate_pdf(self, url: str) -> bool:
        """
        通过 HEAD 与首块字节校验是否为 PDF
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
            r = requests.head(url, timeout=self.timeout, headers=headers, allow_redirects=True)
            ct = (r.headers.get("Content-Type") or "").lower()
            if "application/pdf" in ct:
                return True
        except Exception:
            pass
        try:
            with requests.get(url, timeout=self.timeout, headers=headers, stream=True) as resp:
                resp.raise_for_status()
                chunk = next(resp.iter_content(chunk_size=4096))
                return chunk.startswith(b"%PDF-")
        except Exception:
            return False

    def _extract_pdf_links_from_html(self, page_url: str, html: str) -> List[str]:
        """
        从 HTML 中提取潜在 PDF 链接
        """
        pdfs: List[str] = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            anchors = soup.select("a[href]")
            for a in anchors:
                href = a.get("href", "")
                if not href:
                    continue
                lh = href.lower()
                if "pdf" not in lh and not _is_pdf_url(href):
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
                if "pdf" not in lh and not _is_pdf_url(href):
                    continue
                if href.startswith("/url?"):
                    qs = parse_qs(urlparse(href).query)
                    target = qs.get("q", [""])[0]
                    if target:
                        pdfs.append(target)
                else:
                    pdfs.append(urljoin(page_url, href))
        return list(dict.fromkeys(pdfs))

    def _find_pdf_candidates(self, page_url: str) -> List[str]:
        """
        请求网页并寻找潜在 PDF 链接
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/pdf,*/*",
        }
        try:
            r = requests.get(page_url, timeout=self.timeout, headers=headers, allow_redirects=True)
            ct = r.headers.get("Content-Type", "")
            if "application/pdf" in ct.lower():
                return [page_url]
            if not r.text:
                return []
            return self._extract_pdf_links_from_html(page_url, r.text)
        except Exception as e:
            self.logger.warning(f"请求网页失败 url={page_url} err={e}")
            return []

    def _download_pdf(self, url: str, save_dir: str, filename: Optional[str] = None) -> str:
        """
        下载 PDF 到指定目录
        """
        os.makedirs(save_dir, exist_ok=True)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/pdf,application/octet-stream,*/*",
            "Referer": "https://scholar.google.com",
        }
        resp = requests.get(url, timeout=max(self.timeout, 45), headers=headers, allow_redirects=True, stream=True)
        resp.raise_for_status()
        name = filename or self._safe_filename(url, ".pdf")
        save_path = os.path.join(save_dir, name)
        base, ext = os.path.splitext(save_path)
        idx = 2
        while os.path.exists(save_path):
            save_path = f"{base}_{idx}{ext}"
            idx += 1
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return save_path

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 遍历 `papers`，为缺失 `pdfLink` 的记录查找并下载 PDF
        - 更新数据库并返回更新后的 `papers`
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        papers: List[AIPaper] = state[input_keys[0]]

        updated: List[AIPaper] = []
        for p in papers:
            try:
                if p.pdfLink and os.path.exists(p.pdfLink):
                    updated.append(p)
                    continue
                candidates = []
                if _is_pdf_url(p.urlLink):
                    candidates = [p.urlLink]
                else:
                    candidates = self._find_pdf_candidates(p.urlLink)
                target_url = None
                for c in candidates:
                    if self._validate_pdf(c):
                        target_url = c
                        break
                if not target_url:
                    self.logger.warning(f"未找到有效 PDF url={p.urlLink}")
                    updated.append(p)
                    continue
                save_path = self._download_pdf(target_url, self.download_dir)
                self.db.update_fields(int(p.id), {"pdfLink": save_path})
                p.pdfLink = save_path
                updated.append(p)
            except Exception as e:
                self.logger.error(f"下载 PDF 失败 id={p.id} url={p.urlLink} err={e}")
                updated.append(p)

        state.update({self.output[0]: updated})
        return state


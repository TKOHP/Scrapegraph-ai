"""
PDF 转标准化 Markdown 节点

遍历 `AIPaper` 对象，将 PDF 内容提取为 Markdown，保存到与 PDF 同目录、同名不同扩展的文件中。
"""

import os
from typing import List, Optional

from ..utils import get_logger
from ..nodes.base_node import BaseNode
from .db_manager import DatabaseManager, AIPaper


class PdfToMarkdownNode(BaseNode):
    """
    PDF 转 Markdown 节点
    
    将 `pdfLink` 指定的文件解析为文本，并生成简易结构化 Markdown。
    node_config 支持：
    - db_path: 数据库路径
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "PdfToMarkdown",
    ):
        """
        初始化节点
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger(__name__)
        cfg = self.node_config or {}
        self.db_path = cfg.get("db_path", "data/google_scholar_papers.db")
        self.db = DatabaseManager(self.db_path)

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        使用 PyPDFLoader 提取 PDF 文本，若不可用则回退到空内容
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            text = "\n\n".join([p.page_content for p in pages])
            return text
        except Exception as e:
            self.logger.error(f"解析 PDF 失败 path={pdf_path} err={e}")
            return ""

    def _to_markdown(self, text: str) -> str:
        """
        将纯文本转换为简易 Markdown
        """
        if not text:
            return "# 内容解析失败\n\n"
        lines = [ln.strip() for ln in text.splitlines()]
        md_lines = []
        for i, ln in enumerate(lines):
            if not ln:
                md_lines.append("")
                continue
            if len(ln) < 120 and ln.isupper():
                md_lines.append(f"# {ln.title()}")
            elif i == 0 and len(ln) < 120:
                md_lines.append(f"# {ln}")
            else:
                md_lines.append(ln)
        return "\n".join(md_lines) + "\n"

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 遍历 `papers`，为存在 `pdfLink` 且未生成 `mdLink` 的记录生成 Markdown
        - 保存到同目录同名 `.md` 文件，并更新数据库
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        papers: List[AIPaper] = state[input_keys[0]]

        self.logger.info(f"Markdown节点——开始处理 papers={len(papers)}")
        updated: List[AIPaper] = []
        generated = 0
        reused = 0
        skipped = 0
        failed = 0
        for idx, p in enumerate(papers, start=1):
            try:
                if not p.pdfLink or not os.path.exists(p.pdfLink):
                    skipped += 1
                    self.logger.info(f"Markdown节点——第 {idx}/{len(papers)} 篇跳过：缺少 pdfLink")
                    updated.append(p)
                    continue
                md_path = os.path.splitext(p.pdfLink)[0] + ".md"
                if os.path.exists(md_path):
                    self.db.update_fields(int(p.id), {"mdLink": md_path})
                    p.mdLink = md_path
                    reused += 1
                    self.logger.info(f"Markdown节点——第 {idx}/{len(papers)} 篇复用已存在 md={md_path}")
                    updated.append(p)
                    continue
                text = self._extract_text_from_pdf(p.pdfLink)
                md = self._to_markdown(text)
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md)
                self.db.update_fields(int(p.id), {"mdLink": md_path})
                p.mdLink = md_path
                generated += 1
                self.logger.info(
                    f"Markdown节点——第 {idx}/{len(papers)} 篇生成成功 md={md_path} chars={len(md)}"
                )
                updated.append(p)
            except Exception as e:
                self.logger.error(f"生成 Markdown 失败 id={p.id} path={p.pdfLink} err={e}")
                failed += 1
                updated.append(p)

        self.logger.info(
            f"Markdown节点——处理完成 generated={generated} reused={reused} skipped={skipped} failed={failed}"
        )
        state.update({self.output[0]: updated})
        return state

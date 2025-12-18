"""
文档分类节点

根据 Markdown 内容提取元信息，填充 `meta`、`subject`、`publishTime` 字段。
"""

import os
import re
from typing import List, Optional, Dict

from ..utils import get_logger
from ..nodes.base_node import BaseNode
from .db_manager import DatabaseManager, AIPaper


class DocumentClassifyNode(BaseNode):
    """
    文档分类与元信息提取节点
    
    输入 `papers & subject`，读取每个论文的 Markdown，提取标题、关键词、摘要、发布时间等信息。
    结果以 JSON 字符串写入 `meta` 字段，并更新数据库。
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "DocumentClassify",
    ):
        """
        初始化节点
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger(__name__)
        cfg = self.node_config or {}
        self.db_path = cfg.get("db_path", "data/google_scholar_papers.db")
        self.db = DatabaseManager(self.db_path)

    def _extract_meta(self, md_path: str) -> Dict[str, str]:
        """
        从 Markdown 文件中提取元信息
        """
        meta = {"title": "", "keywords": "", "summary": "", "first_page": ""}
        if not (md_path and os.path.exists(md_path)):
            return meta
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = [ln.strip() for ln in content.splitlines()]
            for ln in lines:
                if not meta["title"] and ln.startswith("# "):
                    meta["title"] = ln[2:].strip()
                if not meta["keywords"]:
                    m = re.search(r"(?:关键词|Keywords?)\s*[:：]\s*(.+)", ln, flags=re.IGNORECASE)
                    if m:
                        meta["keywords"] = m.group(1).strip()
            meta["first_page"] = "\n".join(lines[:60])
            if not meta["summary"]:
                text = "\n".join(lines)
                para = re.split(r"\n\s*\n", text)
                meta["summary"] = (para[0] if para else text[:600])[:1200]
        except Exception as e:
            self.logger.error(f"提取元信息失败 path={md_path} err={e}")
        return meta

    def _extract_publish_time(self, meta_text: str) -> str:
        """
        从文本中提取发布时间（优先匹配 YYYY-MM-DD，其次匹配年份）
        """
        m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", meta_text)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        y = re.search(r"(19|20)\d{2}", meta_text)
        return y.group(0) if y else ""

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 读取每条论文的 Markdown
        - 提取并更新 `meta`、`subject`、`publishTime`
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        papers: List[AIPaper] = state[input_keys[0]]
        subject: str = state[input_keys[1]]

        updated: List[AIPaper] = []
        for p in papers:
            try:
                meta_dict = self._extract_meta(p.mdLink or "")
                meta_text = f"Title: {meta_dict.get('title','')}\nKeywords: {meta_dict.get('keywords','')}\nSummary: {meta_dict.get('summary','')}\n"
                publish_time = self._extract_publish_time(meta_text + (meta_dict.get("first_page", "") or ""))
                updates = {
                    "meta": str(meta_dict),
                    "subject": subject or p.subject,
                    "publishTime": publish_time or p.publishTime,
                }
                if p.id is not None:
                    self.db.update_fields(int(p.id), updates)
                p.meta = updates["meta"]
                p.subject = updates["subject"]
                p.publishTime = updates["publishTime"]
                updated.append(p)
            except Exception as e:
                self.logger.error(f"文档分类失败 id={p.id} err={e}")
                updated.append(p)

        state.update({self.output[0]: updated})
        return state


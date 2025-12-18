"""
单文档总结节点

根据 Markdown 内容生成简短总结，保存为同目录的 `.summary.md` 文件并更新数据库。
"""

import os
from typing import List, Optional

from ..utils import get_logger
from ..nodes.base_node import BaseNode
from .db_manager import DatabaseManager, AIPaper


class DocumentSummaryNode(BaseNode):
    """
    文档总结节点
    
    读取 `mdLink` 的内容，生成简短概要文本并写入 `.summary.md` 文件。
    node_config 支持：
    - db_path: 数据库路径
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "DocumentSummary",
    ):
        """
        初始化节点
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger(__name__)
        cfg = self.node_config or {}
        self.db_path = cfg.get("db_path", "data/google_scholar_papers.db")
        self.db = DatabaseManager(self.db_path)

    def _summarize(self, md_path: str) -> str:
        """
        生成简易总结文本
        """
        if not (md_path and os.path.exists(md_path)):
            return "无法生成总结：Markdown 文件不存在或路径非法。"
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            first_section = lines[:60]
            summary = []
            for ln in first_section:
                if ln.startswith("# "):
                    summary.append(f"主题：{ln[2:].strip()}")
                    break
            summary.append("摘要：")
            summary.extend([ln for ln in first_section if not ln.startswith("# ")][:10])
            return "\n".join(summary) + "\n"
        except Exception as e:
            self.logger.error(f"生成总结失败 path={md_path} err={e}")
            return "生成总结过程中发生错误。"

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 为每条论文生成 `.summary.md` 文件
        - 更新数据库 `summaryLink`
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        papers: List[AIPaper] = state[input_keys[0]]

        updated: List[AIPaper] = []
        for p in papers:
            try:
                if not p.mdLink or not os.path.exists(p.mdLink):
                    updated.append(p)
                    continue
                summary_path = os.path.splitext(p.mdLink)[0] + ".summary.md"
                if not os.path.exists(summary_path):
                    summary_text = self._summarize(p.mdLink)
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(summary_text)
                if p.id is not None:
                    self.db.update_fields(int(p.id), {"summaryLink": summary_path})
                p.summaryLink = summary_path
                updated.append(p)
            except Exception as e:
                self.logger.error(f"写入总结失败 id={p.id} md={p.mdLink} err={e}")
                updated.append(p)

        state.update({self.output[0]: updated})
        return state


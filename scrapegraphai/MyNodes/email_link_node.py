"""
邮件链接提取节点

输入邮件文本列表与固定主题，提取论文网页链接，并创建/更新数据库记录。
"""

import re
from typing import List, Optional
from urllib.parse import urljoin, urlparse

from ..utils import get_logger
from ..nodes.base_node import BaseNode
from .db_manager import DatabaseManager, AIPaper


class EmailLinkNode(BaseNode):
    """
    邮件链接提取节点
    
    从邮件文本中提取论文网页链接，写入 SQLite 数据库，并在状态中返回 `papers` 列表。
    node_config 需包含：
    - db_path: 数据库文件路径
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "EmailLinkExtract",
    ):
        """
        初始化节点
        
        Args:
            input: 输入键表达式，如 "emails & subject"
            output: 输出键列表，建议为 ["papers"]
            node_config: 节点配置，需包含 db_path
            node_name: 节点名称
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger(__name__)
        self.db_path = (self.node_config or {}).get("db_path", "data/google_scholar_papers.db")
        self.db = DatabaseManager(self.db_path)

    def _extract_urls(self, text: str) -> List[str]:
        """
        从文本中提取可能的论文网页地址
        """
        pattern = r"https?://[^\s<>\"]+"
        urls = re.findall(pattern, text)
        cleaned = []
        for u in urls:
            if u.startswith("http://") or u.startswith("https://"):
                cleaned.append(u.rstrip(").,;\"'"))
        return list(dict.fromkeys(cleaned))

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 解析输入邮件文本
        - 提取网址并写入数据库
        - 输出 `papers` 列表
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        emails = state[input_keys[0]]
        subject = state[input_keys[1]]

        if not isinstance(emails, list):
            raise ValueError("emails 输入必须为 List[str]")

        papers: List[AIPaper] = []

        try:
            for text in emails:
                for url in self._extract_urls(text or ""):
                    existing = self.db.find_by_url(url)
                    if existing:
                        if (not existing.subject) and subject:
                            self.db.update_fields(existing.id, {"subject": subject})
                        papers.append(existing)
                        continue

                    paper = AIPaper(
                        id=None,
                        urlLink=url,
                        pdfLink=None,
                        mdLink=None,
                        summaryLink=None,
                        meta=None,
                        publishTime=None,
                        subject=subject,
                    )
                    new_id = self.db.insert_paper(paper)
                    paper.id = new_id
                    papers.append(paper)
        except Exception as e:
            self.logger.error(f"邮件链接提取失败: {e}")
            raise

        state.update({self.output[0]: papers})
        return state


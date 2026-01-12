"""
Google Scholar 邮件到 PDF 最小处理图

仅执行以下步骤：
1) 邮件链接提取
2) PDF 获取与下载
"""

import time
from typing import List, Optional, Type
from pydantic import BaseModel

from scrapegraphai.utils import get_logger
from scrapegraphai.graphs.base_graph import BaseGraph
from scrapegraphai.graphs.abstract_graph import AbstractGraph
from ..Nodes import (
    DatabaseManager,
    EmailLinkNode,
    PdfFetchNode,
    AIPaper,
)


class GoogleScholarMinimalGraph(AbstractGraph):
    """
    GoogleScholarMinimalGraph
    
    仅管理从邮件到 PDF 的最小处理流程，并暴露数据库操作。
    """

    def __init__(
        self,
        prompt: str,
        email_config: dict,
        config: dict,
        schema: Optional[Type[BaseModel]] = None,
    ):
        """
        初始化图实例
        
        参数:
            prompt: 流程说明或占位文本
            email_config: 邮箱抓取配置字典（imap_server、account、password 等）
            config: 图配置；若不使用 LLM，可传入 {"llm": {"model_instance": None, "model_tokens": 8192}}
            schema: 可选的结构模式
        """
        self.email_config = email_config or {}
        super().__init__(prompt, config, email_config, schema)
        self.logger = get_logger()
        self.input_key = "email_config"
        db_path = (self.config or {}).get("db_path")
        self.db = DatabaseManager(db_path)

    def _create_graph(self) -> BaseGraph:
        """
        创建仅包含“邮件链接提取”和“PDF 获取”的执行图
        """
        db_path = (self.config or {}).get("db_path")
        download_dir = (self.config or {}).get("download_dir", "AIpaper/data/papers")

        email_node = EmailLinkNode(
            input="email_config",
            output=["papers"],
            node_config={
                "db_path": db_path,
                "use_qq_email": True,
            },
        )
        pdf_node = PdfFetchNode(
            input="papers",
            output=["papers"],
            node_config={"db_path": db_path, "download_dir": download_dir},
        )

        return BaseGraph(
            nodes=[email_node, pdf_node],
            edges=[
                (email_node, pdf_node),
            ],
            entry_point=email_node,
            graph_name=self.__class__.__name__,
        )

    def run(self) -> List[AIPaper]:
        """
        执行流程并返回处理后的 `AIPaper` 列表
        """
        email_cfg = self.source or {}
        self.logger.info(
            f"最小流程图——开始执行 days_recent={email_cfg.get('days_recent', 7)} "
            f"sender_email={email_cfg.get('sender_email', 'scholaralerts-noreply@google.com')} "
            f"required_subject_contains={(email_cfg.get('required_subject_contains') or '')}"
        )
        started = time.time()
        inputs = {
            "user_prompt": self.prompt,
            "email_config": self.source,
        }
        self.final_state, self.execution_info = self.graph.execute(inputs)
        elapsed_ms = int((time.time() - started) * 1000)
        out_papers = self.final_state.get("papers", []) if isinstance(self.final_state, dict) else []
        self.logger.info(f"最小流程图——执行完成 papers={len(out_papers)} elapsed_ms={elapsed_ms}")
        return self.final_state.get("papers", [])

    def db_insert(self, paper: AIPaper) -> int:
        """
        数据库插入
        """
        return self.db.insert_paper(paper)

    def db_update_fields(self, paper_id: int, updates: dict) -> None:
        """
        数据库字段更新
        """
        self.db.update_fields(paper_id, updates)

    def db_delete(self, paper_id: int) -> None:
        """
        数据库删除
        """
        self.db.delete_paper(paper_id)

    def db_list(self, subject: Optional[str] = None) -> List[AIPaper]:
        """
        数据库查询（可按主题过滤）
        """
        return self.db.list_papers(subject)

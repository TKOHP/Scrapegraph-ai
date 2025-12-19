"""
Google Scholar 订阅论文处理图

按流程图依次执行：
1) 邮件链接提取
2) PDF 获取与下载
3) PDF 转标准化 Markdown
4) 文档分类（提取 meta/subject/publishTime）
5) 单文档总结（生成 summaryLink）

同时提供对数据库的增删改查接口。
"""

import time
from typing import List, Optional, Type
from pydantic import BaseModel

from ..utils import get_logger
from ..nodes.base_node import BaseNode
from ..graphs.base_graph import BaseGraph
from ..graphs.abstract_graph import AbstractGraph
from ..MyNodes import (
    DatabaseManager,
    EmailLinkNode,
    PdfFetchNode,
    PdfToMarkdownNode,
    DocumentClassifyNode,
    DocumentSummaryNode,
    AIPaper,
)


class GoogleScholarPaperGraph(AbstractGraph):
    """
    GoogleScholarPaperGraph
    
    管理从邮件到 PDF/Markdown 的完整处理流程，并暴露数据库操作。
    """

    def __init__(
        self,
        prompt: str,
        email_config: dict,
        subjects: List[str],
        config: dict,
        schema: Optional[Type[BaseModel]] = None,
        simple_llm: Optional[object] = None,
        complex_llm: Optional[object] = None,
    ):
        """
        初始化图实例
        
        Args:
            prompt: 流程说明或占位文本
            email_config: 邮箱抓取配置字典（imap_server、account、password 等）
            subjects: 主题池（中文主题列表，如“金融科技”、“大模型智能体”等）
            config: 图配置，需包含 llm 字段；若不使用 LLM，可传入 {"llm": {"model_instance": None, "model_tokens": 8192}}
            schema: 可选的结构模式
        """
        self.simple_llm = simple_llm
        self.complex_llm = complex_llm
        self.email_config = email_config or {}
        self.subjects_pool = subjects

        super().__init__(prompt, config, email_config, schema)
        self.logger = get_logger(__name__)
        self.input_key = "email_config"
        db_path = (config or {}).get("db_path", "data/google_scholar_papers.db")
        self.db = DatabaseManager(db_path)
        for node in getattr(self, "graph", None).nodes:
            if isinstance(node, DocumentClassifyNode):
                node.llm_model = self.simple_llm
            if isinstance(node, DocumentSummaryNode):
                node.llm_model = self.complex_llm

    def _create_graph(self) -> BaseGraph:
        """
        创建节点并构建执行图
        """
        db_path = (self.config or {}).get("db_path", "data/google_scholar_papers.db")
        download_dir = (self.config or {}).get("download_dir", "data/papers")

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
        md_node = PdfToMarkdownNode(
            input="papers",
            output=["papers"],
            node_config={"db_path": db_path},
        )
        classify_node = DocumentClassifyNode(
            input="papers & subjects",
            output=["papers"],
            node_config={
                "db_path": db_path,
                "llm_model": self.simple_llm,
            },
        )
        summary_node = DocumentSummaryNode(
            input="papers",
            output=["papers"],
            node_config={
                "db_path": db_path,
                "llm_model": self.complex_llm,
            },
        )

        return BaseGraph(
            nodes=[email_node, pdf_node, md_node, classify_node, summary_node],
            edges=[
                (email_node, pdf_node),
                (pdf_node, md_node),
                (md_node, classify_node),
                (classify_node, summary_node),
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
            f"流程图——开始执行 subjects_pool={len(self.subjects_pool)} "
            f"days_recent={email_cfg.get('days_recent', 7)} "
            f"sender_email={email_cfg.get('sender_email', 'scholaralerts-noreply@google.com')} "
            f"required_subject_contains={(email_cfg.get('required_subject_contains') or '')}"
        )
        started = time.time()
        inputs = {
            "user_prompt": self.prompt,
            "email_config": self.source,
            "subjects": self.subjects_pool,
        }
        self.final_state, self.execution_info = self.graph.execute(inputs)
        elapsed_ms = int((time.time() - started) * 1000)
        out_papers = self.final_state.get("papers", []) if isinstance(self.final_state, dict) else []
        self.logger.info(f"流程图——执行完成 papers={len(out_papers)} elapsed_ms={elapsed_ms}")
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

#（已移除：邮箱与模型环境变量；请在示例 main 中配置并传入）

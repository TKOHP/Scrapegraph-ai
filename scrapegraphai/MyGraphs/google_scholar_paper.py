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
        emails: List[str],
        subject: str,
        config: dict,
        schema: Optional[Type[BaseModel]] = None,
    ):
        """
        初始化图实例
        
        Args:
            prompt: 流程说明或占位文本
            emails: 邮件文本列表
            subject: 固定主题（来自外部订阅配置）
            config: 图配置，需包含 llm 字段；若不使用 LLM，可传入 {"llm": {"model_instance": None, "model_tokens": 8192}}
            schema: 可选的结构模式
        """
        super().__init__(prompt, config, emails, schema)
        self.logger = get_logger(__name__)
        self.input_key = "emails"
        self.fixed_subject = subject
        db_path = (config or {}).get("db_path", "data/google_scholar_papers.db")
        self.db = DatabaseManager(db_path)

    def _create_graph(self) -> BaseGraph:
        """
        创建节点并构建执行图
        """
        db_path = (self.config or {}).get("db_path", "data/google_scholar_papers.db")
        download_dir = (self.config or {}).get("download_dir", "data/papers")

        email_node = EmailLinkNode(
            input="emails & subject",
            output=["papers"],
            node_config={"db_path": db_path},
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
            input="papers & subject",
            output=["papers"],
            node_config={"db_path": db_path},
        )
        summary_node = DocumentSummaryNode(
            input="papers",
            output=["papers"],
            node_config={"db_path": db_path},
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
        inputs = {"user_prompt": self.prompt, "emails": self.source, "subject": self.fixed_subject}
        self.final_state, self.execution_info = self.graph.execute(inputs)
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


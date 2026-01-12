"""
arXiv 订阅论文处理图

按流程图依次执行：
1) arXiv 链接提取
2) PDF 获取与下载
3) PDF 转标准化 Markdown
4) 文档分类（提取 meta/subject/publishTime）
5) 文献概览与深度分析（生成 overviewLink 与 analysisLink）
"""

import time
from typing import List, Optional, Type
from pydantic import BaseModel

from scrapegraphai.utils import get_logger
from scrapegraphai.nodes.base_node import BaseNode
from scrapegraphai.graphs.base_graph import BaseGraph
from scrapegraphai.graphs.abstract_graph import AbstractGraph
from ..Nodes import (
    DatabaseManager,
    ArxivLinkNode,
    PdfFetchNode,
    PdfToMarkdownNode,
    DocumentClassifyNode,
    DocumentOverviewNode,
    ResearchAnalysisNode,
    AIPaper,
)


class ArxivPaperGraph(AbstractGraph):
    """
    ArxivPaperGraph
    
    管理从 arXiv 查询到 PDF/Markdown 的完整处理流程，并暴露数据库操作。
    """

    def __init__(
        self,
        prompt: str,
        arxiv_config: dict,
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
            arxiv_config: arXiv 查询配置字典（query/queries/categories/max_results 等）
            subjects: 主题池（中文主题列表，如“金融科技”、“大模型智能体”等）
            config: 图配置，需包含 llm 字段；若不使用 LLM，可传入 {"llm": {"model_instance": None, "model_tokens": 8192}}
            schema: 可选的结构模式
        """
        self.simple_llm = simple_llm
        self.complex_llm = complex_llm
        self.arxiv_config = arxiv_config or {}
        self.subjects_pool = subjects

        super().__init__(prompt, config, arxiv_config, schema)
        self.logger = get_logger()
        self.input_key = "arxiv_config"
        db_path = (self.config or {}).get("db_path")
        self.db = DatabaseManager(db_path)
        for node in getattr(self, "graph", None).nodes:
            if isinstance(node, DocumentClassifyNode):
                node.llm_model = self.simple_llm
            if isinstance(node, (DocumentOverviewNode, ResearchAnalysisNode)):
                node.llm_model = self.complex_llm
            if isinstance(node, PdfToMarkdownNode):
                node.llm_model = self.simple_llm

    def _create_graph(self) -> BaseGraph:
        """
        创建节点并构建执行图
        """
        db_path = (self.config or {}).get("db_path")
        download_dir = (self.config or {}).get("download_dir", "AIpaper/data/papers")
        rebuild_md = bool((self.config or {}).get("rebuild_md", False))
        rebuild_classify = bool((self.config or {}).get("rebuild_classify", False))
        rebuild_overview = bool((self.config or {}).get("rebuild_overview", False))
        rebuild_analysis = bool((self.config or {}).get("rebuild_analysis", False))

        arxiv_node = ArxivLinkNode(
            input="arxiv_config",
            output=["papers"],
            node_config={
                "db_path": db_path,
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
            node_config={
                "db_path": db_path,
                "force_rebuild": rebuild_md,
                "format_with_llm": True,
            },
        )
        classify_node = DocumentClassifyNode(
            input="papers & subjects",
            output=["papers"],
            node_config={
                "db_path": db_path,
                "force_rebuild": rebuild_classify,
            },
        )
        overview_node = DocumentOverviewNode(
            input="papers",
            output=["papers"],
            node_config={
                "db_path": db_path,
                "force_rebuild": rebuild_overview,
            },
        )
        analysis_node = ResearchAnalysisNode(
            input="papers",
            output=["papers"],
            node_config={
                "db_path": db_path,
                "force_rebuild": rebuild_analysis,
            },
        )

        return BaseGraph(
            nodes=[arxiv_node, pdf_node, md_node, classify_node, overview_node, analysis_node],
            edges=[
                (arxiv_node, pdf_node),
                (pdf_node, md_node),
                (md_node, classify_node),
                (classify_node, analysis_node),
                # (classify_node, overview_node),
                # (overview_node, analysis_node),
            ],
            entry_point=arxiv_node,
            graph_name=self.__class__.__name__,
        )

    def run(self) -> List[AIPaper]:
        """
        执行流程并返回处理后的 `AIPaper` 列表
        """
        cfg = self.source or {}
        self.logger.info(
            f"流程图——开始执行 subjects_pool={len(self.subjects_pool)} "
            f"max_results={cfg.get('max_results', 30)} "
            f"query={(cfg.get('query') or (cfg.get('queries') or []))}"
        )
        started = time.time()
        inputs = {
            "user_prompt": self.prompt,
            "arxiv_config": self.source,
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

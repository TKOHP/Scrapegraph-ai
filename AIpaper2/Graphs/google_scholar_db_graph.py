"""
Google Scholar 数据库选择处理图

按流程图依次执行（直接从数据库选择对象，无需邮件）：
1) 数据库选择（由 run 内完成）
2) PDF 获取与下载
3) PDF 转标准化 Markdown
4) 文档分类（提取 meta/subject/publishTime）
5) 文献深度分析（生成 analysisLink）
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
    PdfFetchNode,
    PdfToMarkdownNode,
    DocumentClassifyNode,
    DocumentOverviewNode,
    ResearchAnalysisNode,
    AIPaper,
)


class GoogleScholarDbPaperGraph(AbstractGraph):
    """
    GoogleScholarDbPaperGraph
    
    管理从数据库选择到 PDF/Markdown 的完整处理流程，并暴露数据库操作。
    """

    def __init__(
        self,
        prompt: str,
        selection_config: dict,
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
            selection_config: 选择配置字典（top_n、ids、subject 等）
            subjects: 主题池（中文主题列表，如“金融科技”、“大模型智能体”等）
            config: 图配置，需包含 llm 字段；若不使用 LLM，可传入 {"llm": {"model_instance": None, "model_tokens": 8192}}
            schema: 可选的结构模式
        """
        self.simple_llm = simple_llm
        self.complex_llm = complex_llm
        self.selection_config = selection_config or {}
        self.subjects_pool = subjects

        super().__init__(prompt, config, selection_config, schema)
        self.logger = get_logger()
        self.input_key = "selection_config"
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
            nodes=[pdf_node, md_node, classify_node, overview_node, analysis_node],
            edges=[
                (pdf_node, md_node),
                (md_node, classify_node),
                (classify_node, analysis_node),
            ],
            entry_point=pdf_node,
            graph_name=self.__class__.__name__,
        )

    def run(self) -> List[AIPaper]:
        """
        执行流程并返回处理后的 `AIPaper` 列表
        """
        sel_cfg = self.source or {}
        top_n = int(sel_cfg.get("top_n", 0) or 0)
        ids = sel_cfg.get("ids") or []
        subject = sel_cfg.get("subject")
        self.logger.info(
            f"流程图——从数据库选择开始 ids_count={len(ids)} top_n={top_n} subject={(subject or '')}"
        )
        started = time.time()
        papers: List[AIPaper] = []
        try:
            if ids:
                for pid in ids:
                    try:
                        p = self.db.get_paper_by_id(int(pid))
                        if p:
                            papers.append(p)
                    except Exception:
                        continue
            else:
                all_papers = self.db.list_papers(subject)
                papers = all_papers[:top_n] if top_n > 0 else all_papers
        except Exception:
            papers = []

        inputs = {
            "user_prompt": self.prompt,
            "papers": papers,
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

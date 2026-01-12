"""
文献总结节点（概览）

生成面向研究人员的中文文献概览，总结原文关键信息，尊重原文结构与层级；写入与 Markdown 同目录的 `.overview.md` 文件，并更新数据库 `overviewLink`。
"""

import os
from typing import List, Optional

from scrapegraphai.utils import get_logger
from scrapegraphai.nodes.base_node import BaseNode
try:
    from .db_manager import DatabaseManager, AIPaper
except ImportError:
    from AIpaper2.Nodes.db_manager import DatabaseManager, AIPaper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class DocumentOverviewNode(BaseNode):
    """
    DocumentOverviewNode
    
    文献总结节点：读取 `mdLink` 的内容，生成中文文献概览 Markdown。
    node_config 支持：
    - db_path: 数据库路径
    - force_rebuild: 是否强制重建已存在的概览文件
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "DocumentOverview",
    ):
        """
        初始化节点
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger()
        cfg = self.node_config or {}
        self.db_path = cfg.get("db_path")
        self.db = DatabaseManager(self.db_path)
        self.force_rebuild = bool(cfg.get("force_rebuild", False))

    def _llm_overview(self, md_path: str) -> str:
        """
        使用大模型生成中文文献概览 Markdown（尊重原文结构与层级）
        """
        if not (md_path and os.path.exists(md_path)):
            return ""
        if getattr(self, "llm_model", None) is None:
            return ""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            prompt = PromptTemplate.from_template(
                "只输出结构化的 Markdown 文档，不得包含任何解释或模板外内容。\n"
                "目标：面向研究人员的中文文献概览，在严格尊重原论文的章节标题、层级与顺序的前提下，对各章节内容进行浓缩与重述。\n"
                "要求：\n"
                "- 保留原文的章节标题与编号（如 1./1.1/A./A.1），不得新增顶级章节；\n"
                "- 在各原章节下以 2-6 条要点或短段精炼关键信息（概念、方法名称、数据集与指标名称、核心结论）；\n"
                "- 尽量保留可量化指标（数值、提升幅度、显著性等）；\n"
                "- 删除冗长叙事、无关示例与重复性说明，保留必要上下文保障可读性；\n"
                "- 参考文献与附录：保留编号但不展开正文，仅在需要时以一句话概括要点；\n"
                "- 不需要提取或重复元信息（标题/作者/DOI/日期等）；\n"
                "- 所有内容必须基于原文，避免编造；无法明确的信息使用“暂无信息”。\n\n"
                "{content}"
            )
            chain = prompt | self.llm_model | StrOutputParser()
            return chain.invoke({"content": content}).strip()
        except Exception as e:
            self.logger.error(f"LLM 概览失败 path={md_path} err={e}")
            return ""

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 为每条论文生成 `.overview.md` 文件
        - 优先使用大模型生成概览，若失败则跳过
        - 更新数据库 `overviewLink` 为概览文件路径
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        papers: List[AIPaper] = state[input_keys[0]]

        llm_enabled = getattr(self, "llm_model", None) is not None
        self.logger.info(f"文献概览节点——开始处理 papers={len(papers)} llm_enabled={llm_enabled}")
        updated: List[AIPaper] = []
        generated = 0
        reused = 0
        skipped = 0
        failed = 0
        for idx, p in enumerate(papers, start=1):
            try:
                if not p.mdLink or not os.path.exists(p.mdLink):
                    skipped += 1
                    self.logger.info(f"文献概览节点——第 {idx}/{len(papers)} 篇跳过：缺少 mdLink")
                    updated.append(p)
                    continue
                base_dir = os.path.dirname(p.mdLink)
                overview_path = os.path.join(base_dir, f"{int(p.id)}.overview.md") if p.id is not None else (os.path.splitext(p.mdLink)[0] + ".overview.md")
                need_generate = self.force_rebuild or (not os.path.exists(overview_path))
                if need_generate:
                    if not llm_enabled:
                        skipped += 1
                        updated.append(p)
                        self.logger.info(f"文献概览节点——第 {idx}/{len(papers)} 篇跳过：未配置 LLM")
                        continue
                    overview_text = self._llm_overview(p.mdLink)
                    if not overview_text:
                        failed += 1
                        updated.append(p)
                        self.logger.error(f"文献概览节点——第 {idx}/{len(papers)} 篇生成失败 id={p.id}")
                        continue
                    with open(overview_path, "w", encoding="utf-8") as f:
                        f.write(overview_text)
                    if p.id is not None:
                        self.db.update_fields(int(p.id), {"overviewLink": overview_path})
                    p.overviewLink = overview_path
                    generated += 1
                    self.logger.info(
                        f"文献概览节点——第 {idx}/{len(papers)} 篇生成成功 id={p.id} overview={overview_path} force_rebuild={self.force_rebuild}"
                    )
                else:
                    reused += 1
                    if p.id is not None:
                        self.db.update_fields(int(p.id), {"overviewLink": overview_path})
                    p.overviewLink = overview_path
                    self.logger.info(f"文献概览节点——第 {idx}/{len(papers)} 篇复用已存在 overview={overview_path}")
                updated.append(p)
            except Exception as e:
                self.logger.error(f"写入概览失败 id={p.id} md={p.mdLink} err={e}")
                failed += 1
                updated.append(p)

        self.logger.info(
            f"文献概览节点——处理完成 generated={generated} reused={reused} skipped={skipped} failed={failed}"
        )
        state.update({self.output[0]: updated})
        return state

def run_test_for_id(paper_id: int, db_path: Optional[str] = None) -> None:
    """
    使用指定的论文 ID 测试文献概览生成
    """
    logger = get_logger()
    from .db_manager import DatabaseManager
    logger.info(f"测试开始：paper_id={paper_id} db_path={db_path or DatabaseManager.DEFAULT_DB_PATH}")
    db = DatabaseManager(db_path)
    paper = db.get_paper_by_id(int(paper_id))
    if paper is None:
        print(f"未找到指定 ID 的论文记录：{paper_id}")
        return
    if not paper.mdLink or not os.path.exists(paper.mdLink):
        print(f"该记录缺少有效的 Markdown 文件：id={paper_id} mdLink={paper.mdLink or ''}")
        return
    node = DocumentOverviewNode(
        input="papers",
        output=["papers"],
        node_config={"db_path": db_path, "force_rebuild": True},
    )
    try:
        from AIpaper2.google_scholar_paper_main import build_complex_llm
        node.llm_model = build_complex_llm()
    except Exception:
        node.llm_model = None
    state = {"papers": [paper]}
    result = node.execute(state)
    out = result.get("papers", [])
    if out:
        p = out[0]
        print(f"Overview 生成完成：overview={getattr(p, 'overviewLink', '')}")
    else:
        print("Overview 未生成或生成失败")

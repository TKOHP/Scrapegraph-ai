"""
文献深度分析与评价节点

生成面向研究人员的深度总结与评价 Markdown，不复刻原文结构，不重复元信息；写入与 Markdown 同目录的 `.analysis.md` 文件，并更新数据库 `analysisLink`。
"""

import os
from typing import List, Optional

from scrapegraphai.utils import get_logger
from scrapegraphai.nodes.base_node import BaseNode
try:
    from .db_manager import DatabaseManager, AIPaper
except ImportError:
    from AIpaper.Nodes.db_manager import DatabaseManager, AIPaper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ResearchAnalysisNode(BaseNode):
    """
    ResearchAnalysisNode
    
    文献深度分析与评价节点：读取 `mdLink` 的内容，并结合 `meta` 作为辅助上下文，生成深度总结与评价 Markdown。
    node_config 支持：
    - db_path: 数据库路径
    - force_rebuild: 是否强制重建已存在的分析文件
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "ResearchAnalysis",
    ):
        """
        初始化节点
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger()
        cfg = self.node_config or {}
        self.db_path = cfg.get("db_path", "AIpaper/data/google_scholar_papers.db")
        self.db = DatabaseManager(self.db_path)
        self.force_rebuild = bool(cfg.get("force_rebuild", False))

    def _llm_analysis(self, md_path: str, meta_text: Optional[str] = None) -> str:
        """
        使用大模型生成文献的深度分析与评价 Markdown
        """
        if not (md_path and os.path.exists(md_path)):
            return ""
        if getattr(self, "llm_model", None) is None:
            return ""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            prompt = PromptTemplate.from_template(
                "只输出 Markdown 文档，不得包含任何解释或模板外内容。\n"
                "目标：进行面向研究人员的深度总结与分析，突出关键信息与实证证据，便于快速浏览与判断文献价值。\n"
                "要求：\n"
                "- 不需要提取或重复元信息（标题/作者/DOI/日期等）；\n"
                "- 不必复刻原文结构，可按下述结构组织；\n"
                "- 所有判断必须基于原文内容，避免编造；无法明确的信息使用“暂无信息”或“原文未明确说明”；\n"
                "- 优先给出可量化的指标与结果（数值、区间、提升幅度、显著性等）。\n\n"
                "输入说明：你将获得两部分上下文——meta 与 content。\n"
                "- meta：来自前序节点的元信息（通常为 JSON 字符串），仅用于参考与消歧，不要在输出中逐条罗列；\n"
                "- content：论文的 Markdown 内容，是总结与分析的主要依据；若 meta 与 content 冲突，以 content 为准。\n\n"
                "# 文献深度分析与评价\n"
                "## 一页速览\n"
                "- 研究主题与核心问题：用 1-2 句概括\n"
                "- 核心结论：用 1-2 句直接给出\n"
                "- 关键贡献：不超过 3 点，简明扼要\n"
                "- 方法或技术路线：一句话说明主要方法/模型/数据来源\n"
                "- 代表性结果与指标：列出 3-6 个可量化结果（含指标名与数值/幅度）\n\n"
                "## 背景与动机\n"
                "说明问题背景、研究动机、现实痛点与目标受众（行业/学术/政策等）。\n\n"
                "## 方法与证据\n"
                "概述方法/模型/框架与数据来源；明确评价指标与实验设置（样本量、对照、统计检验等）。\n"
                "列出关键发现（不超过 6 点），尽量提供量化证据与显著性说明。\n\n"
                "## 相对已有工作的差异与创新\n"
                "总结与相关工作的差异化、增量价值与可推广性；如为综述/报告，请说明纳入范围与比较维度。\n\n"
                "## 应用与影响\n"
                "给出潜在应用场景（行业/学术/政策等）与可能影响；如有落地案例或可复用组件，请指出。\n\n"
                "## 局限与未来方向\n"
                "- 局限：不超过 4 条（如数据偏差、外部效度不足、方法假设较强等）\n"
                "- 未来工作：不超过 4 条（如扩展数据、改进方法、更多场景验证等）\n\n"
                "## 质量与可信度评估\n"
                "- 方法严谨性：\n"
                "- 数据与实验质量：\n"
                "- 证据充分性：\n"
                "- 可复现性：\n"
                "- 叙述清晰度：\n"
                "- 综合评级（优秀/良好/一般/较弱）：给出一个等级并用不超过 50 字解释理由。\n\n"
                "【meta】\n"
                "{meta}\n\n"
                "【content】\n"
                "{content}"
            )
            chain = prompt | self.llm_model | StrOutputParser()
            return chain.invoke({"content": content, "meta": (meta_text or "")}).strip()
        except Exception as e:
            self.logger.error(f"LLM 深度分析失败 path={md_path} err={e}")
            return ""

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 为每条论文生成 `.analysis.md` 文件
        - 优先使用大模型生成深度分析；无模型则跳过
        - 更新数据库 `analysisLink` 为分析文件路径
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        papers: List[AIPaper] = state[input_keys[0]]

        llm_enabled = getattr(self, "llm_model", None) is not None
        self.logger.info(f"深度分析节点——开始处理 papers={len(papers)} llm_enabled={llm_enabled}")
        updated: List[AIPaper] = []
        generated = 0
        reused = 0
        skipped = 0
        failed = 0
        for idx, p in enumerate(papers, start=1):
            try:
                if not p.mdLink or not os.path.exists(p.mdLink):
                    skipped += 1
                    self.logger.info(f"深度分析节点——第 {idx}/{len(papers)} 篇跳过：缺少 mdLink")
                    updated.append(p)
                    continue
                base_dir = os.path.dirname(p.mdLink)
                analysis_path = os.path.join(base_dir, f"{int(p.id)}.analysis.md") if p.id is not None else (os.path.splitext(p.mdLink)[0] + ".analysis.md")
                need_generate = self.force_rebuild or (not os.path.exists(analysis_path))
                if need_generate:
                    if not llm_enabled:
                        skipped += 1
                        updated.append(p)
                        self.logger.info(f"深度分析节点——第 {idx}/{len(papers)} 篇跳过：未配置 LLM")
                        continue
                    analysis_text = self._llm_analysis(p.mdLink, getattr(p, "meta", None))
                    if not analysis_text:
                        failed += 1
                        updated.append(p)
                        self.logger.error(f"深度分析节点——第 {idx}/{len(papers)} 篇生成失败 id={p.id}")
                        continue
                    with open(analysis_path, "w", encoding="utf-8") as f:
                        f.write(analysis_text)
                    if p.id is not None:
                        self.db.update_fields(int(p.id), {"analysisLink": analysis_path})
                    p.analysisLink = analysis_path
                    generated += 1
                    self.logger.info(
                        f"深度分析节点——第 {idx}/{len(papers)} 篇生成成功 id={p.id} analysis={analysis_path} force_rebuild={self.force_rebuild}"
                    )
                else:
                    reused += 1
                    if p.id is not None:
                        self.db.update_fields(int(p.id), {"analysisLink": analysis_path})
                    p.analysisLink = analysis_path
                    self.logger.info(f"深度分析节点——第 {idx}/{len(papers)} 篇复用已存在 analysis={analysis_path}")
                updated.append(p)
            except Exception as e:
                self.logger.error(f"写入分析失败 id={p.id} md={p.mdLink} err={e}")
                failed += 1
                updated.append(p)

        self.logger.info(
            f"深度分析节点——处理完成 generated={generated} reused={reused} skipped={skipped} failed={failed}"
        )
        state.update({self.output[0]: updated})
        return state

def run_test_for_id(paper_id: int, db_path: str = "AIpaper/data/google_scholar_papers.db") -> None:
    """
    使用指定的论文 ID 测试文献深度分析与评价
    """
    logger = get_logger()
    logger.info(f"测试开始：paper_id={paper_id} db_path={db_path}")
    db = DatabaseManager(db_path)
    paper = db.get_paper_by_id(int(paper_id))
    if paper is None:
        print(f"未找到指定 ID 的论文记录：{paper_id}")
        return
    if not paper.mdLink or not os.path.exists(paper.mdLink):
        print(f"该记录缺少有效的 Markdown 文件：id={paper_id} mdLink={paper.mdLink or ''}")
        return
    node = ResearchAnalysisNode(
        input="papers",
        output=["papers"],
        node_config={"db_path": db_path, "force_rebuild": True},
    )
    try:
        from AIpaper.google_scholar_paper_main import build_complex_llm
        node.llm_model = build_complex_llm()
    except Exception:
        node.llm_model = None
    state = {"papers": [paper]}
    result = node.execute(state)
    out = result.get("papers", [])
    if out:
        print("Analysis 生成完成")
    else:
        print("Analysis 未生成或生成失败")


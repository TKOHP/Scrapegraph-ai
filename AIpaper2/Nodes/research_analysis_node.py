"""
文献深度分析与评价节点

生成面向研究人员的深度总结与评价 Markdown，不复刻原文结构，不重复元信息；写入与 Markdown 同目录的 `.analysis.md` 文件，并更新数据库 `analysisLink`。
"""

import os
import json
import re
from typing import List, Optional

from scrapegraphai.utils import get_logger
from scrapegraphai.nodes.base_node import BaseNode
try:
    from .db_manager import DatabaseManager, AIPaper
except ImportError:
    from AIpaper2.Nodes.db_manager import DatabaseManager, AIPaper
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
        self.db_path = cfg.get("db_path")
        self.db = DatabaseManager(self.db_path)
        self.force_rebuild = bool(cfg.get("force_rebuild", False))

    def _build_meta_markdown(self, paper: AIPaper) -> str:
        """
        构造基于 AIPaper 的 Markdown 表格文本（仅保留固定字段并紧凑排版）
        
        参数：
        - paper: AIPaper 对象

        返回：
        - 将 meta 转为两列表格（字段 | 值）的 Markdown 字符串；若 meta 为空返回空字符串
        """
        if not paper.meta:
            return ""
        obj = paper.meta
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                obj = {}
        if not isinstance(obj, dict):
            obj = {}

        def _compact_text(s: str) -> str:
            s = s.strip()
            if not s:
                return "暂无信息"
            parts = re.split(r"[,\uFF0C;\u3001]+", s)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                s = "、".join(parts)
            s = s.replace("|", "&#124;").replace("\r", "").replace("\n", " ")
            return s or "暂无信息"

        def _format_value(v) -> str:
            if v is None:
                return "暂无信息"
            if isinstance(v, list):
                items = []
                for it in v:
                    if isinstance(it, (dict, list)):
                        it = json.dumps(it, ensure_ascii=False)
                    else:
                        it = str(it)
                    it = it.strip()
                    if it:
                        items.append(it.replace("|", "&#124;").replace("\r", "").replace("\n", " "))
                return "、".join(items) if items else "暂无信息"
            if isinstance(v, (dict, tuple)):
                s = json.dumps(v, ensure_ascii=False)
                return _compact_text(s)
            return _compact_text(str(v))

        title = _format_value(obj.get("title"))
        keywords = _format_value(obj.get("keywords"))
        publish_time = _format_value(paper.publishTime)
        venue = _format_value(obj.get("venue"))
        doi = _format_value(obj.get("doi"))
        subjects = _format_value(paper.subject)
        paper_type_text = _format_value(paper.type)
        lines = [
            f"| 标题 | {title} |",
            f"| 关键词 | {keywords} |",
            f"| 发表时间 | {publish_time} | 期刊/会议 | {venue} |",
            f"| DOI | {doi} | 主题领域 | {subjects} |",
            f"| 论文类型 | {paper_type_text} |",
        ]
        return "\n".join(lines) + "\n"

    def _get_prompt_for_type(self, paper_type: Optional[str]) -> PromptTemplate:
        """
        根据论文类型返回对应的提示词模板
        """
        t = (paper_type or "").strip()
        if t == "综述型论文":
            return PromptTemplate.from_template(
                "只输出 Markdown 文档，不得包含任何解释或模板外内容。\n"
                "目标：对综述型论文进行系统化的综合总结与评价，强调范围、方法、比较维度与结论的可靠性。\n"
                "要求：\n"
                "- 不需要逐条罗列元信息；\n"
                "- 聚焦文献收集范围、筛选标准、分类框架与比较维度；\n"
                "- 提炼关键主题、趋势与代表性工作，避免编造；\n"
                "- 给出可量化统计（样本数量、覆盖领域、时间范围等）与明确结论。\n\n"
                "# 综述型文献深度分析\n"
                "## 一页速览\n"
                "- 综述范围与主题：\n"
                "- 收录与筛选标准：\n"
                "- 分类与比较维度：\n"
                "- 关键主题与趋势：不超过 5 点\n"
                "- 综合结论：不超过 3 点\n\n"
                "## 文献收集与筛选\n"
                "说明数据源、检索策略、时间范围、收录与剔除标准，并给出基本统计。\n\n"
                "## 分类框架与比较维度\n"
                "给出分类方法与比较维度（如方法、数据、指标、场景等），并陈述代表性工作与差异。\n\n"
                "## 综合主题与趋势\n"
                "总结该领域的主要主题、研究热点与演进趋势，尽量指出证据与统计信息。\n\n"
                "## 应用与影响\n"
                "针对行业/学术/政策给出洞见与潜在应用，指出成熟度与可落地性。\n\n"
                "## 局限与未来方向\n"
                "- 局限：不超过 4 条\n"
                "- 未来工作：不超过 4 条\n\n"
                "## 质量与可信度评估\n"
                "- 收集与筛选的可靠性：\n"
                "- 分类与比较方法的合理性：\n"
                "- 结论与证据的一致性：\n"
                "- 综合评级（优秀/良好/一般/较弱）：给出一个等级并用不超过 50 字解释理由。\n\n"
                "【meta】\n"
                "{meta}\n\n"
                "【content】\n"
                "{content}"
            )
        elif t == "研究型论文":
            return PromptTemplate.from_template(
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
        else:
            return PromptTemplate.from_template(
                "只输出 Markdown 文档，不得包含任何解释或模板外内容。\n"
                "目标：针对非标准研究/综述类文档进行结构化解析与评价，强调可读性与信息提炼。\n"
                "要求：\n"
                "- 不需要逐条罗列元信息；\n"
                "- 根据内容特征组织合理结构；\n"
                "- 所有判断必须基于原文内容，避免编造。\n\n"
                "# 文献内容解析与评价\n"
                "## 一页速览\n"
                "- 文档主题与核心信息：\n"
                "- 主要结论或观点：\n"
                "- 关键信息点：不超过 5 条\n\n"
                "## 内容结构与要点\n"
                "梳理文档结构与关键段落，提炼重要信息与证据。\n\n"
                "## 应用与影响\n"
                "指出可能的应用场景与影响面。\n\n"
                "## 局限与改进空间\n"
                "- 局限：不超过 4 条\n"
                "- 改进建议：不超过 4 条\n\n"
                "【meta】\n"
                "{meta}\n\n"
                "【content】\n"
                "{content}"
            )

    def _llm_analysis(self, md_path: str, meta_text: Optional[str] = None, paper_type: Optional[str] = None) -> str:
        """
        使用大模型生成文献的深度分析与评价 Markdown（加入分段截断逻辑防止超限）
        """
        if not (md_path and os.path.exists(md_path)):
            return ""
        if getattr(self, "llm_model", None) is None:
            return ""

        # 安全阈值设置
        MAX_INPUT_CHARS = 30720
        SAFE_BUFFER = 1024
        
        # 估算提示词固定部分长度
        prompt_template = self._get_prompt_for_type(paper_type)
        # 用空串填充变量来测量模板基础长度
        dummy_prompt = prompt_template.format(content="", meta="")
        prompt_len = len(dummy_prompt)
        
        # 处理 meta 长度：限制 meta 最多占用 4096 字符，避免挤占 content
        meta_str = meta_text or ""
        if len(meta_str) > 4096:
            meta_str = meta_str[:4096] + "\n...(meta truncated)..."
        
        # 计算 content 可用预算
        available_len = MAX_INPUT_CHARS - prompt_len - len(meta_str) - SAFE_BUFFER
        if available_len < 1000:
            available_len = 1000  # 至少保留 1000 字符给正文，即使这意味着轻微超限风险

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 若未超限，直接生成
            if len(content) <= available_len:
                chain = prompt_template | self.llm_model | StrOutputParser()
                return chain.invoke({"content": content, "meta": meta_str}).strip()
            
            # 若超限，进行 Map-Reduce 处理
            self.logger.info(f"内容过长 ({len(content)} chars)，启动 Map-Reduce 分析...")
            chunks = self._split_markdown_by_headings(content, available_len)
            # “语义分段 + 递进式总结”
            # Map 阶段：从每个分段提取关键信息
            map_prompt = PromptTemplate.from_template(
                "请阅读以下文档片段，提取其中的关键信息要点。重点关注：\n"
                "- 研究背景、动机与核心问题\n"
                "- 提出的方法、模型架构、算法创新点\n"
                "- 实验设置、数据集、对比基线\n"
                "- 关键实验结果、量化指标（保留数值）\n"
                "- 结论、局限性与未来方向\n"
                "请以简洁的列表形式输出要点，不要生成完整的文章结构。\n\n"
                "【文档片段】\n{content}"
            )
            
            extracted_infos = []
            for i, chunk in enumerate(chunks):
                self.logger.info(f"正在提取第 {i+1}/{len(chunks)} 个分段的信息...")
                chain = map_prompt | self.llm_model | StrOutputParser()
                res = chain.invoke({"content": chunk})
                extracted_infos.append(res.strip())
            
            # Reduce 阶段：基于汇总信息生成最终报告
            combined_info = "\n\n=== 片段信息分隔 ===\n\n".join(extracted_infos)
            
            # 检查汇总后的信息是否依然超长，若超长则截断（保留前30k字符，通常提取后的信息会小很多）
            if len(combined_info) > MAX_INPUT_CHARS - prompt_len - len(meta_str) - SAFE_BUFFER:
                 combined_info = combined_info[:(MAX_INPUT_CHARS - prompt_len - len(meta_str) - SAFE_BUFFER)] + "\n...(truncated)..."

            self.logger.info("正在基于汇总信息生成最终深度分析...")
            # 使用原始的分析 Prompt，但传入的是汇总后的信息
            chain = prompt_template | self.llm_model | StrOutputParser()
            return chain.invoke({"content": combined_info, "meta": meta_str}).strip()

        except Exception as e:
            self.logger.error(f"LLM 深度分析失败 path={md_path} err={e}")
            return ""

    def _split_markdown_by_headings(self, text: str, max_len: int) -> List[str]:
        """
        按 Markdown 标题分段，确保每段不超过 max_len
        """
        lines = text.splitlines()
        chunks = []
        current_chunk = []
        current_len = 0
        
        for line in lines:
            line_len = len(line) + 1 # +1 for newline
            
            # 遇到顶级或二级标题，且当前块已有内容，尝试切分
            is_heading = line.strip().startswith("#") or re.match(r'^\d+\.', line.strip())
            
            if is_heading and current_len > 0:
                # 如果加上新行还未超限，且当前块比较小，可以选择不切分（这里简化为遇到大标题就切，或者凑满切）
                # 策略：尽量凑满 max_len，但遇到大标题是比较好的切分点
                # 这里采用：只要加了这行会超限，或者当前块已经很大了(> max_len * 0.8)且遇到标题，就切分
                if (current_len + line_len > max_len) or (current_len > max_len * 0.8):
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
            
            # 如果单行本身就极长（超过 max_len），需要强制切分
            if line_len > max_len:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                # 对超长行进行强制切分
                sub_chunks = self._chunk_by_length(line, max_len)
                chunks.extend(sub_chunks)
                continue
                
            current_chunk.append(line)
            current_len += line_len
            
            # 如果累积长度超过限制，强制切分
            if current_len >= max_len:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
                
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks

    def _chunk_by_length(self, text: str, max_len: int) -> List[str]:
        """
        按长度强制切分文本
        """
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]


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
                    meta_text = self._build_meta_markdown(p)
                    analysis_text = self._llm_analysis(p.mdLink, meta_text, getattr(p, "type", None))
                    heading_line = "# 文献深度分析与评价\n"
                    cleaned_analysis = analysis_text
                    if cleaned_analysis.strip().startswith("# 文献深度分析与评价"):
                        tmp = cleaned_analysis.strip()
                        nl = tmp.find("\n")
                        cleaned_analysis = tmp[nl + 1:] if nl != -1 else ""
                    total_text = heading_line + (meta_text or "") + cleaned_analysis
                    if not total_text.strip():
                        failed += 1
                        updated.append(p)
                        self.logger.error(f"深度分析节点——第 {idx}/{len(papers)} 篇生成失败 id={p.id}")
                        continue
                    with open(analysis_path, "w", encoding="utf-8") as f:
                        f.write(total_text)
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

def run_test_for_id(paper_id: int, db_path: Optional[str] = None) -> None:
    """
    使用指定的论文 ID 测试文档分类与元信息抽取
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
    node = ResearchAnalysisNode(
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
        print("Analysis 生成完成")
    else:
        print("Analysis 未生成或生成失败")

def main() -> None:
    """
    本地测试入口：使用固定的 paper_id 调用深度分析测试
    """
    # 在此直接设置需要测试的论文 ID
    paper_id = 197
    from .db_manager import DatabaseManager
    db_path = DatabaseManager.DEFAULT_DB_PATH
    run_test_for_id(paper_id, db_path)

if __name__ == "__main__":
    main()


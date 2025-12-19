"""
文档分类节点

根据 Markdown 内容结合大模型进行分类与摘要生成，并填充 `meta`、`subject`、`publishTime` 字段。
当未提供大模型配置时，回退到规则提取。
"""

import os
import re
from typing import List, Optional, Dict

from ..utils import get_logger
from ..nodes.base_node import BaseNode
from .db_manager import DatabaseManager, AIPaper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class DocumentClassifyNode(BaseNode):
    """
    文档分类与元信息提取节点
    
    输入 `papers & subjects`，读取每个论文的 Markdown，提取标题、关键词、摘要、发布时间等信息。
    结果以 JSON 字符串写入 `meta` 字段，并更新数据库。
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "DocumentClassify",
    ):
        """
        初始化节点
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger(__name__)
        cfg = self.node_config or {}
        self.db_path = cfg.get("db_path", "data/google_scholar_papers.db")
        self.db = DatabaseManager(self.db_path)

    def _extract_meta(self, md_path: str) -> Dict[str, str]:
        """
        从 Markdown 文件中提取元信息（规则法）
        """
        meta = {"title": "", "keywords": "", "summary": "", "first_page": ""}
        if not (md_path and os.path.exists(md_path)):
            return meta
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = [ln.strip() for ln in content.splitlines()]
            for ln in lines:
                if not meta["title"] and ln.startswith("# "):
                    meta["title"] = ln[2:].strip()
                if not meta["keywords"]:
                    m = re.search(r"(?:关键词|Keywords?)\s*[:：]\s*(.+)", ln, flags=re.IGNORECASE)
                    if m:
                        meta["keywords"] = m.group(1).strip()
            meta["first_page"] = "\n".join(lines[:60])
            if not meta["summary"]:
                text = "\n".join(lines)
                para = re.split(r"\n\s*\n", text)
                meta["summary"] = (para[0] if para else text[:600])[:1200]
        except Exception as e:
            self.logger.error(f"提取元信息失败 path={md_path} err={e}")
        return meta

    def _extract_publish_time(self, meta_text: str) -> str:
        """
        从文本中提取发布时间（优先匹配 YYYY-MM-DD，其次匹配年份）
        """
        m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", meta_text)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        y = re.search(r"(19|20)\d{2}", meta_text)
        return y.group(0) if y else ""

    def _llm_classify_subject(self, content: str) -> str:
        """
        使用大模型对文档进行中文领域分类，输出 ≤12 字的标签
        """
        try:
            if getattr(self, "llm_model", None) is None:
                return ""
            prompt = PromptTemplate.from_template(
                "请阅读以下论文内容，输出一个最贴切的中文领域标签（≤12字），仅输出该标签：\n\n{content}"
            )
            chain = prompt | self.llm_model | StrOutputParser()
            label = chain.invoke({"content": content}) or ""
            label = re.sub(r"\s+", " ", label).strip()
            return label[:20]
        except Exception as e:
            self.logger.error(f"LLM 分类失败: {e}")
            return ""

    def _llm_summarize(self, content: str) -> str:
        """
        使用大模型生成中文摘要文本
        """
        try:
            if getattr(self, "llm_model", None) is None:
                return ""
            prompt = PromptTemplate.from_template(
                "请基于以下论文内容用中文生成精炼摘要（不超过500字）：\n\n{content}\n\n只输出摘要文本。"
            )
            chain = prompt | self.llm_model | StrOutputParser()
            summary = chain.invoke({"content": content}) or ""
            summary = summary.strip()
            return summary
        except Exception as e:
            self.logger.error(f"LLM 摘要失败: {e}")
            return ""

    def _llm_select_subjects(self, content: str, subjects: List[str]) -> List[str]:
        """
        使用大模型从给定主题池中进行多标签选择，返回命中的主题列表
        """
        try:
            if getattr(self, "llm_model", None) is None or not subjects:
                return []
            options = "、".join(subjects)
            prompt = PromptTemplate.from_template(
                "请阅读以下论文内容，从给定主题池中选择所有相关的中文主题，"
                "只输出所选主题，使用中文逗号分隔，且必须从下列选项中选择：\n\n"
                "主题池：{options}\n\n内容：\n{content}"
            )
            chain = prompt | self.llm_model | StrOutputParser()
            resp = chain.invoke({"options": options, "content": content}) or ""
            # 解析为列表并仅保留合法选项
            raw = [s.strip() for s in re.split(r"[，,]", resp) if s.strip()]
            valid = []
            for s in raw:
                if s in subjects and s not in valid:
                    valid.append(s)
            return valid
        except Exception as e:
            self.logger.error(f"LLM 多标签主题选择失败: {e}")
            return []

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 读取每条论文的 Markdown
        - 优先使用大模型进行主题池内的多标签分类与摘要，回填 `subject`（逗号分隔）与 `meta.summary`
        - 使用规则提取标题/关键词，并推断 `publishTime`
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        papers: List[AIPaper] = state[input_keys[0]]
        subjects: List[str] = state[input_keys[1]]

        llm_enabled = getattr(self, "llm_model", None) is not None
        self.logger.info(
            f"文档分类节点——开始处理 papers={len(papers)} subjects_pool={len(subjects)} llm_enabled={llm_enabled}"
        )
        updated: List[AIPaper] = []
        skipped = 0
        classified = 0
        failed = 0
        for idx, p in enumerate(papers, start=1):
            try:
                if not p.mdLink or not os.path.exists(p.mdLink):
                    skipped += 1
                    self.logger.info(f"文档分类节点——第 {idx}/{len(papers)} 篇跳过：缺少 mdLink")
                    updated.append(p)
                    continue
                meta_dict = self._extract_meta(p.mdLink or "")
                content_for_llm = meta_dict.get("first_page", "")
                selected_subjects = self._llm_select_subjects(content_for_llm, subjects) if content_for_llm else []
                llm_summary = self._llm_summarize(content_for_llm) if content_for_llm else ""
                if llm_summary:
                    meta_dict["summary"] = llm_summary
                meta_text = f"Title: {meta_dict.get('title','')}\nKeywords: {meta_dict.get('keywords','')}\nSummary: {meta_dict.get('summary','')}\n"
                publish_time = self._extract_publish_time(meta_text + (meta_dict.get("first_page", "") or ""))
                updates = {
                    "meta": str(meta_dict),
                    "subject": (",".join(selected_subjects) if selected_subjects else (p.subject or "")),
                    "publishTime": publish_time or p.publishTime,
                }
                if p.id is not None:
                    self.db.update_fields(int(p.id), updates)
                p.meta = updates["meta"]
                p.subject = updates["subject"]
                p.publishTime = updates["publishTime"]
                updated.append(p)
                classified += 1
                self.logger.info(
                    f"文档分类节点——第 {idx}/{len(papers)} 篇完成 id={p.id} subjects={p.subject or ''} "
                    f"publishTime={p.publishTime or ''}"
                )
            except Exception as e:
                self.logger.error(f"文档分类失败 id={p.id} err={e}")
                failed += 1
                updated.append(p)

        self.logger.info(
            f"文档分类节点——处理完成 classified={classified} skipped={skipped} failed={failed}"
        )
        state.update({self.output[0]: updated})
        return state

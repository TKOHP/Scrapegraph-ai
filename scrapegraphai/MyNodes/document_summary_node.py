"""
单文档总结节点

使用大模型对 Markdown 内容生成中文总结，保存为同目录的 `.summary.md` 文件并更新数据库。
当未提供大模型配置时，回退到简易规则总结。
"""

import os
from typing import List, Optional

from ..utils import get_logger
from ..nodes.base_node import BaseNode
from .db_manager import DatabaseManager, AIPaper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class DocumentSummaryNode(BaseNode):
    """
    文档总结节点
    
    读取 `mdLink` 的内容，生成简短概要文本并写入 `.summary.md` 文件。
    node_config 支持：
    - db_path: 数据库路径
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "DocumentSummary",
    ):
        """
        初始化节点
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger(__name__)
        cfg = self.node_config or {}
        self.db_path = cfg.get("db_path", "data/google_scholar_papers.db")
        self.db = DatabaseManager(self.db_path)

    def _summarize(self, md_path: str) -> str:
        """
        生成简易总结文本（规则法）
        """
        if not (md_path and os.path.exists(md_path)):
            return "无法生成总结：Markdown 文件不存在或路径非法。"
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            first_section = lines[:60]
            summary = []
            for ln in first_section:
                if ln.startswith("# "):
                    summary.append(f"主题：{ln[2:].strip()}")
                    break
            summary.append("摘要：")
            summary.extend([ln for ln in first_section if not ln.startswith("# ")][:10])
            return "\n".join(summary) + "\n"
        except Exception as e:
            self.logger.error(f"生成总结失败 path={md_path} err={e}")
            return "生成总结过程中发生错误。"

    def _llm_summary(self, md_path: str) -> str:
        """
        使用大模型生成中文总结
        """
        if not (md_path and os.path.exists(md_path)):
            return ""
        if getattr(self, "llm_model", None) is None:
            return ""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            prompt = PromptTemplate.from_template(
                "请阅读以下论文内容，用中文输出结构化总结（标题/关键词/研究问题/方法/创新点/结论），以 Markdown 形式输出：\n\n{content}"
            )
            chain = prompt | self.llm_model | StrOutputParser()
            return chain.invoke({"content": content}).strip()
        except Exception as e:
            self.logger.error(f"LLM 总结失败 path={md_path} err={e}")
            return ""

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 为每条论文生成 `.summary.md` 文件
        - 优先使用大模型生成总结，若失败则使用简易总结
        - 更新数据库 `summaryLink`
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        papers: List[AIPaper] = state[input_keys[0]]

        llm_enabled = getattr(self, "llm_model", None) is not None
        self.logger.info(f"文档总结节点——开始处理 papers={len(papers)} llm_enabled={llm_enabled}")
        updated: List[AIPaper] = []
        generated = 0
        reused = 0
        skipped = 0
        failed = 0
        for idx, p in enumerate(papers, start=1):
            try:
                if not p.mdLink or not os.path.exists(p.mdLink):
                    skipped += 1
                    self.logger.info(f"文档总结节点——第 {idx}/{len(papers)} 篇跳过：缺少 mdLink")
                    updated.append(p)
                    continue
                summary_path = os.path.splitext(p.mdLink)[0] + ".summary.md"
                if not os.path.exists(summary_path):
                    used_llm = False
                    summary_text = self._llm_summary(p.mdLink)
                    if summary_text:
                        used_llm = True
                    else:
                        summary_text = self._summarize(p.mdLink)
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(summary_text)
                    generated += 1
                    self.logger.info(
                        f"文档总结节点——第 {idx}/{len(papers)} 篇生成成功 id={p.id} used_llm={used_llm} "
                        f"summary={summary_path}"
                    )
                else:
                    reused += 1
                    self.logger.info(f"文档总结节点——第 {idx}/{len(papers)} 篇复用已存在 summary={summary_path}")
                if p.id is not None:
                    self.db.update_fields(int(p.id), {"summaryLink": summary_path})
                p.summaryLink = summary_path
                updated.append(p)
            except Exception as e:
                self.logger.error(f"写入总结失败 id={p.id} md={p.mdLink} err={e}")
                failed += 1
                updated.append(p)

        self.logger.info(
            f"文档总结节点——处理完成 generated={generated} reused={reused} skipped={skipped} failed={failed}"
        )
        state.update({self.output[0]: updated})
        return state

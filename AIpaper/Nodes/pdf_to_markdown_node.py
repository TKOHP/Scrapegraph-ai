"""
PDF 转标准化 Markdown 节点

遍历 `AIPaper` 对象，将 PDF 内容提取为 Markdown，保存到与 PDF 同目录、同名不同扩展的文件中。
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


class PdfToMarkdownNode(BaseNode):
    """
    PDF 转 Markdown 节点
    
    将 `pdfLink` 指定的文件解析为文本，并生成简易结构化 Markdown。
    node_config 支持：
    - db_path: 数据库路径
    - force_rebuild: 是否强制重建已存在的 Markdown 文件
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "PdfToMarkdown",
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
        self.format_with_llm = bool(cfg.get("format_with_llm", False))
        self.format_prompt_template = cfg.get("format_prompt_template")

    def _default_format_prompt(self) -> PromptTemplate:
        """
        返回用于学术格式化的默认提示词模板
        """
        tpl = (
            "只输出结构化的 Markdown 文档（不得包含任何解释或非 Markdown 内容）：\n"
            "- 尽力还原原文章节结构与层级，优先保留原文的章节标题、列表、引用、公式与参考文献；\n"
            "- 所有章节标题必须使用 Markdown 层级标记：顶级 '# '，二级 '## '，三级 '### '；编号保留在标题文本中（如 '1.', '1.1', 'A.', 'A.1'）。\n"
            "  示例：\n"
            "  # 1. 标题\n"
            "  ## 1.1 子标题\n"
            "  ### 1.1.1 小节\n"
            "  # A. 标题\n"
            "  ## A.1 子标题\n"
            "  ### A.1.1 小节。\n"
            "- 若原文仅给编号未标注层级，则按编号结构推断：'1.'→一级，'1.1'→二级，'1.1.1'→三级；字母同理 'A.'→一级，'A.1'→二级。\n"
            "- 参考文献编号规范：保留原文编号；如无编号，则按首次出现顺序自 [1] 递增，并统一使用 [n] 形式标注。\n"
            "- 标题从内容中抽取；无法确定时再合理提炼；\n"
            "- 若无法识别原始结构，再按常见学术结构组织：\n"
            "  # 标题\n"
            "  ## 摘要\n"
            "  ## 关键词\n"
            "  ## 引言\n"
            "  ## 方法\n"
            "  ## 实验与结果\n"
            "  ## 讨论\n"
            "  ## 结论\n"
            "  ## 参考文献\n\n"
            "{content}"
        )
        return PromptTemplate.from_template(tpl)

    def _format_markdown_with_llm(self, raw_md: str) -> str:
        """
        使用简易模型对初始 Markdown 进行学术格式化
        """
        if not raw_md or getattr(self, "llm_model", None) is None:
            return raw_md
        try:
            prompt = self._default_format_prompt() if not self.format_prompt_template else PromptTemplate.from_template(self.format_prompt_template)
            chain = prompt | self.llm_model | StrOutputParser()
            formatted = chain.invoke({"content": raw_md}).strip()
            return formatted or raw_md
        except Exception as e:
            self.logger.error(f"LLM 学术格式化失败 err={e}")
            return raw_md

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        使用 PyPDFLoader 提取 PDF 文本，若不可用则回退到空内容
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            text = "\n\n".join([p.page_content for p in pages])
            return text
        except Exception as e:
            self.logger.error(f"解析 PDF 失败 path={pdf_path} err={e}")
            return ""

    def _to_markdown(self, text: str) -> str:
        """
        将纯文本转换为简易 Markdown
        """
        if not text:
            return "# 内容解析失败\n\n"
        lines = [ln.strip() for ln in text.splitlines()]
        md_lines = []
        for i, ln in enumerate(lines):
            if not ln:
                md_lines.append("")
                continue
            if len(ln) < 120 and ln.isupper():
                md_lines.append(f"# {ln.title()}")
            elif i == 0 and len(ln) < 120:
                md_lines.append(f"# {ln}")
            else:
                md_lines.append(ln)
        return "\n".join(md_lines) + "\n"

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 遍历 `papers`，为存在 `pdfLink` 且未生成 `mdLink` 的记录生成 Markdown
        - 保存到同目录同名 `.md` 文件，并更新数据库
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        papers: List[AIPaper] = state[input_keys[0]]

        self.logger.info(f"Markdown节点——开始处理 papers={len(papers)}")
        updated: List[AIPaper] = []
        generated = 0
        reused = 0
        skipped = 0
        failed = 0
        for idx, p in enumerate(papers, start=1):
            try:
                if not p.pdfLink or not os.path.exists(p.pdfLink):
                    skipped += 1
                    self.logger.info(f"Markdown节点——第 {idx}/{len(papers)} 篇跳过：缺少 pdfLink")
                    updated.append(p)
                    continue
                base_dir = os.path.dirname(p.pdfLink)
                md_path = os.path.join(base_dir, f"{int(p.id)}.md") if p.id is not None else (os.path.splitext(p.pdfLink)[0] + ".md")
                need_generate = self.force_rebuild or (not os.path.exists(md_path))
                if need_generate:
                    text = self._extract_text_from_pdf(p.pdfLink)
                    md = self._to_markdown(text)
                    if self.format_with_llm and getattr(self, "llm_model", None) is not None:
                        md = self._format_markdown_with_llm(md)
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(md)
                    self.db.update_fields(int(p.id), {"mdLink": md_path})
                    p.mdLink = md_path
                    generated += 1
                    self.logger.info(
                        f"Markdown节点——第 {idx}/{len(papers)} 篇生成成功 md={md_path} chars={len(md)} force_rebuild={self.force_rebuild}"
                    )
                    updated.append(p)
                else:
                    self.db.update_fields(int(p.id), {"mdLink": md_path})
                    p.mdLink = md_path
                    reused += 1
                    self.logger.info(f"Markdown节点——第 {idx}/{len(papers)} 篇复用已存在 md={md_path}")
                    updated.append(p)
            except Exception as e:
                self.logger.error(f"生成 Markdown 失败 id={p.id} path={p.pdfLink} err={e}")
                failed += 1
                updated.append(p)

        self.logger.info(
            f"Markdown节点——处理完成 generated={generated} reused={reused} skipped={skipped} failed={failed}"
        )
        state.update({self.output[0]: updated})
        return state

# =========================
# 测试入口（单文件运行）
# =========================
def run_test_for_id(paper_id: int, db_path: str = "AIpaper/data/google_scholar_papers.db") -> None:
    """
    使用指定的论文 ID 进行节点功能测试：PDF 转 Markdown
    
    Args:
        paper_id: 数据库中 AIpaper 的主键 ID
        db_path: SQLite 数据库路径（默认 AIpaper/data/google_scholar_papers.db）
    """
    logger = get_logger()
    logger.info(f"测试开始：paper_id={paper_id} db_path={db_path}")
    db = DatabaseManager(db_path)
    paper = db.get_paper_by_id(int(paper_id))
    if paper is None:
        print(f"未找到指定 ID 的论文记录：{paper_id}")
        return
    if not paper.pdfLink or not os.path.exists(paper.pdfLink):
        print(f"该记录缺少有效的 PDF 文件：id={paper_id} pdfLink={paper.pdfLink or ''}")
        return
    node = PdfToMarkdownNode(
        input="papers",
        output=["papers"],
        node_config={
            "db_path": db_path,
            "force_rebuild": True,
            "format_with_llm": True,
        },
    )
    try:
        from AIpaper.google_scholar_paper_main import build_simple_llm
        node.llm_model = build_simple_llm()
    except Exception:
        node.llm_model = None
    state = {"papers": [paper]}
    result = node.execute(state)
    out = result.get("papers", [])
    if out and getattr(out[0], "mdLink", None):
        print(f"Markdown 生成完成：md={out[0].mdLink}")
    else:
        print("Markdown 未生成或生成失败")


if __name__ == "__main__":
    from scrapegraphai.utils import set_verbosity_info, set_formatting
    set_verbosity_info()
    set_formatting()
    try:
        paper_id=60
        run_test_for_id(paper_id)
    except Exception as e:
        print(f"输入或执行发生错误：{e}")

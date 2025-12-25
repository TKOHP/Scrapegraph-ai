"""
文档分类节点

根据 Markdown 内容结合大模型进行分类与摘要生成，并填充 `meta`、`subject`、`publishTime` 字段。
当未提供大模型配置时，回退到规则提取。
"""

import os
import re
import json
from typing import List, Optional, Dict, Any

from scrapegraphai.utils import get_logger
from scrapegraphai.nodes.base_node import BaseNode
try:
    from .db_manager import DatabaseManager, AIPaper
except ImportError:
    from AIpaper.Nodes.db_manager import DatabaseManager, AIPaper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class DocumentClassifyNode(BaseNode):
    """
    文档分类与元信息提取节点
    
    输入 `papers & subjects`，读取每个论文的 Markdown，提取标题、关键词、摘要、发布时间等信息。
    结果以 JSON 字符串写入 `meta` 字段，并更新数据库。
    node_config 支持：
    - db_path: 数据库路径
    - force_rebuild: 是否强制重建分类与元信息（忽略已有字段）
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
        self.logger = get_logger()
        cfg = self.node_config or {}
        self.db_path = cfg.get("db_path", "AIpaper/data/google_scholar_papers.db")
        self.db = DatabaseManager(self.db_path)
        self.force_rebuild = bool(cfg.get("force_rebuild", False))

    def _llm_extract_metadata_and_subjects(self, content: str, subjects: List[str]) -> Dict[str, Any]:
        if getattr(self, "llm_model", None) is None:
            return {}
        options = "、".join(subjects or [])
        tpl = (
            "阅读以下论文的 PDF 文本内容，仅抽取已有的元信息并进行主题分类。\n"
            "只输出 JSON，不要输出任何解释。\n"
            "JSON 字段：\n"
            "- title: 从内容中抽取的标题，字符串；若不可得则留空\n"
            "- keywords: 从内容中抽取的关键词数组，如无法识别则为空数组\n"
            "- abstract: 从内容中抽取到的摘要原文，字符串；若不可得则留空\n"
            "- publish_time: 从内容中抽取的发布时间，必须为 YYYY-MM-DD（具体到日）；不得输出仅年份或年份-月份；若无法识别则留空\n"
            "- venue: 出版来源（如期刊/会议/预印本/硕博论文/技术报告等），字符串；若不可得则留空\n"
            "- authors: 作者数组（每项为作者姓名字符串）；若不可得则为空数组\n"
            "- doi: DOI 标识，字符串；若不可得则留空\n"
            "- publish_type: 发表类型（如：期刊论文/会议论文/预印本/硕博论文/技术报告等），字符串；若不可得则留空\n"
            "- subjects_selected: 从给定主题池中选择的主题数组，只能取自提供的主题池；若无匹配则为空数组\n\n"
            "主题池：{options}\n\n"
            "{content}"
        )
        prompt = PromptTemplate.from_template(tpl)
        chain = prompt | self.llm_model | StrOutputParser()
        resp = chain.invoke({"options": options, "content": content}) or ""
        try:
            data = json.loads(resp)
            if not isinstance(data, dict):
                return {}
            return {
                "title": str(data.get("title", "") or ""),
                "keywords": list(data.get("keywords") or []),
                "abstract": str(data.get("abstract", "") or ""),
                "publish_time": str(data.get("publish_time", "") or ""),
                "venue": str(data.get("venue", "") or ""),
                "authors": list(data.get("authors") or []),
                "doi": str(data.get("doi", "") or ""),
                "publish_type": str(data.get("publish_type", "") or ""),
                "subjects_selected": [s for s in (data.get("subjects_selected") or []) if s in (subjects or [])],
            }
        except Exception:
            return {}

    def _fallback_publish_time(self, content: str) -> str:
        """
        当模型未能识别发布时间时做简单格式补全
        """
        m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", content or "")
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        y = re.search(r"(19|20)\d{2}", content or "")
        return y.group(0) if y else ""

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            text = "\n\n".join([p.page_content for p in pages])
            return text
        except Exception as e:
            self.logger.error(f"解析 PDF 失败 path={pdf_path} err={e}")
            return ""

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 读取每条论文的 PDF 文本
        - 使用大模型统一抽取 title/keywords/abstract/publish_time，并从主题池选择 subjects_selected
        - 更新数据库 meta/subject/publishTime
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
                if not p.pdfLink or not os.path.exists(p.pdfLink):
                    skipped += 1
                    self.logger.info(f"文档分类节点——第 {idx}/{len(papers)} 篇跳过：缺少 pdfLink")
                    updated.append(p)
                    continue
                already_has = bool((p.meta and str(p.meta).strip()) or (p.subject and str(p.subject).strip()) or (p.publishTime and str(p.publishTime).strip()))
                if (not self.force_rebuild) and already_has:
                    skipped += 1
                    self.logger.info(f"文档分类节点——第 {idx}/{len(papers)} 篇复用已存在分类信息 id={p.id}")
                    updated.append(p)
                    continue
                pdf_text = self._extract_text_from_pdf(p.pdfLink)
                extract = self._llm_extract_metadata_and_subjects(pdf_text, subjects)
                if not extract:
                    skipped += 1
                    updated.append(p)
                    continue
                publish_time = extract.get("publish_time") or self._fallback_publish_time(pdf_text)
                updates = {
                    "meta": json.dumps(
                        {
                            "title": extract.get("title") or "",
                            "keywords": extract.get("keywords") or [],
                            "abstract": extract.get("abstract") or "",
                            "venue": extract.get("venue") or "",
                            "authors": extract.get("authors") or [],
                            "doi": extract.get("doi") or "",
                            "publish_type": extract.get("publish_type") or "",
                        },
                        ensure_ascii=False,
                    ),
                    "subject": ",".join(extract.get("subjects_selected") or []) or (p.subject or ""),
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

def run_test_for_id(paper_id: int, db_path: str = "AIpaper/data/google_scholar_papers.db") -> None:
    """
    使用指定的论文 ID 测试文档分类与元信息抽取
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
    try:
        from AIpaper.google_scholar_paper_main import build_simple_llm, SUBJECTS_POOL
        llm = build_simple_llm()
        subjects_pool = SUBJECTS_POOL
    except Exception:
        llm = None
        subjects_pool = []
    node = DocumentClassifyNode(
        input="papers & subjects",
        output=["papers"],
        node_config={"db_path": db_path, "force_rebuild": True},
    )
    node.llm_model = llm
    state = {"papers": [paper], "subjects": subjects_pool}
    result = node.execute(state)
    out = result.get("papers", [])
    if out:
        p = out[0]
        print(
            f"id={getattr(p, 'id', None)} subject={getattr(p, 'subject', '')} "
            f"publishTime={getattr(p, 'publishTime', '')} meta={getattr(p, 'meta', '')}"
        )
    else:
        print("分类未完成或发生错误")


if __name__ == "__main__":
    from scrapegraphai.utils import set_verbosity_info, set_formatting
    set_verbosity_info()
    set_formatting()
    try:
        paper_id = 60
        run_test_for_id(paper_id)
    except Exception as e:
        print(f"输入或执行发生错误：{e}")

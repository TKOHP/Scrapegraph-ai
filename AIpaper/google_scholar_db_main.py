"""
Google Scholar 数据库选择处理主入口

在此文件中设定选择规则（前 N 条或指定 ID 列表），调用 `GoogleScholarDbPaperGraph` 执行完整流程。
"""

from typing import List, Optional, Dict, Any

from AIpaper.Graphs import GoogleScholarDbPaperGraph
from scrapegraphai.utils import set_verbosity_info, set_formatting
from AIpaper.common_settings import (
    SUBJECTS_POOL,
    build_simple_llm,
    build_complex_llm,
    build_graph_config,
)

# 选择前 N 条（按 id 倒序），若为 0 或 None 则不启用
SELECT_TOP_N: Optional[int] = 5

# 指定要处理的 ID 列表（优先于 SELECT_TOP_N）
SELECT_IDS: List[int] = [173]

# 可选：按主题过滤（如需仅处理某一主题）
SELECT_SUBJECT: Optional[str] = None


def build_selection_config() -> Dict[str, Any]:
    """
    构造数据库选择配置字典
    
    返回字段：
    - top_n: 选择最新的前 N 条记录（id 倒序）
    - ids: 指定处理的记录主键列表，优先级高于 top_n
    - subject: 按主题过滤
    """
    return {
        "top_n": SELECT_TOP_N if SELECT_TOP_N else 0,
        "ids": SELECT_IDS or [],
        "subject": SELECT_SUBJECT,
    }


def main():
    """
    主函数，创建并运行流程图（数据库选择）
    """
    set_verbosity_info()
    set_formatting()
    selection_config = build_selection_config()
    simple_llm = build_simple_llm()
    complex_llm = build_complex_llm()
    graph_config = build_graph_config()
    graph = GoogleScholarDbPaperGraph(
        prompt="Google Scholar DB Pipeline",
        selection_config=selection_config,
        subjects=SUBJECTS_POOL,
        config=graph_config,
        simple_llm=simple_llm,
        complex_llm=complex_llm,
    )
    papers = graph.run()
    for p in papers:
        print(
            f"id={getattr(p, 'id', None)} subject={getattr(p, 'subject', '')} "
            f"url={getattr(p, 'urlLink', '')} pdf={getattr(p, 'pdfLink', '')} "
            f"md={getattr(p, 'mdLink', '')} overview={getattr(p, 'overviewLink', '')} "
            f"analysis={getattr(p, 'analysisLink', '')}"
        )


if __name__ == "__main__":
    main()


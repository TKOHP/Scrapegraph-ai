"""
arXiv 订阅论文处理主入口

在此文件中设定查询配置，调用 `ArxivPaperGraph` 执行完整流程。
"""

from typing import List

try:
    from AIpaper.Graphs import ArxivPaperGraph
    from scrapegraphai.utils import set_verbosity_info, set_formatting
    from AIpaper.common_settings import (
        SUBJECTS_POOL,
        build_simple_llm,
        build_complex_llm,
        build_graph_config,
        build_arxiv_config,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from AIpaper.Graphs import ArxivPaperGraph
    from scrapegraphai.utils import set_verbosity_info, set_formatting
    from AIpaper.common_settings import (
        SUBJECTS_POOL,
        build_simple_llm,
        build_complex_llm,
        build_graph_config,
        build_arxiv_config,
    )


def main():
    """
    主函数，创建并运行流程图
    """
    set_verbosity_info()
    set_formatting()
    arxiv_config = build_arxiv_config()
    simple_llm = build_simple_llm()
    complex_llm = build_complex_llm()
    graph_config = build_graph_config()
    graph = ArxivPaperGraph(
        prompt="Arxiv Subscription Pipeline",
        arxiv_config=arxiv_config,
        subjects=SUBJECTS_POOL,
        config=graph_config,
        simple_llm=simple_llm,
        complex_llm=complex_llm,
    )
    papers = graph.run()
    for p in papers:
        print(
            f"id={getattr(p, 'id', None)} source={getattr(p, 'source', '')} subject={getattr(p, 'subject', '')} "
            f"url={getattr(p, 'urlLink', '')} pdf={getattr(p, 'pdfLink', '')} "
            f"md={getattr(p, 'mdLink', '')} overview={getattr(p, 'overviewLink', '')} "
            f"analysis={getattr(p, 'analysisLink', '')}"
        )


if __name__ == "__main__":
    main()

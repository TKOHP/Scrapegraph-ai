"""
Google Scholar 订阅论文处理主入口

在此文件中设定固定主题并组装示例邮件内容，调用 `GoogleScholarPaperGraph` 执行完整流程。
"""

from typing import List

from scrapegraphai.MyGraphs import GoogleScholarPaperGraph

# 固定主题（外部订阅信息在代码外配置，此处仅记录）
FIXED_SUBJECT: str = "Financial Technology Survey"


def build_example_emails() -> List[str]:
    """
    构造示例邮件文本列表
    """
    return [
        "Dear researcher, here are new papers: https://scholar.google.com/scholar?cluster=123456 Some other info.",
        "New arxiv update: https://arxiv.org/abs/2401.01234 and a possible pdf link https://arxiv.org/pdf/2401.01234.pdf",
        "Additional resources: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1234567",
    ]


def main():
    """
    主函数，创建并运行流程图
    """
    emails = build_example_emails()
    graph_config = {
        "llm": {"model_instance": None, "model_tokens": 8192},
        "db_path": "data/google_scholar_papers.db",
        "download_dir": "data/papers",
        "verbose": True,
    }
    graph = GoogleScholarPaperGraph(
        prompt="Google Scholar Subscription Pipeline",
        emails=emails,
        subject=FIXED_SUBJECT,
        config=graph_config,
    )
    papers = graph.run()
    for p in papers:
        print(
            f"id={getattr(p, 'id', None)} subject={getattr(p, 'subject', '')} "
            f"url={getattr(p, 'urlLink', '')} pdf={getattr(p, 'pdfLink', '')} "
            f"md={getattr(p, 'mdLink', '')} summary={getattr(p, 'summaryLink', '')}"
        )


if __name__ == "__main__":
    main()


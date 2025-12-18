"""
document_scraper example
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 将项目根目录加入 Python 路径，避免在示例子目录运行时找不到包
# project_root = Path(__file__).resolve().parents[3]
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

from scrapegraphai.graphs import DocumentScraperGraph


def main():
    """运行 DocumentScraperGraph，对输入文本进行主题摘要并输出 JSON 结果"""
    load_dotenv()

    openai_key = os.getenv("OPENAI_APIKEY")

    graph_config = {
    "llm": {
        "api_key": "sk-cd7b54e0eaf5444ea29c71dc2cea3731",
        "model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
}

    source = """
        The Divine Comedy, Italian La Divina Commedia, original name La commedia, long narrative poem written in Italian
        circa 1308/21 by Dante. It is usually held to be one of the world s great works of literature.
        Divided into three major sections—Inferno, Purgatorio, and Paradiso—the narrative traces the journey of Dante
        from darkness and error to the revelation of the divine light, culminating in the Beatific Vision of God.
        Dante is guided by the Roman poet Virgil, who represents the epitome of human knowledge, from the dark wood
        through the descending circles of the pit of Hell (Inferno). He then climbs the mountain of Purgatory, guided
        by the Roman poet Statius, who represents the fulfilment of human knowledge, and is finally led by his lifelong love,
        the Beatrice of his earlier poetry, through the celestial spheres of Paradise.
    """

    pdf_scraper_graph = DocumentScraperGraph(
        prompt="Summarize the text and find the main topics",
        source=source,
        config=graph_config,
    )
    result = pdf_scraper_graph.run()

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()

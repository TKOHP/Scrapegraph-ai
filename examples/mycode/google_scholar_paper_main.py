"""
Google Scholar 订阅论文处理主入口

在此文件中设定固定主题并组装示例邮件内容，调用 `GoogleScholarPaperGraph` 执行完整流程。
"""

import os
from typing import List

from scrapegraphai.MyGraphs import GoogleScholarPaperGraph
from langchain_openai import ChatOpenAI

# 固定主题（外部订阅信息在代码外配置，此处仅记录）
FIXED_SUBJECT: str = "Financial Technology Survey"

SUBJECTS_POOL = [
    "金融科技",
    "大模型智能体",
    "区块链与加密资产",
    "机器学习安全",
    "数据隐私与合规",
]

def build_email_config() -> dict:
    """
    构造邮件抓取所需的配置字典（QQ 邮箱）
    
    返回字段：
    - imap_server: IMAP 服务地址（默认 imap.qq.com）
    - account: 邮箱账号
    - password: 邮箱授权码或密码
    - sender_email: 过滤的发件人（默认 Google Scholar 提醒）
    - days_recent: 近期天数过滤（当前节点未使用，预留）
    """
    # 从环境变量读取邮箱配置，提供默认值
    imap_server = os.getenv("QQ_IMAP_SERVER", "imap.qq.com")
    account = os.getenv("QQ_EMAIL", "1134952622@qq.com")
    password = os.getenv("QQ_PASSWORD", "zhbnmvewqjpljbjg")
    return {
        "imap_server": imap_server,
        "account": account,
        "password": password,
        "sender_email": "scholaralerts-noreply@google.com",
        "days_recent": 7,
    }


def build_simple_llm() -> object:
    """
    构造简易模型实例（用于分类等轻任务）
    """
    try:
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = os.getenv("OPENAI_API_KEY", "sk-cd7b54e0eaf5444ea29c71dc2cea3731")
        return ChatOpenAI(
            model="qwen-flash",
            base_url=base_url,
            api_key=api_key,
            temperature=0.2,
        )
    except Exception:
        return None


def build_complex_llm() -> object:
    """
    构造复杂模型实例（用于总结等重任务）
    """
    try:
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = os.getenv("OPENAI_API_KEY", "sk-cd7b54e0eaf5444ea29c71dc2cea3731")
        return ChatOpenAI(
            model="qwen3-max",
            base_url=base_url,
            api_key=api_key,
            temperature=0.2,
        )
    except Exception:
        return None


def main():
    """
    主函数，创建并运行流程图
    """
    email_config = build_email_config()
    simple_llm = build_simple_llm()
    complex_llm = build_complex_llm()
    graph_config = {
        "llm": {"model_instance": None, "model_tokens": 8192},
        "db_path": "data/google_scholar_papers.db",
        "download_dir": "data/papers",
        "verbose": True,
    }
    graph = GoogleScholarPaperGraph(
        prompt="Google Scholar Subscription Pipeline",
        email_config=email_config,
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
            f"md={getattr(p, 'mdLink', '')} summary={getattr(p, 'summaryLink', '')}"
        )


if __name__ == "__main__":
    main()

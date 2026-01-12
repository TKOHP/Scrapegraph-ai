"""
Google Scholar 最小流程主入口

在此文件中组装示例邮箱配置，调用 `GoogleScholarMinimalGraph` 执行仅“邮件链接提取 + PDF 获取”的流程。
"""

import os
from typing import List

try:
    from AIpaper.Graphs import GoogleScholarMinimalGraph
    from AIpaper.common_settings import build_graph_config
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from AIpaper.Graphs import GoogleScholarMinimalGraph
    from AIpaper.common_settings import build_graph_config
from scrapegraphai.utils import set_verbosity_info, set_formatting


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
        "days_recent": 1,
    }


def main():
    """
    主函数：创建并运行最小流程图（EmailLink + PdfFetch）
    """
    set_verbosity_info()
    set_formatting()
    email_config = build_email_config()
    graph_config = build_graph_config()
    graph = GoogleScholarMinimalGraph(
        prompt="Google Scholar Minimal Pipeline",
        email_config=email_config,
        config=graph_config,
    )
    papers = graph.run()
    for p in papers:
        print(
            f"id={getattr(p, 'id', None)} url={getattr(p, 'urlLink', '')} "
            f"pdf={getattr(p, 'pdfLink', '')}"
        )


if __name__ == "__main__":
    main()

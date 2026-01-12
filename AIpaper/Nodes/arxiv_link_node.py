"""
arXiv 链接提取节点

根据 `arxiv_config`（查询关键词、类别、数量等）调用 arXiv API，
提取论文网页链接，并创建/更新数据库记录。
"""

import time
from typing import List, Optional, Dict, Any
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
import requests

from scrapegraphai.utils import get_logger
from scrapegraphai.nodes.base_node import BaseNode
try:
    from .db_manager import DatabaseManager, AIPaper
except ImportError:
    from AIpaper.Nodes.db_manager import DatabaseManager, AIPaper


class ArxivLinkNode(BaseNode):
    """
    arXiv 链接提取节点
    
    输入 `arxiv_config`，通过 arXiv API 查询并写入 SQLite 数据库，
    在状态中返回 `papers` 列表。
    node_config 需包含：
    - db_path: 数据库文件路径
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "ArxivLinkExtract",
    ):
        """
        初始化节点
        
        Args:
            input: 输入键表达式，如 "arxiv_config"
            output: 输出键列表，建议为 ["papers"]
            node_config: 节点配置，支持：
                - db_path: 数据库文件路径
            node_name: 节点名称
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger()
        self.db_path = (self.node_config or {}).get("db_path", "AIpaper/data/google_scholar_papers.db")
        self.db = DatabaseManager(self.db_path)

    def _append_categories(self, base_query: str, categories: List[str]) -> str:
        """
        将类别限制附加到基础查询后（作为 OR 条件或 AND 条件，取决于需求）
        这里实现为：(base_query) OR (cat:A OR cat:B) 
        注：如果意图是在特定类别下搜索，应改为 AND。但根据原逻辑是扩大搜索范围。
        为了保持原逻辑：查询结果包含“满足关键词”或“属于指定类别”的论文。
        """
        if not categories:
            return base_query
        
        cats_clause = ""
        cat_parts = [f'cat:{c.strip()}' for c in categories if c and isinstance(c, str)]
        if cat_parts:
            cats_clause = " OR ".join(cat_parts)
            
        if not base_query.strip():
            return cats_clause
            
        if cats_clause:
            return f"({base_query}) AND ({cats_clause})"
        return base_query

    def _fetch_arxiv_entries(
        self, search_query: str, start: int = 0, max_results: int = 30, sort_by: str = "submittedDate", sort_order: str = "descending"
    ) -> List[Dict[str, Any]]:
        """
        调用 arXiv API 并解析返回的 Atom Feed，抽取基本字段
        
        返回字典字段：
        - url: 论文网页链接（abs 链接）
        - published: 发布时间字符串
        """
        base = "http://export.arxiv.org/api/query"
        params = {
            "search_query": search_query,
            "start": int(start),
            "max_results": int(max_results),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        url = f"{base}?{urlencode(params)}"
        self.logger.info(f"arXiv API 请求: {url}")
        out: List[Dict[str, Any]] = []
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("a:entry", ns):
                id_el = entry.find("a:id", ns)
                pub_el = entry.find("a:published", ns)
                url_abs = id_el.text if id_el is not None else None
                published = pub_el.text if pub_el is not None else None
                if url_abs:
                    out.append({"url": url_abs, "published": published})
        except Exception as e:
            self.logger.error(f"arXiv API 解析失败: {e}")
        return out

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 读取 `arxiv_config.querys_pool` (独立查询列表)
        - 对每一条查询（附加 categories 后）分别调用 arXiv API
        - 合并所有结果并去重
        - 写入数据库并返回
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        if "arxiv_config" not in input_keys or not isinstance(state.get("arxiv_config"), dict):
            raise ValueError("缺少 arxiv_config：请在图输入中提供 arXiv 查询配置信息")
        
        arxiv_config: dict = state["arxiv_config"]
        start = int((arxiv_config or {}).get("start", 0))
        max_results = int((arxiv_config or {}).get("max_results", 30))
        sort_by = (arxiv_config or {}).get("sort_by", "submittedDate")
        sort_order = (arxiv_config or {}).get("sort_order", "descending")
        
        # 获取查询池（字符串列表）
        querys_pool = arxiv_config.get("querys_pool") or []
        # 兼容旧配置：如果是单字符串 query，转为列表
        if not querys_pool and arxiv_config.get("query"):
            querys_pool = [arxiv_config.get("query")]
            
        categories = arxiv_config.get("categories") or []
        
        if not querys_pool and not categories:
            self.logger.warning("未配置任何查询词或类别，跳过 arXiv 检索")
            state.update({self.output[0]: []})
            return state

        # 如果 querys_pool 为空但有 categories，构造一个仅基于类别的查询
        if not querys_pool and categories:
            querys_pool = [""]

        all_entries: List[Dict[str, Any]] = []
        
        # 遍历每个独立查询
        for q_base in querys_pool:
            if not isinstance(q_base, str):
                self.logger.warning(f"跳过非字符串查询项: {q_base}")
                continue
                
            final_query = self._append_categories(q_base, categories)
            self.logger.info(f"执行独立查询: {final_query}")
            
            entries = self._fetch_arxiv_entries(
                search_query=final_query,
                start=start,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            all_entries.extend(entries)
            # 礼貌延时
            time.sleep(1)

        self.logger.info(f"所有查询共返回原始链接数: {len(all_entries)}")

        papers: List[AIPaper] = []
        inserted_count = 0
        existed_count = 0
        duplicated_count = 0
        unique_urls_set = set()
        
        for it in all_entries:
            url = (it.get("url") or "").strip()
            if not url:
                continue
            if url in unique_urls_set:
                duplicated_count += 1
                continue
            unique_urls_set.add(url)
            
            existing = self.db.find_by_url(url)
            if existing:
                existed_count += 1
                papers.append(existing)
                continue
            paper = AIPaper(
                id=None,
                urlLink=url,
                source="arxiv",
                pdfLink=None,
                mdLink=None,
                overviewLink=None,
                analysisLink=None,
                meta=None,
                publishTime=None,
                subject=None,
                receivedTime=it.get("published"),
                type=None,
            )
            try:
                new_id = self.db.insert_paper(paper)
                paper.id = new_id
                inserted_count += 1
                papers.append(paper)
            except Exception as e:
                self.logger.error(f"插入 arXiv 记录失败: {e}")

        self.logger.info(f"arXiv 链接处理完成 inserted={inserted_count} existed={existed_count} duplicated={duplicated_count} papers_out={len(papers)}")
        state.update({self.output[0]: papers})
        return state

# =========================
# 测试入口（单文件运行）
# =========================
def run_test_arxiv_api(arxiv_config: Dict[str, Any], db_path: str = "AIpaper/data/google_scholar_papers.db") -> None:
    """
    测试 arXiv API 调用逻辑，输出调试信息
    
    Args:
        arxiv_config: 测试用的 arXiv 配置字典
        db_path: 数据库路径
    """
    from scrapegraphai.utils import set_verbosity_info
    set_verbosity_info()  # 开启详细日志
    
    print(f"\n{'='*20} 开始 arXiv API 测试 {'='*20}")
    print(f"数据库路径: {db_path}")
    
    node = ArxivLinkNode(
        input="arxiv_config",
        output=["papers"],
        node_config={"db_path": db_path}
    )
    
    # 打印将要执行的查询列表
    querys_pool = arxiv_config.get("querys_pool") or []
    categories = arxiv_config.get("categories") or []
    print(f"\n[调试] 待执行的独立查询 ({len(querys_pool)} 条):")
    for q in querys_pool:
        final = node._append_categories(q, categories)
        print(f"  - {final}")
    
    # 2. 执行节点逻辑（包含 API 调用）
    state = {"arxiv_config": arxiv_config}
    try:
        start_time = time.time()
        result_state = node.execute(state)
        duration = time.time() - start_time
        
        papers = result_state.get("papers", [])
        print(f"\n[结果] API 调用耗时: {duration:.2f}秒")
        print(f"[结果] 获取论文数量: {len(papers)}")
        
        if papers:
            print("\n[样例数据] 前 3 条论文:")
            for i, p in enumerate(papers[:3]):
                print(f"  {i+1}. URL: {p.urlLink}")
                print(f"     发布时间: {p.receivedTime}")
                print(f"     数据库ID: {p.id}")
        else:
            print("\n[警告] 未获取到任何论文，请检查查询条件或网络连接。")
            
    except Exception as e:
        print(f"\n[错误] 测试执行失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    主测试入口，使用 common_settings 中的真实配置进行测试
    """
    try:
        from AIpaper.common_settings import build_arxiv_config
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from AIpaper.common_settings import build_arxiv_config

    # 获取公共配置
    config = build_arxiv_config()
    
    # 运行测试
    run_test_arxiv_api(config)

if __name__ == "__main__":
    main()

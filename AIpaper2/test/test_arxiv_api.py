"""
arXiv API 测试脚本（不依赖节点，不写入数据库）

自定义查询内容，直接调用 arXiv API，打印查询与结果的详细信息，并进行去重统计。
"""

from typing import Dict, Any, List
import arxiv
import time


def build_arxiv_config() -> Dict[str, Any]:
    """
    构建用于测试的 arXiv 查询配置
    
    返回：
    - arxiv_config 字典，包含 querys_pool/categories/start/max_results/sort_by/sort_order
    """
    return {
        "querys_pool": [
            '(ti:LLM survey)',
        ],
        "categories": [],
        "start": 0,
        "max_results": 30,
        "sort_by": "submittedDate",
        "sort_order": "descending",
    }

def append_categories(base_query: str, categories: List[str]) -> str:
    """
    将类别限制附加到基础查询后
    
    参数：
    - base_query: 基础关键词查询字符串
    - categories: 类别列表（如 ["cs.AI", "cs.CL"]）
    
    返回：
    - 合成后的查询字符串
    """
    if not categories:
        return base_query
    cat_parts = [f"cat:{c.strip()}" for c in categories if c and isinstance(c, str)]
    cats_clause = " OR ".join(cat_parts) if cat_parts else ""
    if not base_query.strip():
        return cats_clause
    if cats_clause:
        return f"({base_query}) AND ({cats_clause})"
    return base_query

def search_arxiv_entries(
    search_query: str,
    start: int = 0,
    max_results: int = 30,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> List[Dict[str, Any]]:
    """
    使用 python-arxiv 包执行查询并返回条目列表
    
    参数：
    - search_query: 查询字符串（直接传给 arXiv API）
    - start: 起始偏移（通过增加抓取数量后切片实现）
    - max_results: 返回条数
    - sort_by: 排序字段（submittedDate 或 lastUpdatedDate）
    - sort_order: 排序方向（ascending 或 descending）
    
    返回：
    - 条目列表，每项包含 url（abs 链接）、published（ISO 字符串）、title
    """
    # 兼容不同版本的 arxiv 包枚举命名（SortCriterion / Sorting）
    sort_cls = getattr(arxiv, "SortCriterion", None) or getattr(arxiv, "Sorting", None)
    order_cls = getattr(arxiv, "SortOrder", None)
    sort_map = {}
    order_map = {}
    if sort_cls is not None:
        sort_map = {
            "submittedDate": getattr(sort_cls, "SubmittedDate", None),
            "lastUpdatedDate": getattr(sort_cls, "LastUpdatedDate", None),
            "relevance": getattr(sort_cls, "Relevance", None),
        }
    if order_cls is not None:
        order_map = {
            "ascending": getattr(order_cls, "Ascending", None),
            "descending": getattr(order_cls, "Descending", None),
        }
    sort_by_enum = sort_map.get(sort_by, sort_map.get("submittedDate"))
    sort_order_enum = order_map.get(sort_order, order_map.get("descending"))
    
    fetch_count = int(max_results) + int(start)
    search_kwargs = {
        "query": search_query,
        "max_results": fetch_count,
    }
    if sort_by_enum is not None:
        search_kwargs["sort_by"] = sort_by_enum
    if sort_order_enum is not None:
        search_kwargs["sort_order"] = sort_order_enum
    search = arxiv.Search(**search_kwargs)
    # 限速设置：减小页面大小并增加延迟，降低 429 风险
    client = arxiv.Client(page_size=max(5, min(25, fetch_count)), delay_seconds=2, num_retries=2)
    out: List[Dict[str, Any]] = []
    try:
        for res in client.results(search):
            url = res.entry_id or ""
            title = (res.title or "").strip()
            published = ""
            try:
                if getattr(res, "published", None):
                    published = res.published.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                published = ""
            if url:
                out.append({"url": url, "published": published, "title": title})
    except Exception as e:
        msg = str(e)
        print(f"[错误] 使用 arxiv 包查询失败: {msg}")
        # 若命中 429，进行一次退避重试：更小页面，更长延迟
        if "429" in msg:
            try:
                wait = 3
                print(f"[提示] 触发 429，等待 {wait}s 后重试（减小 page_size 增加延迟）")
                time.sleep(wait)
                client = arxiv.Client(page_size=max(5, min(10, fetch_count)), delay_seconds=5, num_retries=3)
                for res in client.results(search):
                    url = res.entry_id or ""
                    title = (res.title or "").strip()
                    published = ""
                    try:
                        if getattr(res, "published", None):
                            published = res.published.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        published = ""
                    if url:
                        out.append({"url": url, "published": published, "title": title})
            except Exception as e2:
                print(f"[错误] 二次重试仍失败: {e2}")
    # 应用起始偏移
    s = int(start)
    m = int(max_results)
    return out[s : s + m]


def _print_entries(entries: List[Dict[str, Any]]) -> None:
    """
    打印 arXiv 条目列表的详细信息
    
    参数：
    - entries: 由 _fetch_arxiv_entries 返回的条目列表，每项至少包含 url/published
    """
    for i, it in enumerate(entries, start=1):
        url = it.get("url") or ""
        pub = it.get("published") or ""
        title = it.get("title") or ""
        print(f"  {i}. url={url}")
        print(f"     published={pub}")
        print(f"     title={title}")


def _dedup_stats(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    计算去重统计信息
    
    参数：
    - entries: 原始条目列表
    
    返回：
    - 包含 total、unique、duplicated 三个键的统计字典
    """
    total = len(entries)
    urls = [(it.get("url") or "").strip() for it in entries if (it.get("url") or "").strip()]
    unique = len(set(urls))
    duplicated = total - unique
    return {"total": total, "unique": unique, "duplicated": duplicated}


def run_arxiv_api_test(arxiv_config: Dict[str, Any], db_path: str = "AIpaper/data/arxiv_api_test.db") -> None:
    """
    运行 arXiv API 测试：打印查询与结果详情，不写入数据库
    
    参数：
    - arxiv_config: 查询配置字典
    - db_path: 保留参数以兼容，但不使用
    """
    print("=" * 16, "arXiv API 测试开始", "=" * 16)
    print("querys_pool=", arxiv_config.get("querys_pool") or [])
    print("categories=", arxiv_config.get("categories") or [])
    print(
        f"start={arxiv_config.get('start', 0)} "
        f"max_results={arxiv_config.get('max_results', 30)} "
        f"sort_by={arxiv_config.get('sort_by', 'submittedDate')} "
        f"sort_order={arxiv_config.get('sort_order', 'descending')}"
    )

    all_entries: List[Dict[str, Any]] = []
    querys_pool: List[str] = arxiv_config.get("querys_pool") or []
    categories: List[str] = arxiv_config.get("categories") or []

    # 若仅指定类别，可补一个空查询以执行
    if not querys_pool and categories:
        querys_pool = [""]

    # 逐条查询并打印
    for base in querys_pool:
        final_query = append_categories(base, categories)
        print(f"\n[查询] {final_query}")
        entries = search_arxiv_entries(
            final_query,
            start=int(arxiv_config.get("start", 0)),
            max_results=int(arxiv_config.get("max_results", 30)),
            sort_by=arxiv_config.get("sort_by", "submittedDate"),
            sort_order=arxiv_config.get("sort_order", "descending"),
        )
        if not entries:
            print("  (无返回条目)")
        else:
            _print_entries(entries)
        all_entries.extend(entries)

    # 汇总统计并打印
    stats = _dedup_stats(all_entries)
    print("\n[统计] 原始条目数:", stats["total"])
    print("[统计] 去重后链接数:", stats["unique"])
    print("[统计] 重复链接数:", stats["duplicated"])
    if all_entries:
        print("\n[样例] 前 3 条条目：")
        for i, it in enumerate(all_entries[:3], start=1):
            print(f"  {i}. url={it.get('url')}")
            print(f"     published={it.get('published')}")
            print(f"     title={it.get('title')}")


def main() -> None:
    """
    主入口：构建示例配置并运行测试
    """
    cfg = build_arxiv_config()
    run_arxiv_api_test(cfg)


if __name__ == "__main__":
    main()

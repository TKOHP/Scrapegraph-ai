"""
Semantic Scholar API 测试脚本（不依赖节点，不写入数据库）

自定义查询内容，调用 Semantic Scholar Graph API，打印查询与结果的详细信息，并进行去重统计。
"""

import os
import time
from typing import Dict, Any, List
import requests


def build_semantic_config() -> Dict[str, Any]:
    """
    构建用于测试的 Semantic Scholar 查询配置
    
    返回：
    - semantic_config 字典，包含 querys_pool/offset/limit/fields/sort_by_year_desc/api_key
    """
    return {
        "querys_pool": [
            "LLM survey",
        ],
        "offset": 0,
        "limit": 100,
        "fields": [
            "title",
            "year",
            "venue",
            "url",
            "abstract",
            "authors",
            "externalIds",
            "isOpenAccess",
            "openAccessPdf",
            "citationCount",
        ],
        "sort_by_year_desc": True,
        "api_key": os.environ.get("SEMANTIC_SCHOLAR_API_KEY", None),
    }


def _format_authors(authors: Any) -> str:
    """
    将作者列表格式化为字符串
    
    参数：
    - authors: 作者对象列表或其他结构
    
    返回：
    - 格式化后的作者字符串
    """
    if not isinstance(authors, list):
        return ""
    names = []
    for a in authors:
        if isinstance(a, dict):
            name = str(a.get("name") or "").strip()
            if name:
                names.append(name)
        else:
            s = str(a).strip()
            if s:
                names.append(s)
    return "、".join(names)


def search_semantic_entries(
    query: str,
    offset: int = 0,
    limit: int = 30,
    fields: List[str] = None,
    api_key: str = None,
) -> List[Dict[str, Any]]:
    """
    使用 Semantic Scholar Graph API 执行查询并返回条目列表
    
    参数：
    - query: 查询字符串
    - offset: 偏移量
    - limit: 返回数量
    - fields: 需要返回的字段列表
    - api_key: 可选的 API Key（若提供将提升稳定性与限额）
    
    返回：
    - 条目列表，每项包含 id/title/year/venue/url/abstract/authors/doi/isOpenAccess/pdfUrl/citationCount
    """
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "offset": int(offset),
        "limit": int(limit),
        "fields": ",".join(fields or []),
    }
    headers = {
        "User-Agent": "Scrapegraph-ai-SemanticTest/1.0",
    }
    if api_key:
        headers["x-api-key"] = api_key
    out: List[Dict[str, Any]] = []
    attempts = 2
    for i in range(attempts):
        try:
            resp = requests.get(base, params=params, headers=headers, timeout=20)
            status = resp.status_code
            if status in (429, 503):
                wait = 2 * (i + 1)
                print(f"[提示] 服务限制或繁忙，等待 {wait}s 后重试 ({i+1}/{attempts})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data") or []
            for it in items:
                pid = it.get("paperId") or ""
                title = (it.get("title") or "").strip()
                year = it.get("year")
                venue = (it.get("venue") or "").strip()
                url = (it.get("url") or "").strip()
                abstract = (it.get("abstract") or "").strip()
                authors = _format_authors(it.get("authors"))
                ext_ids = it.get("externalIds") or {}
                doi = (ext_ids.get("DOI") or "").strip() if isinstance(ext_ids, dict) else ""
                is_oa = bool(it.get("isOpenAccess"))
                pdf_obj = it.get("openAccessPdf") or {}
                pdf_url = (pdf_obj.get("url") or "").strip() if isinstance(pdf_obj, dict) else ""
                citation_count = it.get("citationCount")
                out.append({
                    "id": pid,
                    "title": title,
                    "year": year,
                    "venue": venue,
                    "url": url,
                    "abstract": abstract,
                    "authors": authors,
                    "doi": doi,
                    "isOpenAccess": is_oa,
                    "pdfUrl": pdf_url,
                    "citationCount": citation_count,
                })
            break
        except Exception as e:
            print(f"[错误] Semantic Scholar 查询失败: {e}")
            break
    return out


def _print_entries(entries: List[Dict[str, Any]]) -> None:
    """
    打印 Semantic Scholar 条目列表的详细信息
    
    参数：
    - entries: 由 search_semantic_entries 返回的条目列表
    """
    for i, it in enumerate(entries, start=1):
        print(f"  {i}. title={it.get('title') or ''}")
        print(f"     year={it.get('year')}")
        print(f"     venue={it.get('venue') or ''}")
        print(f"     url={it.get('url') or ''}")
        print(f"     doi={it.get('doi') or ''}")
        print(f"     authors={it.get('authors') or ''}")
        print(f"     isOpenAccess={it.get('isOpenAccess')}")
        print(f"     pdfUrl={it.get('pdfUrl') or ''}")
        print(f"     citationCount={it.get('citationCount')}")


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


def sort_entries_by_year_desc(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按年份倒序排序条目列表（最新在前）
    
    参数：
    - entries: 条目列表
    
    返回：
    - 按年份降序排列的条目列表
    """
    return sorted(entries, key=lambda it: (it.get("year") or 0), reverse=True)


def run_semantic_api_test(semantic_config: Dict[str, Any]) -> None:
    """
    运行 Semantic Scholar API 测试：打印查询与结果详情，不写入数据库
    
    参数：
    - semantic_config: 查询配置字典
    """
    print("=" * 16, "Semantic Scholar API 测试开始", "=" * 16)
    print("querys_pool=", semantic_config.get("querys_pool") or [])
    print(
        f"offset={semantic_config.get('offset', 0)} "
        f"limit={semantic_config.get('limit', 30)} "
        f"fields={semantic_config.get('fields') or []} "
        f"api_key={'***' if semantic_config.get('api_key') else '(未提供)'}"
    )
    all_entries: List[Dict[str, Any]] = []
    for base in semantic_config.get("querys_pool") or []:
        print(f"\n[查询] {base}")
        entries = search_semantic_entries(
            base,
            offset=int(semantic_config.get("offset", 0)),
            limit=int(semantic_config.get("limit", 30)),
            fields=semantic_config.get("fields") or [],
            api_key=semantic_config.get("api_key"),
        )
        if entries and semantic_config.get("sort_by_year_desc", False):
            entries = sort_entries_by_year_desc(entries)
        if not entries:
            print("  (无返回条目)")
        else:
            _print_entries(entries)
        time.sleep(1)
        all_entries.extend(entries)
    stats = _dedup_stats(all_entries)
    print("\n[统计] 原始条目数:", stats["total"])
    print("[统计] 去重后链接数:", stats["unique"])
    print("[统计] 重复链接数:", stats["duplicated"])
    if all_entries:
        print("\n[样例] 前 3 条条目：")
        preview = sort_entries_by_year_desc(all_entries) if semantic_config.get("sort_by_year_desc", False) else all_entries
        for i, it in enumerate(preview[:3], start=1):
            print(f"  {i}. title={it.get('title')}")
            print(f"     year={it.get('year')}")
            print(f"     venue={it.get('venue')}")
            print(f"     url={it.get('url')}")
            print(f"     doi={it.get('doi')}")


def main() -> None:
    """
    主入口：构建示例配置并运行测试
    """
    cfg = build_semantic_config()
    run_semantic_api_test(cfg)


if __name__ == "__main__":
    main()

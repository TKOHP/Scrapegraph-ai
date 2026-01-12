"""
公共设置与构造函数

提供主题池、LLM 构造与图配置构造，供多入口脚本复用。
"""

import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI


# 固定主题池（可在项目中统一维护）
SUBJECTS_POOL: List[str] = [
    "金融科技综述",
    "大模型综述",
    "金融科技大模型综述",
    "移动端自动化大模型Agent",
    "其他",
]
# arXiv 查询池：每一项为一条独立的检索语句（支持 arXiv API 语法）
ARXIV_QUERYS_POOL: List[str] = [
    # 1. Fintech 综述
    '(ti:fintech OR abs:fintech OR ti:"financial technology" OR abs:"financial technology") AND (ti:review OR abs:review OR ti:survey OR abs:survey OR ti:tutorial OR abs:tutorial)',
    # 2. LLM 综述
    '(ti:LLM OR abs:LLM OR ti:"Large Language Model" OR abs:"Large Language Model") AND (ti:review OR abs:review OR ti:survey OR abs:survey OR ti:tutorial OR abs:tutorial)',
    # 3. Fintech + LLM + 综述
    '(ti:fintech OR abs:fintech OR ti:"financial technology" OR abs:"financial technology") AND (ti:LLM OR abs:LLM OR ti:"Large Language Model" OR abs:"Large Language Model") AND (ti:review OR abs:review OR ti:survey OR abs:survey OR ti:tutorial OR abs:tutorial)',
    # 4. Mobile Automation + LLM + Agent
    '(ti:"mobile automation" OR abs:"mobile automation") AND (ti:LLM OR abs:LLM OR ti:MLLM OR abs:MLLM) AND (ti:Agent* OR abs:Agent* OR ti:"Multi-Agent*" OR abs:"Multi-Agent*")',
]
# 论文类型池（仅允许选择其一）
TYPE_POOL: List[str] = [
    "综述型论文",
    "研究型论文",
    "其他",
]


def build_simple_llm() -> object:
    """
    构造简易模型实例（用于分类等轻任务）
    """
    try:
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = os.getenv("OPENAI_API_KEY", "sk-cd7b54e0eaf5444ea29c71dc2cea3731")
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY 环境变量")
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
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY 环境变量")
        return ChatOpenAI(
            model="qwen-max",
            base_url=base_url,
            api_key=api_key,
            temperature=0.2,
        )
    except Exception:
        return None


def build_graph_config() -> Dict[str, Any]:
    """
    构造图配置字典
    """
    return {
        "llm": {"model_instance": None, "model_tokens": 8192},
        "db_path": "AIpaper/data/google_scholar_papers.db",
        "download_dir": "AIpaper/data/papers",
        "verbose": True,
        "rebuild_md": False,
        "rebuild_overview": False,
        "rebuild_analysis": False,
        "rebuild_classify": False,
    }


def build_arxiv_config() -> Dict[str, Any]:
    """
    构造 arXiv 查询配置字典
    
    返回字段：
    - querys_pool: 查询语句列表（每项为一条独立检索式）
    - categories: 类别列表（如 ["cs.AI"]），将作为可选的 OR 条件附加到每个查询后
    - start: 起始偏移
    - max_results: 每个查询的最大返回数量
    """
    return {
        "querys_pool": ARXIV_QUERYS_POOL,
        "categories": ["cs.AI", "cs.CL"],
        "start": 0,
        "max_results": 2,
        "sort_by": "submittedDate",
        "sort_order": "descending",
    }


"""
SQLite 数据库管理模块

该模块提供对 `AIpaper` 表的增删改查封装，供流程节点与图使用。
"""

import os
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

from ..utils import get_logger


@dataclass
class AIPaper:
    """AIpaper 数据实体"""
    id: Optional[int]
    urlLink: str
    pdfLink: Optional[str]
    mdLink: Optional[str]
    summaryLink: Optional[str]
    meta: Optional[str]
    publishTime: Optional[str]
    subject: Optional[str]


class DatabaseManager:
    """
    SQLite 数据库管理器
    
    提供初始化表结构，以及对 `AIpaper` 的增删改查接口。
    """

    def __init__(self, db_path: str):
        """
        初始化数据库管理器并确保表结构存在
        
        Args:
            db_path: 数据库文件路径
        """
        self.logger = get_logger(__name__)
        self.db_path = db_path
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        """
        确保数据库所在目录存在
        """
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        except Exception as e:
            self.logger.error(f"创建数据库目录失败: {e}")
            raise

    def _get_conn(self) -> sqlite3.Connection:
        """
        获取 SQLite 连接
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            self.logger.error(f"打开数据库失败: {e}")
            raise

    def _init_db(self):
        """
        初始化 `AIpaper` 表结构
        """
        sql = """
        CREATE TABLE IF NOT EXISTS AIpaper (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            urlLink TEXT NOT NULL,
            pdfLink TEXT,
            mdLink TEXT,
            summaryLink TEXT,
            meta TEXT,
            publishTime TEXT,
            subject TEXT
        );
        """
        try:
            with self._get_conn() as conn:
                conn.execute(sql)
                conn.commit()
            self.logger.info("数据库初始化完成: AIpaper 表就绪")
        except Exception as e:
            self.logger.error(f"初始化数据库失败: {e}")
            raise

    def insert_paper(self, paper: AIPaper) -> int:
        """
        插入一条 `AIpaper` 记录
        
        Args:
            paper: AIPaper 实例，id 可为 None
        
        Returns:
            新记录的 id
        """
        try:
            with self._get_conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO AIpaper (urlLink, pdfLink, mdLink, summaryLink, meta, publishTime, subject)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper.urlLink,
                        paper.pdfLink,
                        paper.mdLink,
                        paper.summaryLink,
                        paper.meta,
                        paper.publishTime,
                        paper.subject,
                    ),
                )
                conn.commit()
                new_id = cur.lastrowid
                self.logger.info(f"插入论文记录成功 id={new_id} url={paper.urlLink}")
                return new_id
        except Exception as e:
            self.logger.error(f"插入论文记录失败: {e}")
            raise

    def upsert_by_url(self, url: str, updates: Dict[str, Any]) -> int:
        """
        按 `urlLink` 进行插入或更新
        
        Args:
            url: 论文网页地址
            updates: 需要更新的字段字典
        
        Returns:
            受影响记录 id（新插入或已存在的记录 id）
        """
        try:
            with self._get_conn() as conn:
                cur = conn.execute("SELECT id FROM AIpaper WHERE urlLink = ?", (url,))
                row = cur.fetchone()
                if row:
                    set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                    params = list(updates.values()) + [url]
                    conn.execute(f"UPDATE AIpaper SET {set_clause} WHERE urlLink = ?", params)
                    conn.commit()
                    self.logger.info(f"更新论文记录成功 id={row['id']} url={url}")
                    return int(row["id"])
                else:
                    paper = AIPaper(
                        id=None,
                        urlLink=url,
                        pdfLink=updates.get("pdfLink"),
                        mdLink=updates.get("mdLink"),
                        summaryLink=updates.get("summaryLink"),
                        meta=updates.get("meta"),
                        publishTime=updates.get("publishTime"),
                        subject=updates.get("subject"),
                    )
                    return self.insert_paper(paper)
        except Exception as e:
            self.logger.error(f"Upsert 失败: {e}")
            raise

    def update_fields(self, paper_id: int, updates: Dict[str, Any]) -> None:
        """
        按 id 更新指定字段
        
        Args:
            paper_id: 论文记录 id
            updates: 字段更新字典
        """
        try:
            with self._get_conn() as conn:
                set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                params = list(updates.values()) + [paper_id]
                conn.execute(f"UPDATE AIpaper SET {set_clause} WHERE id = ?", params)
                conn.commit()
            self.logger.info(f"更新字段成功 id={paper_id} fields={list(updates.keys())}")
        except Exception as e:
            self.logger.error(f"更新字段失败: {e}")
            raise

    def delete_paper(self, paper_id: int) -> None:
        """
        删除指定 id 的论文记录
        """
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM AIpaper WHERE id = ?", (paper_id,))
                conn.commit()
            self.logger.info(f"删除论文记录成功 id={paper_id}")
        except Exception as e:
            self.logger.error(f"删除论文记录失败: {e}")
            raise

    def get_paper_by_id(self, paper_id: int) -> Optional[AIPaper]:
        """
        按 id 查询论文记录
        """
        try:
            with self._get_conn() as conn:
                cur = conn.execute("SELECT * FROM AIpaper WHERE id = ?", (paper_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return AIPaper(
                    id=row["id"],
                    urlLink=row["urlLink"],
                    pdfLink=row["pdfLink"],
                    mdLink=row["mdLink"],
                    summaryLink=row["summaryLink"],
                    meta=row["meta"],
                    publishTime=row["publishTime"],
                    subject=row["subject"],
                )
        except Exception as e:
            self.logger.error(f"查询论文失败: {e}")
            raise

    def list_papers(self, subject: Optional[str] = None) -> List[AIPaper]:
        """
        列出论文记录，可按主题过滤
        """
        try:
            with self._get_conn() as conn:
                if subject:
                    cur = conn.execute(
                        "SELECT * FROM AIpaper WHERE subject = ? ORDER BY id DESC",
                        (subject,),
                    )
                else:
                    cur = conn.execute("SELECT * FROM AIpaper ORDER BY id DESC")
                rows = cur.fetchall()
                return [
                    AIPaper(
                        id=r["id"],
                        urlLink=r["urlLink"],
                        pdfLink=r["pdfLink"],
                        mdLink=r["mdLink"],
                        summaryLink=r["summaryLink"],
                        meta=r["meta"],
                        publishTime=r["publishTime"],
                        subject=r["subject"],
                    )
                    for r in rows
                ]
        except Exception as e:
            self.logger.error(f"列出论文失败: {e}")
            raise

    def find_by_url(self, url: str) -> Optional[AIPaper]:
        """
        通过 urlLink 查找论文记录
        """
        try:
            with self._get_conn() as conn:
                cur = conn.execute("SELECT * FROM AIpaper WHERE urlLink = ?", (url,))
                r = cur.fetchone()
                if not r:
                    return None
                return AIPaper(
                    id=r["id"],
                    urlLink=r["urlLink"],
                    pdfLink=r["pdfLink"],
                    mdLink=r["mdLink"],
                    summaryLink=r["summaryLink"],
                    meta=r["meta"],
                    publishTime=r["publishTime"],
                    subject=r["subject"],
                )
        except Exception as e:
            self.logger.error(f"按 URL 查询失败: {e}")
            raise


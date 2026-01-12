"""
邮件链接提取节点

输入 `email_config`（邮箱账号密码等配置信息），
提取论文网页链接，并创建/更新数据库记录。
"""

import os
import re
import imaplib
import email
from email import policy
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
import time
import quopri
import base64
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse, parse_qs, unquote
import json

from scrapegraphai.utils import get_logger
from scrapegraphai.nodes.base_node import BaseNode
try:
    from .db_manager import DatabaseManager, AIPaper
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from AIpaper2.Nodes.db_manager import DatabaseManager, AIPaper
try:
    from AIpaper2.common_settings import SUBJECT_KEYWORDS_MAP
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from AIpaper2.common_settings import SUBJECT_KEYWORDS_MAP
except Exception:
    SUBJECT_KEYWORDS_MAP = {}


def _cn_timezone() -> timezone:
    """
    返回北京时间（UTC+8）的时区对象
    """
    return timezone(timedelta(hours=8))


def _now_in_cn() -> datetime:
    """
    返回北京时间（UTC+8）的当前时间（有时区）
    """
    return datetime.now(timezone.utc).astimezone(_cn_timezone())


def _to_cn_time(dt: Optional[datetime]) -> Optional[datetime]:
    """
    将任意 datetime 统一转换为北京时间（UTC+8）的有时区时间
    """
    if dt is None:
        return None
    try:
        if getattr(dt, "tzinfo", None) is None:
            return dt.replace(tzinfo=_cn_timezone())
        return dt.astimezone(_cn_timezone())
    except Exception:
        return dt


class EmailLinkNode(BaseNode):
    """
    邮件链接提取节点
    
    从邮件文本中提取论文网页链接，写入 SQLite 数据库，并在状态中返回 `papers` 列表。
    node_config 需包含：
    - db_path: 数据库文件路径
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "EmailLinkExtract",
    ):
        """
        初始化节点
        
        Args:
            input: 输入键表达式，如 "email_config"
            output: 输出键列表，建议为 ["papers"]
            node_config: 节点配置，支持：
                - db_path: 数据库文件路径
                - use_qq_email: 是否通过 QQ 邮箱抓取邮件
            node_name: 节点名称
        """
        super().__init__(node_name, "node", input, output, node_config=node_config)
        self.logger = get_logger()
        self.db_path = (self.node_config or {}).get("db_path")
        self.db = DatabaseManager(self.db_path)
        self.use_qq_email = bool((self.node_config or {}).get("use_qq_email", True))
        self.save_email_html = bool((self.node_config or {}).get("save_email_html", False))

    def _decode_google_redirect(self, href: str) -> str:
        """
        解码 Google 跳转链接，返回最终 URL
        """
        try:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            target = qs.get("url", [""])[0] or qs.get("q", [""])[0]
            if target:
                return unquote(target)
        except Exception:
            return href
        return href

    def _parse_div_authors_publisher(self, text: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        解析第二个 div 块文本以提取作者、出版商与时间信息
        
        参数：
        - text: 该 div 的完整文本，格式通常为：作者信息 - 出版商 , 时间信息
        
        返回：
        - (authors, publisher, publish_time) 三元组，若无法判断则返回 None
        """
        try:
            if not text:
                return None, None, None
            s = str(text).strip()
            authors: Optional[str] = None
            publisher: Optional[str] = None
            publish_time: Optional[str] = None

            # 第一步：查询是否存在 “,” 后跟数字的年份信息
            year_match = re.search(r",\s*(\d{4})\b", s)
            if year_match:
                publish_time = (year_match.group(1) or "").strip() or None
                s = s[:year_match.start()].strip()

            # 第二步：针对剩余文本，查询是否存在 “-” 后跟随的文本作为出版商
            dash_match = re.search(r"^(.*?)\s*[-–—]\s*(.*)$", s)
            if dash_match:
                authors = (dash_match.group(1) or "").strip() or None
                publisher = (dash_match.group(2) or "").strip() or None
            else:
                authors = s or None
                publisher = None

            return authors, publisher, publish_time
        except Exception:
            return None, None, None

    def _parse_alert_items(self, html: str) -> List[Dict[str, Any]]:
        """
        解析 Google Scholar 订阅邮件中的文献条目，提取结构化信息
        
        返回的每个条目包含字段：
        - url: 论文网页链接
        - title: 论文标题
        - authors: 作者信息（可能为空）
        - publication_info: 发表信息（期刊/会议等，可能为空）
        - abstract_snippet: 摘要片段（可能为空）
        """
        items: List[Dict[str, Any]] = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # 规则一：根据用户给出的结构，h3 内的 a 提供 url/title，h3 的下一个 div 是作者/元信息，下下个 div 是摘要
            h3s = soup.find_all("h3")
            for h in h3s:
                a = h.find("a", href=True)
                if not a:
                    continue
                href = a.get("href", "")
                final_url = self._decode_google_redirect(href)
                if not final_url or not final_url.startswith("http"):
                    continue
                title = a.get_text(strip=True) if a else None

                div1 = self._get_next_div_sibling(h, 1)
                div2 = self._get_next_div_sibling(h, 2)
                authors_text = None
                publication_info = None
                abstract_snippet = None

                # 作者与出版商解析：作者取第一段；出版商优先域名映射，其次文本末段或关键词
                div1_text = None
                publish_time = None
                if div1:
                    div1_text = div1.get_text(" ", strip=True)
                    authors_text, publication_info, publish_time = self._parse_div_authors_publisher(div1_text)

                # 摘要片段
                if div2:
                    abstract_snippet = div2.get_text(" ", strip=True)

                # 不再使用近邻 div 或域名映射来推断出版商，严格依赖第二个 div 块的文本格式

                items.append({
                    "url": final_url,
                    "title": title,
                    "authors": authors_text,
                    "publication": publication_info,
                    "abstract": abstract_snippet,
                    "publishTime": publish_time,
                })

        except Exception:
            pass
        dedup: List[Dict[str, Any]] = []
        seen = set()
        for it in items:
            u = it.get("url")
            if u and u not in seen:
                dedup.append(it)
                seen.add(u)
        return dedup

    def _infer_subscribe_from(self, html: str) -> Optional[str]:
        """
        根据邮件中的完整搜索链接关键词，结合 SUBJECT_KEYWORDS_MAP 推断订阅来源主题
        
        解析所有包含查询参数的 Scholar 链接，选取最长的查询字符串进行关键词匹配；
        命中数量最多的主题即为 subscribe_from。
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            queries: List[str] = []
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if ("scholar.google" in href) and ("?" in href) and ("q=" in href or "as_q" in href):
                    parsed = urlparse(href)
                    qs = parse_qs(parsed.query)
                    q = " ".join(qs.get("q", [])) or " ".join(qs.get("as_q", []))
                    if q:
                        queries.append(unquote(q))
            q_text = max(queries, key=len) if queries else ""
            q_lower = q_text.lower()
            best_subject = None
            best_score = 0
            for subject, keywords in (SUBJECT_KEYWORDS_MAP or {}).items():
                score = sum(1 for kw in (keywords or []) if kw and kw.lower() in q_lower)
                if score > best_score and score > 0:
                    best_subject = subject
                    best_score = score
            return best_subject
        except Exception:
            return None

    def _decode_part_content(self, part) -> str:
        """
        解码邮件内容片段（参考 debug/agent_crewai.py 的处理方式）
        """
        try:
            charset = part.get_content_charset() or "utf-8"
            payload = part.get_payload(decode=True) or b""
            cte = (part.get("Content-Transfer-Encoding") or "").lower().strip()
            if cte == "quoted-printable":
                try:
                    return quopri.decodestring(payload).decode(charset, errors="ignore")
                except Exception:
                    return payload.decode(charset, errors="ignore")
            if cte == "base64":
                try:
                    return base64.b64decode(payload, validate=True).decode(charset, errors="ignore")
                except Exception:
                    return payload.decode(charset, errors="ignore")
            return payload.decode(charset, errors="ignore")
        except Exception:
            return ""

    def _format_datetime_str(self, dt: Optional[datetime]) -> Optional[str]:
        """
        将 datetime 格式化为字符串 "YYYY-MM-DD HH:MM:SS"
        """
        if dt is None:
            return None
        try:
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    def _strip_url_from_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        移除解析条目中的 url 字段，保留其他结构化信息
        
        参数：
        - item: 解析得到的文献条目字典，通常包含 url、title、authors、publication_info、abstract_snippet
        
        返回：
        - 去除 url 键后的字典副本
        """
        try:
            return {k: v for k, v in (item or {}).items() if k != "url"}
        except Exception:
            return {}

    def _sanitize_filename(self, name: Optional[str]) -> str:
        """
        清理用于文件名的字符串，移除不安全字符并限制长度
        
        参数：
        - name: 原始名称字符串
        
        返回：
        - 适用于文件系统的安全文件名片段
        """
        try:
            text = (name or "").strip()
            if not text:
                return "no_subject"
            # 统一为可见的有限字符集合
            text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5._-]+", "_", text)
            # 收敛长度，避免超长路径
            if len(text) > 80:
                text = text[:80]
            # 避免全是分隔符
            text = text.strip("._-")
            return text or "no_subject"
        except Exception:
            return "no_subject"

    def _save_email_html(self, html: Optional[str], received_time: Optional[str], subject: Optional[str]) -> Optional[str]:
        """
        将邮件的 HTML 文本保存到指定目录并返回文件路径
        
        参数：
        - html: 邮件 HTML 文本内容
        - received_time: 接收时间的字符串（可选）
        - subject: 邮件主题（可选）
        
        返回：
        - 成功保存时返回文件的绝对路径；否则返回 None
        """
        try:
            if not html or not html.strip():
                return None
            base_dir = os.path.join(
                "d:\\workproject\\paperProjects\\Scrapegraph-ai\\AIpaper2",
                "data",
                "email_html",
            )
            os.makedirs(base_dir, exist_ok=True)
            # 组织文件名：时间片段 + 主题片段 + 唯一序列
            dt_part = (received_time or "").replace(":", "-").replace(" ", "_") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            subj_part = self._sanitize_filename(subject)
            uniq_part = str(int(time.time() * 1000))
            filename = f"{dt_part}_{subj_part}_{uniq_part}.html"
            filepath = os.path.join(base_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            return filepath
        except Exception:
            return None

    def _get_next_div_sibling(self, tag, nth: int = 1):
        """
        获取给定标签的第 nth 个同级后续 div 标签
        
        参数：
        - tag: 当前 bs4 标签对象
        - nth: 第几个 div 兄弟（从 1 开始）
        
        返回：
        - 匹配到的 div 标签；未找到返回 None
        """
        try:
            count = 0
            sib = tag
            while True:
                sib = getattr(sib, "next_sibling", None)
                if sib is None:
                    return None
                # 跳过空白/字符串
                try:
                    if getattr(sib, "name", None) != "div":
                        continue
                except Exception:
                    continue
                count += 1
                if count == nth:
                    return sib
        except Exception:
            return None

    def _split_authors_and_meta(self, text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        将包含作者与元信息的文本按分隔符拆分
        
        参数：
        - text: 原始文本，可能形如 "Author1, Author2 - Journal 2025 - Publisher"
        
        返回：
        - (authors, publication_info)
        """
        try:
            if not text:
                return None, None
            parts = [p.strip() for p in str(text).split(" - ") if p.strip()]
            if len(parts) >= 2:
                authors = parts[0]
                publication_info = " - ".join(parts[1:])
                return authors or None, publication_info or None
            return text or None, None
        except Exception:
            return text or None, None

    def _imap_fetch_email_contents(
        self,
        imap_server: str,
        email_account: str,
        password: str,
        sender_email: str,
        days_recent: int,
        required_subject_contains: Optional[str],
    ) -> List[Tuple[str, Optional[str]]]:
        """
        通过 QQ 邮箱 IMAP 拉取符合条件的邮件正文内容，并返回 (正文, 收到时间) 列表
        """
        self.logger.info(
            f"邮件节点——开始拉取邮箱内容 imap_server={imap_server} sender_email={sender_email} "
            f"days_recent={days_recent} required_subject_contains={required_subject_contains or ''}"
        )
        contents: List[Tuple[str, Optional[str]]] = []
        mail = None
        for attempt in range(5):
            try:
                mail = imaplib.IMAP4_SSL(imap_server)
                mail.login(email_account, password)
                mail.select("inbox")
                break
            except Exception as e:
                self.logger.error(f"QQ 邮箱连接失败 attempt={attempt + 1} err={e}")
                try:
                    if mail is not None:
                        mail.logout()
                except Exception:
                    pass
                mail = None
                time.sleep(min(2 ** attempt, 10))

        if mail is None:
            self.logger.error("邮件节点——拉取失败：无法建立 IMAP 连接")
            return contents

        try:
            typ, data = mail.search(None, "FROM", sender_email)
            if typ != "OK":
                self.logger.warning(f"邮件节点——搜索失败 typ={typ}")
                return contents

            ids = data[0].split()
            ids = list(reversed(ids))[:300]
            self.logger.info(f"邮件节点——命中候选邮件 {len(ids)} 封（最多取 300 封）")

            earliest = _now_in_cn() - timedelta(days=max(int(days_recent), 0))
            scanned = 0
            accepted = 0
            dropped_by_sender = 0
            dropped_by_subject = 0
            dropped_by_date = 0
            dropped_by_decode = 0

            for eid in ids:
                try:
                    scanned += 1
                    typ_f, msg_data = mail.fetch(eid, "(RFC822)")
                    if typ_f != "OK":
                        dropped_by_decode += 1
                        continue
                    raw = msg_data[0][1]
                    email_message = email.message_from_bytes(raw, policy=policy.default)

                    msg_from = str(email_message.get("From") or "")
                    if sender_email and (sender_email not in msg_from):
                        dropped_by_sender += 1
                        continue

                    subject = str(email_message.get("Subject") or "")
                    if required_subject_contains and (required_subject_contains not in subject):
                        dropped_by_subject += 1
                        continue

                    date_str = str(email_message.get("Date") or "")
                    msg_dt = parsedate_to_datetime(date_str) if date_str else None
                    if msg_dt is not None:
                        msg_dt = _to_cn_time(msg_dt)
                        if int(days_recent) > 0 and msg_dt < earliest:
                            dropped_by_date += 1
                            self.logger.info("邮件节点——达到时间范围边界，提前结束拉取")
                            break

                    received_str = self._format_datetime_str(msg_dt) if msg_dt is not None else (date_str or None)

                    body = ""
                    html_content = ""
                    if email_message.is_multipart():
                        for part in email_message.walk():
                            ctype = part.get_content_type()
                            if ctype == "text/html":
                                html_content += self._decode_part_content(part)
                            if ctype in ["text/plain", "text/html"]:
                                body += self._decode_part_content(part)
                    else:
                        decoded = self._decode_part_content(email_message)
                        body = decoded
                        try:
                            if (email_message.get_content_type() or "").lower().strip() == "text/html":
                                html_content = decoded
                        except Exception:
                            pass

                    if body:
                        contents.append((body, received_str))
                        accepted += 1
                        if self.save_email_html:
                            try:
                                self._save_email_html(html_content, received_str, subject)
                            except Exception:
                                pass
                except Exception:
                    dropped_by_decode += 1
                    continue

        except Exception as e:
            self.logger.error(f"QQ 邮箱抓取失败: {e}")
            return contents
        finally:
            try:
                mail.close()
            except Exception:
                pass
            try:
                mail.logout()
            except Exception:
                pass
        self.logger.info(
            f"邮件节点——拉取完成 scanned={scanned} accepted={accepted} "
            f"dropped_sender={dropped_by_sender} dropped_subject={dropped_by_subject} "
            f"dropped_date={dropped_by_date} dropped_decode={dropped_by_decode}"
        )
        return contents

    def execute(self, state: dict) -> dict:
        """
        执行节点逻辑：
        - 根据 `email_config` 通过 IMAP 获取邮件正文（近 `days_recent` 天）
        - 提取论文网页链接并写入数据库
        - 不处理主题标签（主题分类在后续节点完成）
        - 输出 `papers` 列表
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        if "email_config" not in input_keys or not isinstance(state.get("email_config"), dict):
            raise ValueError("缺少 email_config：请在图输入中提供邮箱配置信息")

        email_config: dict = state["email_config"]

        emails: List[Tuple[str, Optional[str]]] = []
        if self.use_qq_email:
            imap_server = (email_config or {}).get("imap_server", "imap.qq.com")
            email_account = (email_config or {}).get("account") or ""
            password = (email_config or {}).get("password") or ""
            sender_email = (email_config or {}).get("sender_email", "scholaralerts-noreply@google.com")
            days_recent = int((email_config or {}).get("days_recent", 7))
            required_subject_contains = (email_config or {}).get("required_subject_contains")
            if not email_account or not password:
                raise ValueError("缺少 QQ 邮箱账号或授权码：请在 email_config 中提供 account/password")
            self.logger.info(
                f"邮件节点——开始获取邮件 sender_email={sender_email} days_recent={days_recent} "
                f"required_subject_contains={required_subject_contains or ''}"
            )
            emails = self._imap_fetch_email_contents(
                imap_server=imap_server,
                email_account=email_account,
                password=password,
                sender_email=sender_email,
                days_recent=days_recent,
                required_subject_contains=required_subject_contains,
            )
            self.logger.info(f"邮件节点——成功获取 {len(emails)} 条邮件正文")
        else:
            raise ValueError("EmailLinkNode 已禁用 QQ 邮箱抓取：请在 node_config 中设置 use_qq_email=True")

        papers: List[AIPaper] = []
        total_urls = 0
        unique_urls: List[str] = []
        unique_urls_set = set()
        inserted_count = 0
        existed_count = 0

        try:
            for idx, item in enumerate(emails, start=1):
                text, received_time = item
                parsed_items = self._parse_alert_items(text or "")
                subscribe_subject = self._infer_subscribe_from(text or "")
                total_urls += len(parsed_items)
                self.logger.info(f"邮件节点——第 {idx}/{len(emails)} 封邮件解析到 {len(parsed_items)} 条文献")
                for it in parsed_items:
                    url = it.get("url", "")
                    if url not in unique_urls_set:
                        unique_urls.append(url)
                        unique_urls_set.add(url)
                    existing = self.db.find_by_url(url)
                    if existing:
                        existed_count += 1
                        # 尝试补充缺失的结构化信息与订阅来源
                        updates: Dict[str, Any] = {}
                        try:
                            if not getattr(existing, "raw_email_text", None):
                                updates["raw_email_text"] = json.dumps(self._strip_url_from_item(it), ensure_ascii=False)
                            if subscribe_subject and not getattr(existing, "subscribe_from", None):
                                updates["subscribe_from"] = subscribe_subject
                            if updates and existing.id is not None:
                                self.db.update_fields(existing.id, updates)
                                if "raw_email_text" in updates:
                                    existing.raw_email_text = updates["raw_email_text"]
                                if "subscribe_from" in updates:
                                    existing.subscribe_from = updates["subscribe_from"]
                        except Exception:
                            pass
                        papers.append(existing)
                        
                        continue

                    paper = AIPaper(
                        id=None,
                        urlLink=url,
                        source="google_scholar_email",
                        pdfLink=None,
                        mdLink=None,
                        overviewLink=None,
                        analysisLink=None,
                        meta=None,
                        publishTime=None,
                        subject=None,
                        receivedTime=received_time,
                        type=None,
                        raw_email_text=json.dumps(self._strip_url_from_item(it), ensure_ascii=False),
                        subscribe_from=subscribe_subject,
                    )
                    new_id = self.db.insert_paper(paper)
                    paper.id = new_id
                    inserted_count += 1
                    papers.append(paper)
        except Exception as e:
            self.logger.error(f"邮件链接提取失败: {e}")
            raise

        self.logger.info(
            f"邮件节点——链接处理完成 total_urls={total_urls} unique_urls={len(unique_urls)} "
            f"inserted={inserted_count} existed={existed_count} papers_out={len(papers)}"
        )
        state.update({self.output[0]: papers})
        return state

def _decode_part_generic(part) -> str:
    """
    解码邮件内容片段为文本
    
    参数：
    - part: email.message.Message 或其子部件
    
    返回：
    - 该片段的解码文本内容
    """
    try:
        charset = part.get_content_charset() or "utf-8"
        payload = part.get_payload(decode=True) or b""
        cte = (part.get("Content-Transfer-Encoding") or "").lower().strip()
        if cte == "quoted-printable":
            try:
                return quopri.decodestring(payload).decode(charset, errors="ignore")
            except Exception:
                return payload.decode(charset, errors="ignore")
        if cte == "base64":
            try:
                return base64.b64decode(payload, validate=True).decode(charset, errors="ignore")
            except Exception:
                return payload.decode(charset, errors="ignore")
        return payload.decode(charset, errors="ignore")
    except Exception:
        return ""


def _decode_email_message(email_message) -> str:
    """
    解码整封邮件的正文文本（合并 text/plain 与 text/html）
    
    参数：
    - email_message: 完整的邮件对象
    
    返回：
    - 邮件正文的完整文本（可能包含 HTML）
    """
    try:
        if email_message.is_multipart():
            body = ""
            for part in email_message.walk():
                if part.get_content_type() in ["text/plain", "text/html"]:
                    body += _decode_part_generic(part)
            return body
        return _decode_part_generic(email_message)
    except Exception:
        return ""

def run_test_email_order(email_config: dict) -> None:
    """
    测试函数：基于 EmailLinkNode（L65）执行实际解析与入库，输出结果摘要
    
    参数:
        email_config: 邮箱配置字典，需包含 account、password、imap_server、sender_email、days_recent、required_subject_contains
    """
    logger = get_logger()
    if not (email_config or {}).get("account") or not (email_config or {}).get("password"):
        raise ValueError("缺少 QQ 邮箱账号或授权码：请在 email_config 中提供 account/password")
    try:
        node = EmailLinkNode(
            input="email_config",
            output=["papers"],
            node_config={
                "use_qq_email": True,
                "save_email_html": True,
            },
            node_name="EmailLinkExtractTest",
        )
        state = {"email_config": email_config}
        out_state = node.execute(state)
        papers = out_state.get("papers", []) if isinstance(out_state, dict) else []
        print(f"测试节点执行完成：papers={len(papers)}")
        for i, p in enumerate(papers, start=1):
            rid = getattr(p, "id", None)
            url = getattr(p, "urlLink", "")
            sub = getattr(p, "subscribe_from", None)
            raw = getattr(p, "raw_email_text", None)
            title = None
            try:
                if raw:
                    obj = json.loads(raw)
                    title = obj.get("title")
            except Exception:
                title = None
            print(f"{i}. id={rid} subscribe_from={sub or ''} title={title or ''} url={url}")
    except Exception as e:
        logger.error(f"EmailLinkNode 测试执行失败: {e}")
        print(f"测试失败：{e}")

if __name__ == "__main__":
    from scrapegraphai.utils import set_verbosity_info, set_formatting
    set_verbosity_info()
    set_formatting()

    from AIpaper2.google_scholar_paper_main import build_email_config
    cfg = build_email_config()
    run_test_email_order(cfg)

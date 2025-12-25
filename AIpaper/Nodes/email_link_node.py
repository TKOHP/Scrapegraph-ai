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
from datetime import datetime, timedelta
import time
import quopri
import base64
from typing import List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, unquote

from scrapegraphai.utils import get_logger
from scrapegraphai.nodes.base_node import BaseNode
try:
    from .db_manager import DatabaseManager, AIPaper
except ImportError:
    from AIpaper.Nodes.db_manager import DatabaseManager, AIPaper


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
        self.db_path = (self.node_config or {}).get("db_path", "AIpaper/data/google_scholar_papers.db")
        self.db = DatabaseManager(self.db_path)
        self.use_qq_email = bool((self.node_config or {}).get("use_qq_email", True))

    def _extract_urls(self, text: str) -> List[str]:
        """
        从文本中提取可能的论文网页地址
        """
        pattern = r"https?://[^\s<>\"]+"
        urls = re.findall(pattern, text)
        cleaned = []
        for u in urls:
            if u.startswith("http://") or u.startswith("https://"):
                cleaned.append(u.rstrip(").,;\"'"))
        return list(dict.fromkeys(cleaned))

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

    def _extract_urls_from_email_html(self, html: str) -> List[str]:
        """
        从邮件 HTML 内容中提取论文链接，优先解析 google scholar 的 alert 格式
        """
        urls: List[str] = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True, class_=lambda c: c and "gse_alrt_title" in c):
                href = a["href"]
                final = self._decode_google_redirect(href)
                if final.startswith("http"):
                    urls.append(final)
            if not urls:
                urls = self._extract_urls(html)
        except Exception:
            urls = self._extract_urls(html)
        # 去重
        dedup = []
        for u in urls:
            if u not in dedup:
                dedup.append(u)
        return dedup

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

            earliest = datetime.now() - timedelta(days=max(int(days_recent), 0))
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
                        msg_dt = msg_dt.replace(tzinfo=None) if getattr(msg_dt, "tzinfo", None) else msg_dt
                        if int(days_recent) > 0 and msg_dt < earliest:
                            dropped_by_date += 1
                            self.logger.info("邮件节点——达到时间范围边界，提前结束拉取")
                            break

                    received_str = self._format_datetime_str(msg_dt) if msg_dt is not None else (date_str or None)

                    body = ""
                    if email_message.is_multipart():
                        for part in email_message.walk():
                            if part.get_content_type() in ["text/plain", "text/html"]:
                                body += self._decode_part_content(part)
                    else:
                        body = self._decode_part_content(email_message)

                    if body:
                        contents.append((body, received_str))
                        accepted += 1
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
                urls = self._extract_urls_from_email_html(text or "")
                total_urls += len(urls)
                self.logger.info(f"邮件节点——第 {idx}/{len(emails)} 封邮件提取到 {len(urls)} 条链接")
                for url in urls:
                    if url not in unique_urls_set:
                        unique_urls.append(url)
                        unique_urls_set.add(url)
                    existing = self.db.find_by_url(url)
                    if existing:
                        existed_count += 1
                        papers.append(existing)
                        
                        continue

                    paper = AIPaper(
                        id=None,
                        urlLink=url,
                        pdfLink=None,
                        mdLink=None,
                        overviewLink=None,
                        analysisLink=None,
                        meta=None,
                        publishTime=None,
                        subject=None,
                        receivedTime=received_time,
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

def run_test_email_order(email_config: dict) -> None:
    """
    测试函数：列出符合条件的邮件标题与时间，并判断是否按时间倒序排序
    
    参数:
        email_config: 邮箱配置字典，需包含 account、password、imap_server、sender_email、days_recent、required_subject_contains
    """
    logger = get_logger()
    imap_server = (email_config or {}).get("imap_server", "imap.qq.com")
    email_account = (email_config or {}).get("account") or ""
    password = (email_config or {}).get("password") or ""
    sender_email = (email_config or {}).get("sender_email", "scholaralerts-noreply@google.com")
    days_recent = int((email_config or {}).get("days_recent", 7))
    required_subject_contains = (email_config or {}).get("required_subject_contains")
    if not email_account or not password:
        raise ValueError("缺少 QQ 邮箱账号或授权码：请在 email_config 中提供 account/password")
    logger.info(
        f"测试邮件排序——开始获取邮件 sender_email={sender_email} days_recent={days_recent} "
        f"required_subject_contains={required_subject_contains or ''}"
    )
    mail = None
    try:
        t_all_start = time.perf_counter()
        t0 = time.perf_counter()
        mail = imaplib.IMAP4_SSL(imap_server)
        connect_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        mail.login(email_account, password)
        login_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        mail.select("inbox")
        select_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        typ, data = mail.search(None, "FROM", sender_email)
        search_time = time.perf_counter() - t0
        if typ != "OK":
            print("邮件搜索失败")
            return
        ids = data[0].split()
        ids = list(reversed(ids))[:300]
        earliest = datetime.now() - timedelta(days=max(int(days_recent), 0))
        items = []
        scanned = 0
        fetch_total_time = 0.0
        parse_total_time = 0.0
        max_fetch_time = 0.0
        max_parse_time = 0.0
        for eid in ids:
            scanned += 1
            try:
                tf = time.perf_counter()
                typ_f, msg_data = mail.fetch(eid, "(RFC822)")
                fetch_dt = time.perf_counter() - tf
                fetch_total_time += fetch_dt
                if fetch_dt > max_fetch_time:
                    max_fetch_time = fetch_dt
                if typ_f != "OK":
                    continue
                raw = msg_data[0][1]
                tp = time.perf_counter()
                email_message = email.message_from_bytes(raw, policy=policy.default)
                subject = str(email_message.get("Subject") or "").strip()
                date_str = str(email_message.get("Date") or "")
                dt = None
                try:
                    dt = parsedate_to_datetime(date_str) if date_str else None
                    if dt is not None:
                        dt = dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt
                    if days_recent > 0 and dt is not None and dt < earliest:
                        print("达到时间范围边界，提前结束拉取")
                        break
                except Exception:
                    dt = None
                if required_subject_contains and (required_subject_contains not in subject):
                    continue
                parse_dt = time.perf_counter() - tp
                parse_total_time += parse_dt
                if parse_dt > max_parse_time:
                    max_parse_time = parse_dt
                items.append({"subject": subject, "date": dt, "raw_date": date_str})
            except Exception:
                continue
        t_all = (time.perf_counter() - t_all_start)
        print(f"测试邮件统计：总数={len(items)} 扫描={scanned}")
        for i, it in enumerate(items, start=1):
            disp = it["date"].strftime("%Y-%m-%d %H:%M:%S") if it["date"] else (it["raw_date"] or "")
            print(f"{i}. {disp} | {it['subject']}")
        is_desc = True
        for i in range(len(items) - 1):
            left = items[i]["date"] or datetime.min
            right = items[i + 1]["date"] or datetime.min
            if left < right:
                is_desc = False
                break
        print(f"是否按时间倒序：{is_desc}")
        print(
            "阶段耗时(ms)："
            f"连接={connect_time*1000:.1f} 登录={login_time*1000:.1f} 选择={select_time*1000:.1f} "
            f"搜索={search_time*1000:.1f} 拉取总计={fetch_total_time*1000:.1f} 解析总计={parse_total_time*1000:.1f} 总计={t_all*1000:.1f}"
        )
        durations = {
            "连接": connect_time,
            "登录": login_time,
            "选择": select_time,
            "搜索": search_time,
            "拉取总计": fetch_total_time,
            "解析总计": parse_total_time,
        }
        longest_stage = max(durations.items(), key=lambda kv: kv[1])
        print(
            f"单封最大拉取耗时(ms)={max_fetch_time*1000:.1f} 最大解析耗时(ms)={max_parse_time*1000:.1f}"
        )
        print(f"耗时最长阶段：{longest_stage[0]} {longest_stage[1]*1000:.1f} ms")
    except Exception as e:
        logger.error(f"测试邮件排序失败: {e}")
        print(f"测试失败：{e}")
    finally:
        try:
            if mail is not None:
                mail.close()
        except Exception:
            pass

def build_email_config_for_test() -> dict:
    """
    构造测试所需的邮箱配置字典（QQ 邮箱）
    
    返回字段:
        - imap_server: IMAP 服务地址（默认 imap.qq.com）
        - account: 邮箱账号，默认读取环境变量 QQ_EMAIL
        - password: 邮箱授权码或密码，默认读取环境变量 QQ_PASSWORD
        - sender_email: 过滤的发件人（默认 Google Scholar 提醒）
        - days_recent: 近期天数过滤
        - required_subject_contains: 主题包含关键词，可选，默认读取环境变量 REQUIRED_SUBJECT_CONTAINS
    """
    imap_server = os.getenv("QQ_IMAP_SERVER", "imap.qq.com")
    account = os.getenv("QQ_EMAIL", "")
    password = os.getenv("QQ_PASSWORD", "")
    required_subject_contains = os.getenv("REQUIRED_SUBJECT_CONTAINS", None)
    return {
        "imap_server": imap_server,
        "account": account,
        "password": password,
        "sender_email": "scholaralerts-noreply@google.com",
        "days_recent": int(os.getenv("DAYS_RECENT", "2") or "2"),
        "required_subject_contains": required_subject_contains,
    }

if __name__ == "__main__":
    from scrapegraphai.utils import set_verbosity_info, set_formatting
    set_verbosity_info()
    set_formatting()
    try:
        from AIpaper.google_scholar_paper_main import build_email_config
        cfg = build_email_config()
        run_test_email_order(cfg)
    except Exception as e:
        try:
            cfg = build_email_config_for_test()
            run_test_email_order(cfg)
        except Exception as e2:
            print(f"输入或执行发生错误：{e2}")

#!/usr/bin/env python3
"""PixelRAG 知识库桌面客户端 —— 「Notion 纸感」浅色聊天界面。

打开即自动检测/启动检索服务,输入问题后检索最相关的论文页面截图,交给视觉模型
流式生成答案,正文下方附引用卡片(文件名 / 页码 / 缩略图 / 点击打开 PDF)。
顶栏可切换深色主题。

界面样式全部取自 ``pixelrag_theme`` 的设计令牌;tkinter 缺失的圆角、hover 等能力
由 ``pixelrag_widgets`` 用 Pillow 离屏渲染补齐。检索与生成逻辑与旧版一致。

依赖:标准库 tkinter + 已安装的 Pillow、anthropic。
"""

import base64
import io
import json
import os
import re
import subprocess
import threading
import time
import tkinter as tk
import urllib.request

import anthropic
from PIL import Image, ImageTk

import pixelrag_history as H
import pixelrag_theme as T
import pixelrag_widgets as W

ROOT = os.path.dirname(os.path.abspath(__file__))
HOST = "localhost"
PORT = 30001
N_DOCS = 5
HEALTH_URL = f"http://{HOST}:{PORT}/health"
SEARCH_URL = f"http://{HOST}:{PORT}/search"
ARTICLES_JSON = os.path.join(ROOT, "knowledge_index", "articles.json")
TILES_DIR = os.path.join(
    ROOT, "knowledge_index", "tiles"
)  # 与 _serve_cmd 里的 --tiles-dir 同一处

# ---- VLM 生成配置(DeepSeek-V4.1-Flash,原生视觉)----
VLM_MODEL = (
    "deepseek-flash"  # DeepSeek-V4.1-Flash 官方主名;旧名 deepseek-v4-flash 已下线
)
VLM_BASE_URL = "https://api.deepseek.com/anthropic"
VLM_MAX_IMAGES = 6  # 最多发几张截图
VLM_MAX_SIDE = 1568  # 截图长边像素上限(超出即压缩)
# 输出上限。注意:deepseek-flash 是带思考的模型,思考 token 也计入这里 ——
# 图多、问题复杂时思考轻松吃掉 2000+,上限太小会导致"一个字都没写就被切断"
# (界面表现就是"模型没有返回任何内容")。实测 8192 能稳定作答。
VLM_MAX_TOKENS = 8192

# 窗口逻辑尺寸(会按屏幕 DPI 缩放)。比原来宽,是因为左边多了历史栏。
WIN_W, WIN_H = 1140, 800
WIN_MIN_W = 880  # 侧栏可收起,但收起后对话区也不该挤到读不了


# ---------------------------------------------------------------------------
# 服务与检索(与旧版一致,未改动)
# ---------------------------------------------------------------------------


def check_health() -> bool:
    try:
        urllib.request.urlopen(HEALTH_URL, timeout=3)
        return True
    except Exception:
        return False


def _serve_cmd() -> list[str]:
    exe = os.path.join(ROOT, ".venv", "Scripts", "pixelrag.exe")
    return [
        exe,
        "serve",
        "--index-dir",
        os.path.join(ROOT, "knowledge_index"),
        "--tiles-dir",
        os.path.join(ROOT, "knowledge_index", "tiles"),
        "--articles-json",
        os.path.join(ROOT, "knowledge_index", "articles.json"),
        "--device",
        "cpu",
        "--port",
        str(PORT),
    ]


def start_serve_process() -> subprocess.Popen:
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        _serve_cmd(),
        env=env,
        creationflags=flags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def search(text: str, n_docs: int = N_DOCS) -> dict:
    payload = json.dumps(
        {"queries": [{"text": text}], "n_docs": n_docs, "include_images": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        SEARCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))


def title_of(url: str) -> str:
    name = os.path.basename((url or "").replace("\\", "/"))
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name


def decode_b64_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _load_titles() -> dict[int, str]:
    """articles.json 的下标即 article_id,用它拿到比文件名更干净的标题。"""
    try:
        with open(ARTICLES_JSON, encoding="utf-8") as f:
            arts = json.load(f)
        return {i: (a.get("title") or "") for i, a in enumerate(arts)}
    except Exception:
        return {}


def refs_of(hits) -> list[dict]:
    """把命中列表压成"引用页"列表,按 (文章, 页, 段) 去重。

    存进历史的是**数据不是图片**:缩略图由 :func:`history_tiles` 从本地 tiles 现读,
    所以历史文件很小,而且 tiles 重新渲染过也照样对得上。

    去重是因为同一个问题走不同路径(直接生成 / 降级只显示引用)可能拿到同一页两条命中,
    不去重历史里就会存两份一模一样的卡片。
    """
    out: list[dict] = []
    seen: set = set()
    for h in hits or []:
        if not isinstance(h, dict):
            continue
        ref = {
            "article_id": h.get("article_id"),
            "tile_index": h.get("tile_index"),
            "chunk_index": h.get("chunk_index"),
            "score": h.get("score"),
            "url": h.get("url"),
        }
        key = (ref["article_id"], ref["tile_index"], ref["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def tiles_from(hits) -> list[tuple[str, dict]]:
    """命中列表 → 引用卡片要的 ``(图片 base64, hit)``。

    按相似度降序排、按 (文章, 页, 段) 去重,并且丢掉没带截图的命中 —— 卡片是靠
    截图显示"引用的是哪一页"的,没有图就没得显示。
    """
    out: list[tuple[str, dict]] = []
    seen: set = set()
    for h in sorted(hits or [], key=lambda x: float(x.get("score", 0.0)), reverse=True):
        if not isinstance(h, dict):
            continue
        key = (h.get("article_id"), h.get("tile_index"), h.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        b64 = h.get("image_base64")
        if b64:
            out.append((b64, h))
    return out


def history_tiles(item: dict) -> list[tuple[str, dict]]:
    """把历史里的引用页还原成 ``(图片 base64, hit)``。

    读的是本地 tiles 目录,所以打开旧问答既不联网也不用检索服务;
    文件被删掉时缩略图留空,卡片其余信息照常显示。
    """
    out: list[tuple[str, dict]] = []
    for ref in item.get("refs") or []:
        if not isinstance(ref, dict):
            continue
        b64 = ""
        try:
            article = int(ref.get("article_id"))
            tile = int(ref.get("tile_index"))
            path = os.path.join(
                TILES_DIR, f"{article}.png.tiles", f"tile_{tile:04d}.jpg"
            )
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception:
            b64 = ""
        out.append((b64, dict(ref)))
    return out


def screen_scale() -> float:
    try:
        import ctypes

        return ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
    except Exception:
        return 1.0


SYSTEM_PROMPT = (
    "你是一个科研文献视觉问答助手。用户会提供若干张论文页面的截图,并附上一个问题。"
    "请只根据这些截图中的可见内容回答,严格遵守:\n"
    "1. 只能依据截图内容回答,严禁编造截图里没有的数据或事实。\n"
    '2. 若截图信息不足以回答,直接说"根据现有资料无法回答",不要猜测。\n'
    "3. 回答要具体、简洁,直接给出结论。\n"
    "4. 每个关键事实后用 [1][2] 等编号标注,编号对应第几张截图(从 1 开始)。\n"
    "5. 若答案来自某张图表,说明依据的是第几张图的哪个部分。\n"
)


def _load_vlm_config() -> dict:
    """读取 VLM 配置:优先环境变量 PIXELRAG_VLM_*,其次 ANTHROPIC_* 环境变量,最后回退 ~/.claude/settings.json。"""
    cfg = {
        "api_key": os.environ.get("PIXELRAG_VLM_API_KEY"),
        "base_url": os.environ.get("PIXELRAG_VLM_BASE_URL"),
        "model": os.environ.get("PIXELRAG_VLM_MODEL"),
    }
    if not cfg["api_key"]:
        cfg["api_key"] = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        cfg["base_url"] = os.environ.get("ANTHROPIC_BASE_URL") or VLM_BASE_URL
    if not cfg["api_key"]:
        try:
            p = os.path.join(os.path.expanduser("~"), ".claude", "settings.json")
            env = json.load(open(p, encoding="utf-8")).get("env", {})
            cfg["api_key"] = env.get("ANTHROPIC_AUTH_TOKEN")
            cfg["base_url"] = env.get("ANTHROPIC_BASE_URL") or VLM_BASE_URL
        except Exception:
            pass
    cfg["base_url"] = cfg["base_url"] or VLM_BASE_URL
    cfg["model"] = cfg["model"] or VLM_MODEL
    return cfg


def prepare_images(hits: list, k: int = VLM_MAX_IMAGES) -> list:
    """取 Top-K 相关截图:按相关度降序、按 (article_id, tile_index) 去重、压缩为长边 ≤ VLM_MAX_SIDE 的 JPEG。

    返回 [(b64_jpeg, hit), ...],顺序即引用编号 [1][2]...。
    """
    seen = set()
    out = []
    for h in sorted(hits, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        key = (h.get("article_id"), h.get("tile_index"))
        if key in seen:
            continue
        seen.add(key)
        b64 = h.get("image_base64")
        if not b64:
            continue
        try:
            img = decode_b64_image(b64)
            img.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64_out = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            continue
        out.append((b64_out, h))
        if len(out) >= k:
            break
    return out


def vlm_stream(images, question: str, cfg: dict):
    """把多张截图按相关度顺序 + 问题发给 VLM,流式 yield 答案文本(过滤 thinking)。

    两个容易被误判成"网络坏了"的情况在这里说明白:模型把输出额度全花在思考上、
    一个字都没写出来时,抛一条能看懂的错误,而不是静默返回空串。
    """
    client = anthropic.Anthropic(api_key=cfg["api_key"], base_url=cfg["base_url"])
    content = []
    for i, (b64, _h) in enumerate(images, 1):
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            }
        )
        content.append({"type": "text", "text": f"[第{i}张截图]"})
    content.append({"type": "text", "text": question})

    got_text = False
    with client.messages.stream(
        model=cfg["model"],
        max_tokens=VLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        for ev in stream:
            if getattr(ev, "type", None) == "text":
                got_text = True
                yield ev.text
        stop_reason = getattr(stream.get_final_message(), "stop_reason", None)

    if stop_reason == "max_tokens":
        if not got_text:
            raise RuntimeError(
                f"模型的思考占满了输出上限(max_tokens={VLM_MAX_TOKENS}),"
                "还没来得及作答就被切断了。请再试一次,或把 pixelrag_client.py 里的 "
                "VLM_MAX_TOKENS 调得更大。"
            )
        yield "\n\n（回答达到长度上限,可能不完整）"


_CODE_RE = re.compile(r"```[ \t]*[\w+-]*\n(.*?)```", re.DOTALL)


def _split_code_blocks(text: str) -> list[tuple[str, str]]:
    """把答案切成 ('text'|'code', 片段) 序列,用于分别排版代码块。"""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _CODE_RE.finditer(text):
        if m.start() > pos:
            out.append(("text", text[pos : m.start()]))
        out.append(("code", m.group(1).rstrip("\n")))
        pos = m.end()
    if pos < len(text):
        out.append(("text", text[pos:]))
    return out or [("text", text)]


# ---- 答案正文里的行内标记 ---------------------------------------------------
# 模型给的答案不是纯文本,常见的是「**小标题**」和行首「* 条目」,原样塞进 Text 就会
# 把星号显示出来。这里只认模型真会写的那三种,不做通用 Markdown —— 宁可漏掉罕见的
# 写法,也不要把正文里的星号当成标记吃掉(数学式 a*b、脚注 * 之类都很常见)。
_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.DOTALL)
_BULLET_RE = re.compile(r"^([ \t]*)\*[ \t]+")
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+")


def _rich_runs(text: str) -> list[tuple[str, str]]:
    """把答案正文切成 ``(文字, 标签)``:标签 ``""`` 是普通文字,``"bold"`` 是加粗。

    - ``**强调**`` → 去掉星号,文字加粗(模型常拿它当小标题用,粗体得保住);
    - 行首 ``#`` 标题 → 去掉 # 号与空格,整行加粗(各级都是加粗,不改字号);
    - 行首 ``* 条目`` → 星号换成圆点(列表还是列表,屏幕上不再有星号);
    - ``-`` 开头的列表不动:它不是星号,换成圆点反而和破折号撞脸。

    代码块不经过这里 —— 调用方先用 :func:`_split_code_blocks` 把它们分出去了。
    """
    runs: list[tuple[str, str]] = []
    pos = 0
    for m in _BOLD_RE.finditer(text):
        if m.start() > pos:
            _plain_runs(text[pos : m.start()], runs)
        runs.append((m.group(1), "bold"))
        pos = m.end()
    if pos < len(text):
        _plain_runs(text[pos:], runs)
    return runs


def _plain_runs(chunk: str, runs: list[tuple[str, str]]) -> None:
    """普通片段里的行首标记:先换圆点、再剥 ``#`` 标题,逐行处理(换行符原样留着)。"""
    for line in chunk.splitlines(keepends=True):
        head = _BULLET_RE.match(line)
        if head:
            # 只换星号那一个字符:后面用来对齐的空格原样保留,列表的缩进才不变
            i = len(head.group(1))
            line = line[:i] + "•" + line[i + 1 :]
        m = _HEADING_RE.match(line)
        if m:
            body = line[m.end() :]
            nl = "\n" if body.endswith("\n") else ""
            body = body[:-1] if nl else body
            # 整行本来就要加粗,行内的 ** 就只剩标记作用了,一并去掉
            body = _BOLD_RE.sub(r"\1", body)
            if body:
                runs.append((body, "bold"))
            if nl:
                runs.append((nl, ""))
        elif line:
            runs.append((line, ""))


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PixelRAG 视觉问答")
        root.configure(bg=T.c("bg"))
        root.minsize(int(WIN_MIN_W * T.UI_SCALE), int(480 * T.UI_SCALE))

        # 消息数据模型:主题切换 / 窗口缩放时据此重放渲染
        self.messages: list[dict] = []
        self._photos: list[ImageTk.PhotoImage] = []  # 防止图片被 GC
        self._answer_text: tk.Text | None = None
        self._answer_msg: dict | None = None
        self._paint_job = None
        self._loading_job = None
        self._loading_active = False
        self._loading_base = "正在检索相关页面"
        self._resize_job = None
        self._send_hover = False
        self._stop_hover = False
        self._busy = False
        # 停止生成:主线程置位,后台消费循环轮询。generator 另存一份是为了能
        # 显式 close(),让底层的 HTTP 流立刻断开而不是等 GC。
        self._stop_event = threading.Event()
        self._gen = None
        self._stopped = False
        self._last_question: str | None = None
        # 查询历史:落盘存储 + 左侧栏 + 输入框上下键召回
        self.history = H.History()
        self.sidebar: H.HistorySidebar | None = None
        self._sidebar_open = True  # 顶栏「历史」按钮切换它
        self._current_hist: dict | None = None  # 当前打开的那轮问答
        self._hist_entry: dict | None = None  # 本次提问对应的历史条目,答完回填
        self._refetching = False  # 正在给老记录补检索引用页(同时只补一条)
        self._hist_pos: int | None = None  # None = 不在召回态;0 = 最新一条
        self._hist_draft = ""  # 召回前用户已经敲了一半的内容
        self._serve_ready = check_health()
        self._titles = _load_titles()
        self._content_w = T.CONTENT_MAX_WIDTH
        self._inset = T.SPACE[
            "xl"
        ]  # 消息列到画布边的留白(宽屏时变大,见 _content_metrics)
        self._canvas_w = 0

        self._apply_window_geometry()
        self._build_ui()
        self._add_welcome()
        self._set_status()

        if not self._serve_ready:
            threading.Thread(target=self._ensure_serve, daemon=True).start()

    def _apply_window_geometry(self):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(int(WIN_W * T.UI_SCALE), sw - int(80 * T.UI_SCALE))
        h = min(int(WIN_H * T.UI_SCALE), sh - int(120 * T.UI_SCALE))
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ---------- UI 构建 ----------

    def _build_ui(self):
        self._build_topbar()
        self._build_bottom_bar()
        self._build_body()

    def _build_topbar(self):
        self.topbar = tk.Frame(self.root, bg=T.c("surface"))
        self.topbar.pack(side="top", fill="x")
        inner = tk.Frame(self.topbar, bg=T.c("surface"))
        inner.pack(fill="x", padx=T.SPACE["lg"], pady=T.SPACE["sm"])

        self.brand = tk.Label(
            inner,
            text="PixelRAG",
            bg=T.c("surface"),
            fg=T.c("text"),
            font=T.font("title"),
        )
        self.brand.pack(side="left")
        self.brand_sub = tk.Label(
            inner,
            text="视觉问答",
            bg=T.c("surface"),
            fg=T.c("text_faint"),
            font=T.font("caption"),
        )
        self.brand_sub.pack(
            side="left", padx=(T.SPACE["sm"], 0), pady=(T.SPACE["xs"], 0)
        )

        # 主题切换
        self.theme_btn = tk.Label(
            inner,
            text="",
            bg=T.c("surface"),
            fg=T.c("text_muted"),
            font=T.font("caption"),
            cursor="hand2",
            padx=T.SPACE["sm"],
            pady=T.SPACE["xs"],
        )
        self.theme_btn.pack(side="right")
        self.theme_btn.bind("<Button-1>", lambda _e: self._toggle_theme())
        W.make_focusable(self.theme_btn, T.c("surface"), self._toggle_theme)
        W.bind_hover(
            self.theme_btn,
            on_enter=lambda: self.theme_btn.configure(
                fg=T.c("accent_text"), bg=T.c("accent_soft")
            ),
            on_leave=lambda: self.theme_btn.configure(
                fg=T.c("text_muted"), bg=T.c("surface")
            ),
        )
        self._sync_theme_btn()

        # 查询历史
        self.history_btn = tk.Label(
            inner,
            text="历史",
            bg=T.c("surface"),
            fg=T.c("text_muted"),
            font=T.font("caption"),
            cursor="hand2",
            padx=T.SPACE["sm"],
            pady=T.SPACE["xs"],
        )
        self.history_btn.pack(side="right", padx=(0, T.SPACE["xs"]))
        self.history_btn.bind("<Button-1>", lambda _e: self._on_history())
        W.make_focusable(self.history_btn, T.c("surface"), self._on_history)
        W.bind_hover(
            self.history_btn,
            on_enter=lambda: self.history_btn.configure(
                fg=T.c("accent_text"), bg=T.c("accent_soft")
            ),
            on_leave=lambda: self.history_btn.configure(
                fg=T.c("text_muted"), bg=T.c("surface")
            ),
        )

        self.status = tk.Label(
            inner,
            text="",
            bg=T.c("surface"),
            fg=T.c("text_muted"),
            font=T.font("caption"),
        )
        self.status.pack(side="right", padx=(0, T.SPACE["md"]))

        self.topbar_line = tk.Frame(self.root, bg=T.c("border"), height=1)
        self.topbar_line.pack(side="top", fill="x")

    def _sync_theme_btn(self):
        text = "切换深色" if T.mode() == "light" else "切换浅色"
        self.theme_btn.configure(text=text)

    def _build_bottom_bar(self):
        self.bottom = tk.Frame(self.root, bg=T.c("bg"))
        self.bottom.pack(side="bottom", fill="x")
        tk.Frame(self.bottom, bg=T.c("border"), height=1).pack(side="top", fill="x")

        # 输入框正对着正文那条窄栏:同宽、同留白,不是通栏(DeepSeek 就是这样)。
        # 留白在 _on_canvas_configure 里跟着聊天画布的宽度同步 —— 侧栏一开一合,
        # 输入框和消息列会一起挪,不会各走各的。
        self.bottom_row = tk.Frame(self.bottom, bg=T.c("bg"))
        self.bottom_row.pack(
            fill="x", padx=T.SPACE["xl"], pady=(T.SPACE["md"], T.SPACE["sm"])
        )

        # 输入框:圆角方框 + 1px 描边,聚焦时描边变主色(DeepSeek 样式)。
        # tk 没有 CSS 圆角,所以复用引用卡片那套画法 —— Canvas 铺一张圆角底图并
        # tag_lower 沉底,再用 create_window 放进 Text 和两个按钮。Text 自己的底色
        # 和填充同色,于是方角被藏进圆角里,只有描边露出来。
        self.input_box = tk.Canvas(
            self.bottom_row, bg=T.c("bg"), highlightthickness=0, bd=0, height=1
        )
        self.input_box.pack(fill="x", expand=True)
        self._input_bg_id = None
        self._input_win_id = None
        self._input_photo = None
        self._input_focused = False
        self._input_pad = (T.SPACE["lg"], T.SPACE["md"])  # 文字到描边的距离

        # 发送 / 停止按钮是画布里的窗口项,浮在方框右下角(DeepSeek 的按钮在框内)。
        # 底色取 input_bg 而不是 bg:它们压在圆角填充上,方角才不会露出来。
        # bd/padx/pady 全清零:Label 的默认边框和内边距会给按钮图套一圈看不见的
        # 外框,让底图看起来离描边比设定的间距更远。
        self.send_btn = tk.Label(
            self.input_box,
            bg=T.c("input_bg"),
            bd=0,
            padx=0,
            pady=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.send_btn.bind("<Button-1>", lambda _e: self._on_send())
        W.bind_hover(
            self.send_btn, on_enter=self._send_hover_on, on_leave=self._send_hover_off
        )
        W.make_focusable(self.send_btn, T.c("input_bg"), self._on_send)

        # 停止按钮:只在生成期间显示(位置一直预留着,文字区不会跟着跳)
        self.stop_btn = tk.Label(
            self.input_box,
            bg=T.c("input_bg"),
            bd=0,
            padx=0,
            pady=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.stop_btn.bind("<Button-1>", lambda _e: self._on_stop())
        W.bind_hover(
            self.stop_btn, on_enter=self._stop_hover_on, on_leave=self._stop_hover_off
        )
        W.make_focusable(self.stop_btn, T.c("input_bg"), self._on_stop)
        self._stop_shown = False
        self._stop_item = None
        self._send_item = None
        self._btn_reserve = 0  # 文字右侧留给按钮的宽度(建完按钮才知道)
        self._send_img_w = self._stop_img_w = 0
        self._bottom_inset = (-1, -1)  # 底栏当前 (左, 右) 留白,用来跳过无变化的设置

        sp1, sp2, sp3 = W.line_spacing(T.font("input"))
        self.input = tk.Text(
            self.input_box,
            height=2,
            bg=T.c("input_bg"),
            fg=T.c("text"),
            insertbackground=T.c("accent"),
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=T.font("input"),
            wrap="word",
            padx=0,
            pady=0,
            spacing1=sp1,
            spacing2=sp2,
            spacing3=sp3,
        )
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Escape>", lambda _e: self._clear_input())
        self.input.bind("<Control-a>", self._select_all_input)
        # ↑/↓ 召回历史:只在光标处于首行(↑)/已在召回态(↓)时接管,否则让 tk 正常移光标
        self.input.bind("<Up>", self._on_input_up)
        self.input.bind("<Down>", self._on_input_down)
        W.attach_text_menu(self.input, self.root, editable=True)
        self.input.bind("<FocusIn>", lambda _e: self._layout_input_box(True))
        self.input.bind("<FocusOut>", lambda _e: self._layout_input_box(False))

        # 先把两张按钮图渲染出来:一是量出预留宽度,二是保证第一次要显示停止按钮时
        # 手里已经有图(_update_stop_button 在未显示时是直接返回的)。
        # 预留宽度按"停止按钮也在"的最宽情况算一次就定住 —— 之后停止按钮显隐、发送
        # 按钮在「发送 / 生成中」之间切换,输入框里的文字都不会跟着伸缩。
        self._render_send_image()
        self._render_stop_image()
        # 发送按钮的宽度被 min_w 钉住(「发送」和「生成中」一样宽),所以按当前这张量
        # 就等于按最宽情况量。
        self._btn_reserve = (
            T.SPACE["sm"] + self._send_img_w + T.SPACE["sm"] + self._stop_img_w
        )
        self._update_send_button()
        self.stop_btn.configure(image=self._stop_photo)
        self.stop_btn.image = self._stop_photo  # 先装上,量宽度也让它有尺寸
        self.input_box.bind("<Configure>", lambda _e: self._layout_input_box())
        self.root.after_idle(self._layout_input_box)

        hint = tk.Label(
            self.bottom,
            text="Enter 发送  ·  Shift+Enter 换行  ·  ↑ 历史  ·  Esc 清空",
            bg=T.c("bg"),
            fg=T.c("text_faint"),
            font=T.font("micro"),
        )
        hint.pack(side="top", pady=(0, T.SPACE["sm"]))

    def _layout_input_box(self, focused: bool | None = None):
        """重画输入框的圆角底,并把 ``Text`` 摆到内边距里面。

        ``focused=None`` 表示"焦点态没变,只是尺寸变了"(``<Configure>`` 用)。底图走
        ``W.rounded_photo``,两张(聚焦/未聚焦)按 key 缓存,来回切焦点不会重画。
        """
        box = self.input_box
        if not box.winfo_exists():
            return
        if focused is not None:
            self._input_focused = focused
        pad_x, pad_y = self._input_pad
        # -height 是"几行",tk 已经把它折算成含 spacing1/2/3 的完整行盒;拿不到就按
        # 令牌兜底(理论上不会走到)。输入框不随内容长高,超长内容由 Text 自己滚动。
        text_h = int(self.input.winfo_reqheight() or 0) or 2 * T.line_box_px("input")
        w = max(1, box.winfo_width())
        h = text_h + 2 * pad_y
        if int(box.cget("height")) != h:
            box.configure(height=h)
        photo = W.rounded_photo(
            w,
            h,
            T.RADIUS["lg"],
            T.c("input_bg"),
            T.c("bg"),
            outline=T.c("focus") if self._input_focused else T.c("input_line"),
        )
        self._input_photo = photo  # 必须持有,否则会被 GC 成白块
        if self._input_bg_id is None:
            self._input_bg_id = box.create_image(0, 0, anchor="nw", image=photo)
            box.tag_lower(self._input_bg_id)
        else:
            box.itemconfigure(self._input_bg_id, image=photo)
            box.coords(self._input_bg_id, 0, 0)

        # 文字区右端让开按钮区(预留宽度在 _build_bottom_bar 里按最宽情况定死)
        edge = T.SPACE["sm"]
        win_w = max(1, w - pad_x - self._btn_reserve)
        if self._input_win_id is None:
            self._input_win_id = box.create_window(
                pad_x, pad_y, anchor="nw", window=self.input, width=win_w, height=text_h
            )
        else:
            box.coords(self._input_win_id, pad_x, pad_y)
            box.itemconfigure(self._input_win_id, width=win_w, height=text_h)

        # 按钮贴在方框右下角:停止在左、发送在右,离描边各留一个小间距。
        # 焦点环(highlight)也算在控件外框里,锚点要把它让出来,底图才真的离描边 edge。
        ring = int(self.send_btn.cget("highlightthickness") or 0)
        btn_y = h - edge + ring
        send_w = self._send_img_w
        if self._send_item is None:
            self._send_item = box.create_window(
                w - edge + ring, btn_y, anchor="se", window=self.send_btn
            )
        else:
            box.coords(self._send_item, w - edge + ring, btn_y)
        stop_x = w - edge + ring - send_w - T.SPACE["sm"]
        if self._stop_item is None:
            self._stop_item = box.create_window(
                stop_x,
                btn_y,
                anchor="se",
                window=self.stop_btn,
                state="normal" if self._stop_shown else "hidden",
            )
        else:
            box.coords(self._stop_item, stop_x, btn_y)
        box.tag_raise(self._stop_item)
        box.tag_raise(self._send_item)

    def _layout_bottom_row(self, avail_w: int):
        """让输入框与消息列左右对齐(同宽同留白)。

        ``avail_w`` 是聊天画布的宽度 —— 输入框要对齐的是正文那条窄栏,不是整个窗口,
        所以侧栏一开一合输入框也要跟着挪。
        """
        _, inset = self._content_metrics(avail_w)
        # 底栏是通栏的(侧栏一开一合它不动),所以要让开聊天区左边那截侧栏、右边那截
        # 滚动条,才能和消息列真正对齐 —— 只补 inset 会在侧栏打开时整体左移一整个侧栏。
        left = right = inset
        sb = getattr(self, "sidebar", None)
        if sb is not None and sb.winfo_exists():
            left += sb.winfo_width() + 1  # 侧栏 + 它右边那条 1px 分隔线
        sc = getattr(self, "scroll", None)
        if sc is not None and sc.winfo_exists():
            right += sc.winfo_width()
        if (left, right) == self._bottom_inset:
            return
        self._bottom_inset = (left, right)
        self.bottom_row.pack_configure(padx=(left, right))

    def _build_body(self):
        body = tk.Frame(self.root, bg=T.c("bg"))
        body.pack(side="top", fill="both", expand=True)
        self.body = body

        # 左侧历史栏(DeepSeek 式)。顶栏/底栏仍是通栏,只有中间分左右。
        self.sidebar = None
        if self._sidebar_open:
            self.sidebar = H.HistorySidebar(
                body,
                self.history,
                on_open=self._open_history_item,
                on_reask=self._reask,
                on_new=self._new_chat,
            )
            self.sidebar.pack(side="left", fill="y")
            tk.Frame(body, bg=T.c("border"), width=1).pack(side="left", fill="y")
            self.sidebar.set_current(self._current_hist)

        self.canvas = tk.Canvas(body, bg=T.c("bg"), highlightthickness=0, bd=0)
        self.scroll = tk.Scrollbar(
            body,
            orient="vertical",
            command=self.canvas.yview,
            bg=T.c("scrollbar"),
            troughcolor=T.c("bg"),
            activebackground=T.c("border_strong"),
            borderwidth=0,
            width=T.SCROLLBAR_W,
            relief="flat",
            elementborderwidth=0,
        )
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.frame = tk.Frame(self.canvas, bg=T.c("bg"))
        self._win = self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_frame_configure(self, _e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _content_metrics(self, canvas_w: int) -> tuple[int, int]:
        """消息列的 ``(列宽, 左右留白)``。

        DeepSeek 的正文是一条居中的窄栏:列宽封顶 ``CONTENT_MAX_WIDTH``,窗口再宽也
        不变宽,多出来的宽度两边均分。窗口窄的时候留白退回 ``SPACE["xl"]``(此时
        列宽本身就等于可用宽度,没有富余可分)。
        """
        avail = max(1, canvas_w - 2 * T.SPACE["xl"])
        w = max(280, min(T.CONTENT_MAX_WIDTH, avail))
        return w, max(T.SPACE["xl"], (canvas_w - w) // 2)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._win, width=event.width)
        self._layout_bottom_row(event.width)  # 输入框跟着消息列一起居中对齐
        new_w, new_inset = self._content_metrics(event.width)
        # 画布宽变了也要重排(即使 _content_w 已到上限):两侧留白是按画布宽算的,
        # 不重排的话窗口一宽正文就一直贴在左边,和居中的引用卡片对不齐。
        moved = abs(event.width - self._canvas_w) > 8
        self._canvas_w = event.width
        if (
            abs(new_w - self._content_w) > 8
            or abs(new_inset - self._inset) > 8
            or moved
        ):
            self._content_w, self._inset = new_w, new_inset
            self._schedule_reflow()

    def _schedule_reflow(self):
        """窗口尺寸变化后防抖重排(圆角图与折行宽度都需要重算)。"""
        if self._resize_job:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after(
            T.MOTION["resize_debounce"], self._render_all
        )

    def _on_wheel(self, event):
        # 这个绑定是 bind_all 的,滚轮落在侧栏上时不该带着对话区一起滚
        if self.sidebar is not None and str(event.widget).startswith(str(self.sidebar)):
            return
        try:
            self.canvas.yview_scroll(int(-event.delta / 120) * 3, "units")
        except Exception:
            pass

    # ---------- 状态 ----------

    def _set_status(self, state: str | None = None):
        if state == "starting":
            self.status.configure(text="● 正在启动检索服务", fg=T.c("warning"))
        elif state == "failed":
            self.status.configure(text="● 服务启动失败(见控制台)", fg=T.c("danger"))
        elif self._serve_ready:
            self.status.configure(text="● 服务运行中 · 端口 30001", fg=T.c("success"))
        else:
            self.status.configure(text="● 服务启动中", fg=T.c("warning"))

    # ---------- 消息模型与渲染 ----------

    def _append(self, msg: dict) -> dict:
        stick = self._at_bottom()  # 先取状态:追加后必然"不在底部"
        self.messages.append(msg)
        self._render_message(msg)
        self._scroll_bottom(stick)
        return msg

    def _add_welcome(self):
        self._append(
            {
                "kind": "welcome",
                "text": "输入问题,我会检索知识库里的论文页面截图,并基于截图内容作答。\n"
                "答案下方给出引用来源,点击卡片可打开对应的 PDF。",
            }
        )

    def _add_user(self, text: str):
        self._append({"kind": "user", "text": text})

    def _add_note(self, text: str):
        self._append({"kind": "note", "text": text})

    def _add_error(self, text: str, retry: bool = False):
        """``retry=True`` 时在这条错误下挂一个「重新生成」按钮。"""
        self._append({"kind": "error", "text": text, "retry": retry})

    def _render_all(self, view_state: tuple[bool, float] | None = None):
        """清空并重放全部消息(主题切换 / 窗口尺寸变化时调用)。

        ``view_state`` 由调用方在重建界面前取得;不传则读当前画布。
        """
        self._resize_job = None
        stick, top = view_state if view_state is not None else self._view_state()
        was_loading = self._loading_active
        self._cancel_loading_job()
        self._loading_active = False

        for w in self.frame.winfo_children():
            w.destroy()
        self._photos.clear()
        self._answer_text = None

        # 每条消息只带"下边距 = SPACE['msg']"(消息间距只有这一个来源),所以最上面
        # 那条前面得补一份上留白,否则会被画布顶边贴住。
        if self.messages:
            tk.Frame(self.frame, bg=T.c("bg"), height=T.SPACE["msg"]).pack(fill="x")

        for msg in self.messages:
            self._render_message(msg)
        if was_loading:
            self._start_loading(self._loading_base)

        self.canvas.update_idletasks()  # 让 bbox 反映新内容,否则滚动范围是旧的
        self._on_frame_configure()
        try:
            self.canvas.yview_moveto(1.0 if stick else top)
        except Exception:
            pass
        if stick:
            self._scroll_bottom(True)  # 几何稳定后再吸一次,避免滚动范围随后变大

    def _render_message(self, msg: dict):
        kind = msg.get("kind")
        if kind == "user":
            self._render_user(msg)
        elif kind == "welcome":
            self._render_welcome(msg)
        elif kind == "note":
            self._render_note(msg)
        elif kind == "error":
            self._render_error(msg)
        elif kind == "section":
            self._render_section(msg["text"])
        elif kind == "answer":
            self._render_answer(msg)
        elif kind == "citations":
            self._render_citations(msg)

    # ---- 各类消息 ----

    def _render_welcome(self, msg: dict):
        wrap = tk.Frame(self.frame, bg=T.c("bg"))
        wrap.pack(fill="x", padx=self._inset, pady=(T.SPACE["2xl"], T.SPACE["lg"]))
        tk.Label(
            wrap,
            text=msg["text"],
            bg=T.c("bg"),
            fg=T.c("text_muted"),
            font=T.font("body"),
            justify="center",
            wraplength=self._content_w - 2 * T.SPACE["md"],
        ).pack(anchor="center")

    def _render_user(self, msg: dict):
        row = tk.Frame(self.frame, bg=T.c("bg"))
        row.pack(fill="x", padx=self._inset, pady=(0, T.SPACE["msg"]))
        pad_x, pad_y = T.BUBBLE_PAD
        img = W.render_label_box(
            msg["text"],
            "bubble",
            T.c("user_text"),
            T.c("bg"),
            box_fill=T.c("user_bubble"),
            radius=T.RADIUS["md"],
            pad_x=pad_x,
            pad_y=pad_y,
            max_w=int(self._content_w * 0.78),
        )
        photo = W.photo_of(img)
        self._photos.append(photo)
        lbl = tk.Label(row, image=photo, bg=T.c("bg"), bd=0)
        lbl.image = photo
        lbl.pack(side="right")

    def _render_note(self, msg: dict):
        tk.Label(
            self.frame,
            text=msg["text"],
            bg=T.c("bg"),
            fg=T.c("text_muted"),
            font=T.font("body"),
            justify="left",
            anchor="w",
            wraplength=self._content_w,
        ).pack(fill="x", padx=self._inset, pady=(0, T.SPACE["msg"]))

    def _render_error(self, msg: dict):
        row = tk.Frame(self.frame, bg=T.c("bg"))
        row.pack(fill="x", padx=self._inset, pady=(0, T.SPACE["msg"]))
        img = W.render_label_box(
            msg["text"],
            "body",
            T.c("danger"),
            T.c("bg"),
            box_fill=T.c("danger_bg"),
            radius=T.RADIUS["md"],
            pad_x=T.SPACE["md"],
            pad_y=T.SPACE["sm"],
            max_w=self._content_w,
        )
        photo = W.photo_of(img)
        self._photos.append(photo)
        lbl = tk.Label(row, image=photo, bg=T.c("bg"), bd=0)
        lbl.image = photo
        lbl.pack(anchor="w")  # 竖排:气泡在上,重试按钮在下
        if msg.get("retry"):
            self._render_retry_button(row)

    def _render_retry_button(self, parent):
        """次要按钮:描边胶囊,悬停变主色。"""
        btn = tk.Label(parent, bg=T.c("bg"), bd=0, cursor="hand2")
        btn.pack(anchor="w", pady=(T.SPACE["sm"], 0))

        def paint(hover: bool):
            img = W.render_label_box(
                "重新生成",
                "button",
                T.c("accent_text") if hover else T.c("text"),
                T.c("bg"),
                box_fill=T.c("card_hover") if hover else T.c("card"),
                outline=T.c("accent") if hover else T.c("border_strong"),
                radius=T.RADIUS_PILL,
                pad_x=T.SPACE["md"],
                pad_y=T.SPACE["sm"],
                align="center",
            )
            photo = W.photo_of(img)
            self._photos.append(photo)
            try:
                btn.configure(image=photo)
                btn.image = photo
            except Exception:
                pass

        paint(False)
        btn.bind("<Button-1>", lambda _e: self._on_regenerate())
        W.bind_hover(btn, on_enter=lambda: paint(True), on_leave=lambda: paint(False))
        W.make_focusable(btn, T.c("bg"), self._on_regenerate)

    def _render_section(self, text: str):
        wrap = tk.Frame(self.frame, bg=T.c("bg"))
        # 区块标题属于它上面那条消息的一部分,所以贴着上面走(上边距 0),只留
        # 下边距 —— 它的"和上一条消息的距离"由上面那条的 msg 边距提供。
        wrap.pack(fill="x", padx=self._inset, pady=(0, T.SPACE["sm"]))
        tk.Label(
            wrap,
            text=text,
            bg=T.c("bg"),
            fg=T.c("text_muted"),
            font=T.font("h2"),
        ).pack(side="left")
        tk.Frame(wrap, bg=T.c("border"), height=1).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(T.SPACE["sm"], 0),
            pady=(T.SPACE["sm"], 0),
        )

    def _render_answer(self, msg: dict):
        container = tk.Frame(self.frame, bg=T.c("bg"))
        container.pack(fill="x", padx=self._inset, pady=(0, T.SPACE["msg"]))
        msg["_widget"] = container

        # 正文用只读 Text 而不是 Label —— Label 不能选中,复制不出来。
        # Text 没有 wraplength,行宽靠容器的 padx=self._inset 压回 _content_w,
        # 否则窗口一宽,长行就会横贯整个画布(和引用卡片对不齐)。
        txt = W.selectable_text(container, T.font("answer"), T.c("bg"), T.c("text"))
        txt.pack(fill="x")
        W.attach_text_menu(txt, self.root)
        # 代码块同样按 LINE_HEIGHT 铺行距,但要按**等宽字体自己的**行盒算:两种字体
        # 的自然行高差得不少,照抄正文那个像素值会让代码行挤在一起。tk 的折行间距
        # 会被对半分给折点两侧,所以 spacing1/3 取 gap 的上/下半(与 selectable_text 同理)。
        code_font = T.font_mono(T.px_size("code"))
        sp1, sp2, sp3 = W.line_spacing(code_font)
        txt.tag_configure(
            "code",
            font=code_font,
            background=T.c("code_bg"),
            lmargin1=T.SPACE["md"],
            lmargin2=T.SPACE["md"],
            rmargin=T.SPACE["md"],
            spacing1=sp1,
            spacing2=sp2,
            spacing3=sp3,
        )
        # 行内加粗(答案里的小标题)。它自带行距,理由和代码块一样:粗体的行盒比
        # 正文字体高 1px,不补 spacing 的话同一段里会隔三差五冒出一行高 1px。
        bold_font = T.font("answer", "bold")
        bs1, bs2, bs3 = W.line_spacing(bold_font)
        txt.tag_configure(
            "bold", font=bold_font, spacing1=bs1, spacing2=bs2, spacing3=bs3
        )
        self._answer_text = txt

        text = msg.get("text", "")
        if msg.get("streaming"):
            self._set_answer_into(txt, (text + " ▌") if text else "…", settle=True)
            self._answer_msg = msg
            return

        self._fill_answer(txt, text)
        self._render_copy_button(container, text)

    def _set_answer_into(self, txt: tk.Text, s: str, settle: bool = False):
        """整体重写正文。内容短、节流到 60ms 一次,所以不做增量插入。

        流式也走 :func:`_rich_runs`:否则生成过程中会先看到一堆星号,定稿时才消失。
        代价是半个标记(``**开头`` 还没等到收尾)会短暂按原样显示,收尾一到就变粗。
        """
        txt.delete("1.0", "end")
        self._insert_runs(txt, s)
        W.fit_height(txt, settle=settle)

    def _fill_answer(self, txt: tk.Text, text: str):
        """定稿排版:代码段套 code 标签(等宽 + 淡底 + 内缩),其余按行内标记分粗/不粗。"""
        for seg_kind, seg_text in _split_code_blocks(text):
            if seg_kind == "code":
                txt.insert("end", seg_text + "\n", "code")
            elif seg_text:
                self._insert_runs(txt, seg_text)
        W.fit_height(txt, settle=True)

    @staticmethod
    def _insert_runs(txt: tk.Text, text: str):
        """按 :func:`_rich_runs` 的分段插入:加粗的套 bold 标签,其余原样。"""
        for s, tag in _rich_runs(text):
            if tag:
                txt.insert("end", s, tag)
            else:
                txt.insert("end", s)

    def _render_copy_button(self, parent, text: str):
        bar = tk.Frame(parent, bg=T.c("bg"))
        bar.pack(fill="x", pady=(T.SPACE["sm"], 0))
        btn = tk.Label(
            bar,
            text="复制答案",
            bg=T.c("bg"),
            fg=T.c("text_faint"),
            font=T.font("micro"),
            cursor="hand2",
            padx=T.SPACE["xs"],
        )
        btn.pack(side="left")

        def copy(_e=None):
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            btn.configure(text="已复制", fg=T.c("success"))
            self.root.after(
                1400, lambda: btn.configure(text="复制答案", fg=T.c("text_faint"))
            )

        btn.bind("<Button-1>", copy)
        W.bind_hover(
            btn,
            on_enter=lambda: btn.configure(fg=T.c("accent_text")),
            on_leave=lambda: btn.configure(fg=T.c("text_faint")),
        )
        W.make_focusable(btn, T.c("bg"), copy)

    def _render_citations(self, msg: dict):
        self._render_section("引用来源")
        # 补检索来的引用页要写清楚来路:这些页不是当时那几页,不能让人当成本来的出处
        note = msg.get("note")
        if note:
            tk.Label(
                self.frame,
                text=note,
                bg=T.c("bg"),
                fg=T.c("text_faint"),
                font=T.font("caption"),
                justify="left",
                anchor="w",
                wraplength=self._content_w,
            ).pack(fill="x", padx=self._inset, pady=(0, T.SPACE["sm"]))
        tiles = msg["tiles"]
        # 卡片是一组:卡与卡之间用 sm(挨着才像一组),整组结束后才轮到消息间距。
        for i, (b64, hit) in enumerate(tiles, 1):
            self._render_citation_card(b64, hit, i, last=(i == len(tiles)))

    def _render_citation_card(self, b64: str, hit: dict, idx: int, last: bool = True):
        """白底描边卡片:缩略图 + 标题 + 页码/相似度。点缩略图看大图,点其余开 PDF。"""
        pad = T.SPACE["md"]
        w = max(240, self._content_w)
        thumb_box = (int(74 * T.UI_SCALE), int(96 * T.UI_SCALE))

        card = tk.Canvas(
            self.frame,
            width=w,
            height=int(104 * T.UI_SCALE),  # 占位高度,量完信息块后修正
            bg=T.c("bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        card.pack(
            fill="x",
            padx=self._inset,
            pady=(0, T.SPACE["msg"] if last else T.SPACE["sm"]),
        )

        title = self._titles.get(hit.get("article_id")) or title_of(hit.get("url", ""))
        x = pad
        thumb_lbl = None
        thumb_h = 0
        if b64:
            try:
                img = decode_b64_image(b64)
                img.thumbnail(thumb_box, Image.LANCZOS)
                thumb = W.photo_of(img)
                self._photos.append(thumb)
                thumb_lbl = tk.Label(
                    card, image=thumb, bg=T.c("card"), bd=0, cursor="hand2"
                )
                thumb_lbl.image = thumb
                x += thumb.width() + pad
                thumb_h = thumb.height()
            except Exception:
                thumb_lbl = None

        tw = max(120, w - x - pad)

        # 信息块:交给 Frame + pack 自己排垂直布局,避免手算 y 导致换行标题被盖住
        info = tk.Frame(card, bg=T.c("card"))
        title_lbl = tk.Label(
            info,
            text=f"[{idx}]  {title}",
            bg=T.c("card"),
            fg=T.c("accent_text"),  # 白底上的强调色**文字**用 accent_text(4.33:1 的
            # accent 是给填充/描边用的,小字不加粗时对比度不够)
            font=T.font("label"),
            justify="left",
            anchor="w",
            wraplength=tw,
            cursor="hand2",
        )
        title_lbl.pack(anchor="w", fill="x")

        page = int(hit.get("tile_index", 0)) + 1
        meta = f"第 {page} 页"
        if int(hit.get("chunk_index", 0) or 0) > 0:
            meta += f" · 第 {int(hit['chunk_index']) + 1} 段"
        meta += f" · 相似度 {float(hit.get('score', 0.0)):.3f}"
        meta_lbl = tk.Label(
            info,
            text=meta,
            bg=T.c("card"),
            fg=T.c("text_muted"),
            font=T.font("caption"),
            justify="left",
            anchor="w",
        )
        meta_lbl.pack(anchor="w", fill="x", pady=(T.SPACE["xs"], 0))

        hint_lbl = tk.Label(
            info,
            text="点击打开 PDF",
            bg=T.c("card"),
            fg=T.c("text_faint"),
            font=T.font("micro"),
            justify="left",
            anchor="w",
            cursor="hand2",
        )
        hint_lbl.pack(anchor="w", fill="x", pady=(T.SPACE["sm"], 0))

        info.update_idletasks()
        h = max(int(80 * T.UI_SCALE), thumb_h, info.winfo_reqheight()) + 2 * pad
        card.configure(height=h)

        # 三种外观:常态 / 悬停 / 键盘聚焦(聚焦用主色描边,与 hover 区分开)
        normal = W.rounded_photo(
            w, h, T.RADIUS["md"], T.c("card"), T.c("bg"), outline=T.c("border")
        )
        hover = W.rounded_photo(
            w,
            h,
            T.RADIUS["md"],
            T.c("card_hover"),
            T.c("bg"),
            outline=T.c("border_strong"),
        )
        focused = W.rounded_photo(
            w,
            h,
            T.RADIUS["md"],
            T.c("card_hover"),
            T.c("bg"),
            outline=T.c("focus"),
            outline_w=2,
        )
        self._photos.extend([normal, hover, focused])
        bg_id = card.create_image(0, 0, anchor="nw", image=normal)
        card.tag_lower(bg_id)  # 背景图是后建的,必须沉底,否则会盖住信息块

        card.create_window(x, pad, anchor="nw", window=info, width=tw)
        if thumb_lbl is not None:
            card.create_window(
                pad, max(pad, (h - thumb_h) // 2), anchor="nw", window=thumb_lbl
            )

        def repaint(color: str):
            def walk(widget):
                for child in widget.winfo_children():
                    try:
                        child.configure(bg=color)
                    except Exception:
                        pass
                    walk(child)

            walk(card)

        def open_pdf(_e=None):
            self._open_hit(hit, title)

        # 悬停与聚焦会同时发生,用一份状态决定最终外观(聚焦优先)
        state = {"hover": False, "focus": False}

        def paint(_e=None):
            if state["focus"]:
                card.itemconfigure(bg_id, image=focused)
                repaint(T.c("card_hover"))
            elif state["hover"]:
                card.itemconfigure(bg_id, image=hover)
                repaint(T.c("card_hover"))
            else:
                card.itemconfigure(bg_id, image=normal)
                repaint(T.c("card"))

        def set_state(key, value):
            def apply(_e=None):
                state[key] = value
                paint()

            return apply

        W.bind_hover(card, set_state("hover", True), set_state("hover", False))
        card.bind("<FocusIn>", set_state("focus", True), add="+")
        card.bind("<FocusOut>", set_state("focus", False), add="+")

        # 键盘可达:Tab 聚焦,Enter / Space 打开 PDF
        card.configure(takefocus=1)
        for key in ("<Return>", "<KP_Enter>", "<space>"):
            card.bind(key, open_pdf, add="+")

        for widget in (card, info, title_lbl, meta_lbl, hint_lbl):
            widget.bind("<Button-1>", open_pdf, add="+")
        if thumb_lbl is not None:
            thumb_lbl.bind(
                "<Button-1>", lambda _e: self._open_image(b64, title), add="+"
            )

    # ---- 加载态 ----

    def _start_loading(self, text: str = "正在检索相关页面"):
        if self._loading_active:
            return
        self._loading_active = True
        self._loading_base = text
        self._loading_frame = tk.Frame(self.frame, bg=T.c("bg"))
        self._loading_frame.pack(fill="x", padx=self._inset, pady=(0, T.SPACE["msg"]))
        self._loading_label = tk.Label(
            self._loading_frame,
            text=text,
            bg=T.c("bg"),
            fg=T.c("text_muted"),
            font=T.font("body"),
            anchor="w",
            justify="left",
        )
        self._loading_label.pack(fill="x")

        skel = tk.Frame(self._loading_frame, bg=T.c("bg"))
        skel.pack(fill="x", pady=(T.SPACE["sm"], 0))
        for frac in (0.92, 0.68):
            bar = W.rounded_photo(
                max(40, int(self._content_w * frac)),
                int(12 * T.UI_SCALE),
                T.RADIUS["sm"],
                T.c("skeleton"),
                T.c("bg"),
            )
            self._photos.append(bar)
            lbl = tk.Label(skel, image=bar, bg=T.c("bg"), bd=0)
            lbl.image = bar
            lbl.pack(anchor="w", pady=(0, T.SPACE["xs"]))

        self._dot = 0
        self._tick_loading()
        self._scroll_bottom(True)

    def _tick_loading(self):
        if not self._loading_active:
            return
        self._dot = (self._dot + 1) % 4
        stick = self._at_bottom()
        try:
            self._loading_label.configure(text=self._loading_base + "·" * self._dot)
        except Exception:
            return
        self._loading_job = self.root.after(420, self._tick_loading)
        self._scroll_bottom(stick)

    def _set_loading_text(self, text: str):
        self._loading_base = text
        if self._loading_active:
            try:
                self._loading_label.configure(text=text)
            except Exception:
                pass

    def _cancel_loading_job(self):
        if self._loading_job:
            try:
                self.root.after_cancel(self._loading_job)
            except Exception:
                pass
            self._loading_job = None

    def _stop_loading(self):
        self._cancel_loading_job()
        self._loading_active = False
        frame = getattr(self, "_loading_frame", None)
        if frame is not None:
            try:
                frame.destroy()
            except Exception:
                pass
            self._loading_frame = None

    # ---------- 滚动 ----------

    def _at_bottom(self) -> bool:
        """当前视图是否停在底部。必须在改动内容【之前】调用才有意义。"""
        try:
            return self.canvas.yview()[1] >= 0.995
        except Exception:
            return True

    def _view_state(self) -> tuple[bool, float]:
        """(是否停在底部, 顶部比例)。重建界面前取一次,重建后据此还原。"""
        try:
            top, bottom = self.canvas.yview()
            return bottom >= 0.995, top
        except Exception:
            return True, 0.0

    def _scroll_bottom(self, stick: bool = True):
        """stick=True 时把视图吸到底部;False 表示用户正在上翻,不去抢。"""
        if not stick:
            return

        def go():
            try:
                # 顺序很关键:先让几何算完并刷新滚动范围,再移动。
                # 否则 moveto(1.0) 会按【旧】范围钳制,结果停在中途。
                self.canvas.update_idletasks()
                h = max(self.frame.winfo_reqheight(), self.canvas.winfo_height())
                self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), h))
                self.canvas.yview_moveto(1.0)
            except Exception:
                pass

        self.root.after_idle(go)

    # ---------- 交互 ----------

    def _on_enter(self, event):
        if event.state & 0x0001:  # Shift+Enter → 换行
            return None
        self._on_send()
        return "break"

    def _clear_input(self):
        self.input.delete("1.0", "end")

    def _on_send(self):
        if self._busy:
            return
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        self.input.delete("1.0", "end")
        self._add_user(text)

        if not self._serve_ready:
            self._add_note("检索服务尚未就绪,正在启动,请稍候几秒再试…")
            threading.Thread(target=self._ensure_serve, daemon=True).start()
            return

        self._run_query(text)

    def _run_query(self, text: str):
        """检索 + 生成。重试时复用同一个问题,不再追加用户气泡。"""
        self._last_question = text
        self._stopped = False
        self._stop_event.clear()
        # 记进历史("重新生成"同问同答,会在存储层合并成一条)
        self._hist_entry = self.history.remember(text)
        self._hist_pos = None
        self._hist_draft = ""
        # 新问一轮就离开"打开旧问答"的状态;这一步顺带把新条目刷进侧栏
        self._current_hist = None
        if self.sidebar is not None:
            self.sidebar.set_current(None)
        self._busy = True
        self._update_send_button()
        self._show_stop(True)
        self._start_loading("正在检索相关页面")
        threading.Thread(target=self._do_search, args=(text,), daemon=True).start()

    def _on_stop(self):
        """停止生成:置取消信号,后台循环下一轮就退出,已收到的文字保留。"""
        if not self._busy:
            return
        self._stopped = True
        self._stop_event.set()
        if self._loading_active:
            self._set_loading_text("正在停止…")
        self._update_stop_button()

    def _on_regenerate(self):
        """重新生成:用上一次的问题重跑。答案为空或生成失败时给出这个入口。"""
        if self._busy or not self._last_question:
            return
        self._run_query(self._last_question)

    def _select_all_input(self, _e=None):
        self.input.tag_add("sel", "1.0", "end-1c")
        self.input.mark_set("insert", "1.0")
        return "break"

    # ---------- 查询历史 ----------

    def _on_history(self):
        """顶栏「历史」= 收起 / 展开左侧栏。"""
        self._sidebar_open = not self._sidebar_open
        self._rebuild_ui()

    def _refresh_sidebar(self):
        if self.sidebar is not None:
            self.sidebar.refresh()

    def _new_chat(self):
        """开一段新对话:主区域回到欢迎语,历史一条不动。"""
        if self._busy:  # 生成中换视图会让答案落到错误的对话里
            return
        self._current_hist = None
        self.messages = []
        self._render_all((True, 0.0))
        self._add_welcome()
        self._clear_input()
        if self.sidebar is not None:
            self.sidebar.set_current(None)
        self._scroll_bottom(True)

    def _open_history_item(self, item: dict):
        """打开一条历史:还原当时的提问 / 答案 / 引用卡片。

        答案和引用页都存在历史里,所以这里不检索、不调模型;缩略图直接读本地 tiles。
        """
        if self._busy:
            return
        self._current_hist = item
        if self.sidebar is not None:
            self.sidebar.set_current(item)
        self._clear_input()
        self.messages = []
        self._render_all((False, 0.0))
        self._add_user(item.get("question", ""))
        answer = item.get("answer") or ""
        if answer:
            self._append({"kind": "answer", "text": answer, "streaming": False})
        else:
            self._add_note("这条提问没有存下答案,点它右边的「重问」可以重新生成。")
        tiles = history_tiles(item)
        if tiles:
            self._append({"kind": "citations", "tiles": tiles})
        elif (item.get("question") or "").strip():
            # 这条记录里没有引用页(存引用页的功能上线前留下的老记录,或者检索成功
            # 但没走完生成的那一轮)。按原问题补跑一次检索,别让引用卡片就这么空着。
            self._refetch_refs(item)
        self._scroll_bottom(True)

    def _refetch_refs(self, item: dict):
        """给没存下引用页的记录补一次检索,把引用卡片补齐。

        只问本地检索服务,没有模型调用,所以打开旧问答依然是"不花一分钱"。补到的页
        写回历史,下次点开直接就有;补不到就明说没补到 —— 空着不吭声会让人以为
        这条记录本来就没有引用。
        """
        if self._refetching or not (item.get("question") or "").strip():
            return
        if not self._serve_ready:
            self._add_note("这条记录当初没有存下引用页,当前检索服务未就绪,没法补齐。")
            return
        self._refetching = True
        question = item["question"]
        self._start_loading("正在补齐这条记录的引用页")

        def work():
            try:
                hits = search(question)["results"][0]["hits"]
            except Exception:
                hits = []
            self.root.after(0, lambda: apply(hits))

        def apply(hits: list):
            self._refetching = False
            self._stop_loading()
            tiles = tiles_from(hits)
            if tiles:
                # 数据先写回历史:"这条记录引用过哪几页"是跟这条记录有关的事实,
                # 与用户此刻正在看哪一条无关。补到一次就存下来,下次点开不用再查。
                self.history.update(
                    item, hits=len(tiles), refs=refs_of([h for _b64, h in tiles])
                )
                self._refresh_sidebar()
            # 这期间用户可能已经点开别的记录、或者又问了一轮 —— 内容就别往新对话里塞
            if self._current_hist is not item:
                return
            if not tiles:
                self._add_note(
                    "这条记录当初没有存下引用页,刚按原问题补检索也没拿到截图。"
                    "「重问」可以重新生成完整的一轮。"
                )
                return
            self._append(
                {
                    "kind": "citations",
                    "tiles": tiles,
                    "note": "这条记录当初没有存下引用页。以下是按原问题重新检索到的"
                    "页面,未必是当时那几页;要还原当时那一轮请点「重问」。",
                }
            )

        threading.Thread(target=work, daemon=True).start()

    def _reask(self, question: str):
        """从历史里点一条:填进输入框并直接提问。"""
        self._replace_input(question)
        self._on_send()

    def _replace_input(self, text: str):
        self.input.delete("1.0", "end")
        self.input.insert("1.0", text)
        self.input.mark_set("insert", "end-1c")
        self.input.focus_set()

    def _on_input_up(self, _e=None):
        """↑:光标在首行时召回上一条历史问题。"""
        on_first_line = self.input.index("insert").split(".")[0] == "1"
        if self._hist_pos is not None or on_first_line:
            return self._recall_history(+1)
        return None  # 交给 tk:光标上移一行

    def _on_input_down(self, _e=None):
        """↓:只有在召回态才接管(往回翻到更新的一条)。"""
        if self._hist_pos is None:
            return None
        return self._recall_history(-1)

    def _recall_history(self, delta: int):
        """``delta=+1`` 往更早翻,``-1`` 往更新翻;翻过头就恢复召回前的草稿。"""
        items = self.history.questions()
        if not items:
            return None
        if self._hist_pos is None:
            self._hist_draft = self.input.get("1.0", "end-1c")
            self._hist_pos = 0
        else:
            self._hist_pos += delta
        if self._hist_pos < 0:  # 回到最新之后 = 回到草稿
            self._hist_pos = None
            self._replace_input(self._hist_draft)
            return "break"
        self._hist_pos = min(self._hist_pos, len(items) - 1)
        self._replace_input(items[self._hist_pos])
        return "break"

    def _send_hover_on(self):
        self._send_hover = True
        self._update_send_button()

    def _send_hover_off(self):
        self._send_hover = False
        self._update_send_button()

    def _stop_hover_on(self):
        self._stop_hover = True
        self._update_stop_button()

    def _stop_hover_off(self):
        self._stop_hover = False
        self._update_stop_button()

    def _show_stop(self, show: bool):
        """停止按钮的显隐 = 画布窗口项的显示/隐藏(位置一直给它留着)。"""
        if show == self._stop_shown:
            return
        self._stop_shown = show
        if show:
            self._update_stop_button()
        if self._stop_item is not None:
            self.input_box.itemconfigure(
                self._stop_item, state="normal" if show else "hidden"
            )

    def _render_stop_image(self):
        """画「停止生成」的按钮图(不管此刻显不显示 —— 建界面时要先量宽度)。"""
        fill = T.c("border_strong") if self._stop_hover else T.c("card")
        img = W.render_label_box(
            "停止生成",
            "button",
            T.c("text"),
            T.c("input_bg"),
            box_fill=fill,
            outline=T.c("border_strong"),
            radius=T.RADIUS_PILL,
            pad_x=T.SPACE["lg"],
            pad_y=T.SPACE["sm"],
            align="center",
        )
        photo = W.photo_of(img)
        self._stop_photo = photo
        self._stop_img_w = img.width
        return photo

    def _update_stop_button(self):
        photo = self._render_stop_image()
        if not self._stop_shown:
            return
        try:
            self.stop_btn.configure(image=photo, cursor="hand2")
            self.stop_btn.image = photo
        except Exception:
            pass

    def _render_send_image(self):
        """画「发送 / 生成中」的按钮图(按当前 busy / hover 态)。"""
        if self._busy:
            fill, fg, label, cursor = (
                T.c("border"),
                T.c("text_faint"),
                "生成中",
                "arrow",
            )
        elif self._send_hover:
            fill, fg, label, cursor = (
                T.c("accent_hover"),
                T.c("on_accent"),
                "发送",
                "hand2",
            )
        else:
            fill, fg, label, cursor = T.c("accent"), T.c("on_accent"), "发送", "hand2"
        img = W.render_label_box(
            label,
            "button",
            fg,
            T.c("input_bg"),
            box_fill=fill,
            radius=T.RADIUS_PILL,
            pad_x=T.SPACE["lg"],
            pad_y=T.SPACE["sm"],
            align="center",
            min_w=T.SEND_MIN_W,
        )
        photo = W.photo_of(img)
        self._send_photo = photo
        self._send_img_w = img.width
        return photo, cursor

    def _update_send_button(self):
        photo, cursor = self._render_send_image()
        try:
            self.send_btn.configure(image=photo, cursor=cursor)
            self.send_btn.image = photo
        except Exception:
            pass

    # ---------- 检索 → 生成 ----------

    def _do_search(self, text: str):
        try:
            resp = search(text)
            hits = resp["results"][0]["hits"]
        except Exception as e:
            self.root.after(0, self._stop_loading)
            self.root.after(0, lambda err=e: self._show_error(err))
            self.root.after(0, self._clear_busy)
            return
        if not hits:
            self.root.after(0, self._stop_loading)
            self.root.after(
                0, lambda: self._add_note("没有找到相关结果,换个说法试试。")
            )
            self.root.after(0, self._clear_busy)
            return
        self._generate(text, hits)  # 仍在后台线程,继续生成答案

    def _show_results(self, hits: list):
        """VLM 不可用时的降级:只展示检索到的引用卡片。

        顺带把这轮检索到的页记进历史。这一轮没有答案,但"引用过哪几页"是检索的
        结果、和答案不是一回事,不该跟着答案一起丢 —— 否则回头点开这条记录,
        侧栏写着「未完成」、引用卡片一张也不剩。
        """
        tiles = tiles_from(hits)
        if tiles:
            self._append({"kind": "citations", "tiles": tiles})
        else:
            self._add_note("没有可显示的截图。")
        self.history.update(
            self._hist_entry,
            hits=len(tiles),
            refs=refs_of([h for _b64, h in tiles]),
        )
        self._refresh_sidebar()

    def _generate(self, text: str, hits: list):
        """准备截图 → 调 VLM 流式生成 → 前端流式显示答案 + 引用卡片。运行于后台线程。"""
        if self._stop_event.is_set():
            # 检索期间用户就点了停止,别再开生成
            self.root.after(0, self._stop_loading)
            self.root.after(0, lambda: self._add_note("已停止。"))
            self.root.after(0, self._clear_busy)
            return

        images = prepare_images(hits)
        if not images:
            self.root.after(0, self._stop_loading)
            self.root.after(
                0,
                lambda: self._add_error(
                    "检索到了结果,但截图加载失败。以下为检索到的相关页面:"
                ),
            )
            self.root.after(0, lambda: self._show_results(hits))
            self.root.after(0, self._clear_busy)
            return

        cfg = _load_vlm_config()
        if not cfg["api_key"]:
            self.root.after(0, self._stop_loading)
            self.root.after(
                0,
                lambda: self._add_error(
                    "未找到 VLM API key。请设置环境变量 PIXELRAG_VLM_API_KEY,\n"
                    "或确保 ~/.claude/settings.json 里有 ANTHROPIC_AUTH_TOKEN。\n"
                    "以下为检索到的相关页面:"
                ),
            )
            self.root.after(0, lambda: self._show_results(hits))
            self.root.after(0, self._clear_busy)
            return

        self.root.after(0, lambda: self._set_loading_text("正在生成答案"))
        self._answer_ready = threading.Event()
        self.root.after(0, self._begin_answer)
        if not self._answer_ready.wait(timeout=5):
            self.root.after(0, self._clear_busy)
            return

        full = ""
        stopped = False
        failed = False
        try:
            self._gen = vlm_stream(images, text, cfg)
            for chunk in self._gen:
                if self._stop_event.is_set():
                    stopped = True
                    break
                full += chunk
                self.root.after(0, lambda t=full: self._set_answer(t))
        except Exception as e:
            if self._stop_event.is_set():
                stopped = True  # 主动中止引发的异常不算失败,别再弹一次错
            else:
                failed = True  # 已经报过错了,_finish_answer 别再补一条"没返回内容"
                err = str(e)
                self.root.after(
                    0, lambda m=err: self._add_error(f"生成失败:{m}", retry=True)
                )
                if full:
                    self.root.after(0, lambda t=full: self._set_answer(t))
        finally:
            # 显式 close 掉生成器,底层的 HTTP 流立刻断开,不用等 GC
            gen, self._gen = self._gen, None
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass
            self.root.after(0, lambda: self._finish_answer(images, stopped, failed))

    def _begin_answer(self):
        self._stop_loading()
        msg = {"kind": "answer", "text": "", "streaming": True}
        self.messages.append(msg)
        self._render_message(msg)
        self._answer_msg = msg
        self._scroll_bottom(True)
        self._answer_ready.set()

    def _set_answer(self, text: str):
        msg = self._answer_msg
        if msg is None:
            return
        msg["text"] = text
        if self._paint_job is None:
            self._paint_job = self.root.after(
                T.MOTION["stream_throttle"], self._paint_answer
            )

    def _paint_answer(self):
        self._paint_job = None
        msg, txt = self._answer_msg, self._answer_text
        if msg is None or txt is None:
            return
        text = msg.get("text", "")
        stick = self._at_bottom()
        try:
            self._set_answer_into(txt, (text + " ▌") if text else "…")
        except tk.TclError:
            return  # 主题切换把控件重建了,这一帧直接丢掉
        self._scroll_bottom(stick)

    def _finish_answer(self, images, stopped: bool = False, failed: bool = False):
        if self._paint_job:
            try:
                self.root.after_cancel(self._paint_job)
            except Exception:
                pass
            self._paint_job = None

        msg = self._answer_msg
        self._answer_msg = None
        self._answer_text = None

        if msg is not None:
            msg["streaming"] = False
            text = msg.get("text", "")
            if stopped:
                # 保留已经吐出来的部分,补一句话说明是用户主动停的
                msg["text"] = (
                    (text.rstrip() + "\n\n（已停止生成）")
                    if text.strip()
                    else "（已停止生成）"
                )
                self._redraw_answer(msg)
            elif not text.strip():
                # 一个字都没有:留个空气泡没意义,换成可重试的错误。
                # failed 时上面已经报过具体原因了,这里不再补一条重复的。
                self._drop_message(msg)
                if not failed:
                    self._add_error("模型没有返回任何内容。", retry=True)
            else:
                self._redraw_answer(msg)

        # 回填历史:命中页数 + 答案 + 引用页。存下引用页是为了以后点开这条历史时
        # 能把引用卡片一起还原 —— 缩略图从本地 tiles 读,不用再问一次检索服务。
        self.history.update(
            self._hist_entry,
            hits=len(images),
            answer=(msg or {}).get("text", ""),
            refs=refs_of([h for _b64, h in images]),
        )
        self._refresh_sidebar()

        if images:
            self._append({"kind": "citations", "tiles": list(images)})
        self._scroll_bottom(True)
        self._clear_busy()

    def _redraw_answer(self, msg: dict):
        """定稿重绘:分段代码块 + 复制按钮。"""
        widget = msg.get("_widget")
        if widget is not None:
            try:
                widget.destroy()
            except Exception:
                pass
        self._render_message(msg)

    def _drop_message(self, msg: dict):
        """把一条消息从数据模型和界面上一起撤掉。"""
        try:
            self.messages.remove(msg)
        except ValueError:
            pass
        widget = msg.get("_widget")
        if widget is not None:
            try:
                widget.destroy()
            except Exception:
                pass

    def _show_error(self, e: Exception):
        self._add_error(
            f"查询失败:{e}\n请确认检索服务已启动、索引文件完整。", retry=True
        )

    def _clear_busy(self):
        self._busy = False
        self._stop_hover = False
        self._show_stop(False)
        self._update_send_button()

    # ---------- 主题 ----------

    def _toggle_theme(self):
        T.set_mode(T.other_mode())
        W._clear_cache()
        self._apply_theme()

    def _apply_theme(self):
        """换主题 = 用令牌重放全部界面(tkinter 不支持热改样式)。"""
        self._rebuild_ui()

    def _rebuild_ui(self):
        """按当前令牌重放整个界面。换主题、收起/展开侧栏都走这里。

        tkinter 没有"只重画"这条路,一律销毁重建;输入到一半的内容和滚动位置
        在销毁前取出来,重建后放回去。
        """
        draft = self.input.get("1.0", "end-1c")
        state = self._view_state()  # 必须在销毁旧界面之前取

        for w in self.root.winfo_children():
            w.destroy()
        self._photos.clear()
        self._answer_text = None

        self.root.configure(bg=T.c("bg"))
        self._build_ui()
        self.input.insert("1.0", draft)
        self._render_all(view_state=state)
        self._set_status()
        self._update_send_button()

    # ---------- 打开图片 / PDF ----------

    def _open_hit(self, hit: dict, title: str):
        url = hit.get("url", "")
        if url and os.path.exists(url):
            os.startfile(url)
        else:
            self._add_note(f"找不到 PDF 文件:{title}")

    def _open_image(self, b64: str, title: str):
        try:
            img = decode_b64_image(b64)
        except Exception:
            return
        win = tk.Toplevel(self.root)
        win.title(title or "页面截图")
        win.configure(bg=T.c("bg"))
        max_w = int(win.winfo_screenwidth() * 0.85)
        max_h = int(win.winfo_screenheight() * 0.85)
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        photo = W.photo_of(img)
        self._photos.append(photo)
        lbl = tk.Label(win, image=photo, bg=T.c("bg"), bd=0)
        lbl.image = photo
        lbl.pack()
        # 焦点必须落在预览窗内,Escape 才会送到这里(否则被主窗吃掉)
        win.bind("<Escape>", lambda _e: win.destroy())
        win.transient(self.root)
        win.focus_set()

    # ---------- 服务 ----------

    def _ensure_serve(self):
        if check_health():
            self._serve_ready = True
            self.root.after(0, self._set_status)
            return
        self.root.after(0, self._set_status, "starting")
        start_serve_process()
        deadline = time.time() + 180
        while time.time() < deadline:
            time.sleep(2)
            if check_health():
                self._serve_ready = True
                self.root.after(0, self._set_status)
                return
        self.root.after(0, self._set_status, "failed")


def main():
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 高分屏下不再被位图拉伸
    except Exception:
        pass
    T.set_ui_scale(screen_scale())
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

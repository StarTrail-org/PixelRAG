"""PixelRAG 桌面客户端的绘制与交互原语。

tkinter 原生没有圆角、没有 hover 状态、文字也不可选中复制。本模块用 Pillow
离屏渲染补齐这些能力:

- ``rounded_photo``  : 抗锯齿圆角/描边矩形(带缓存,避免重复渲染与 PhotoImage 被 GC)
- ``render_label_box``: 把文本直接画进圆角框,返回整块图片(用户气泡 / 按钮)
- ``bind_hover``      : 正确的 hover 绑定(处理指针移入子控件时的假 Leave)
- ``bind_click_recursive``: 让整张卡片(含所有子控件)都可点击

只做"画"和"绑事件",不涉及任何业务逻辑。
"""

from __future__ import annotations

import re
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont, ImageTk

import pixelrag_theme as T

# ---------------------------------------------------------------------------
# 字体(PIL 侧需要真实字号像素值,而 tk 用点值)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _font_path(bold: bool) -> str | None:
    candidates = (
        (
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
        )
        if bold
        else (
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        )
    )
    import os

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@lru_cache(maxsize=64)
def pil_font(size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """取 PIL 字体;系统字体缺失时回退到位图字体(不会崩)。"""
    path = _font_path(bold)
    if not path:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(path, size_px)
    except Exception:
        return ImageFont.load_default()


def font_px(token: str) -> int:
    """把主题里的点值字号换算成像素,已含 FONT_SCALE。

    换算系数必须用 tk 自己的(实测 DPI),不能按 96dpi 写死 4/3 —— 在 150% 缩放的屏上
    那会让 Pillow 画的字比同一 token 的 tk 文字小 31%(用户气泡/按钮长期偏小就是这原因)。
    """
    return max(1, round(T.pt(token) * T.tk_scaling()))


def line_gap_px(token: str, bold: bool = False) -> int:
    """Pillow 侧的行距补偿:目标行盒(``T.line_box_px``)减去字体自然行高。

    Pillow 的 ``getmetrics()`` 给出的是字体自身的 asc+desc(tk 的 ``linespace`` 同理),
    单倍行高约等于字号的 1.29 倍;要凑到 1.6 倍,补的就是这个差值。行高是"目标 - 实测",
    所以换字体、改 FONT_SCALE、换 DPI 都不用回来改这里的数字。
    """
    font = pil_font(font_px(token), bold=bold or T.TYPE[token][1] == "bold")
    asc, desc = font.getmetrics()
    return max(0, T.line_box_px(token) - (asc + desc))


def tk_line_gap_px(font: tuple) -> int:
    """tk 侧的同一个补偿量(送给 Text 的 ``spacing2``)。

    tk 的字体元组只给点值,物理字号 = 点值 × ``tk_scaling()`` —— 拿它算目标行盒,
    再用 ``linespace`` 量字体自然行高,两边相减即可。这样即使传进来的是等宽字体
    (代码块)也算得对,不需要知道 token 名。
    """
    try:
        f = tkfont.Font(font=font)
        natural = int(f.metrics("linespace") or 0)
        size_px = abs(int(f.actual("size"))) * T.tk_scaling()
    except Exception:
        return 0
    target = round(size_px * T.LINE_HEIGHT)
    return max(0, target - natural)


def line_spacing(font: tuple) -> tuple[int, int, int]:
    """给 ``tk.Text``(控件级或 tag 级)用的 ``(spacing1, spacing2, spacing3)``。

    tk 会把 ``spacing2``(**段内折行**)对半分给折点两侧,于是同一个行高要求下,
    段首行只吃到下半份、段末行只吃到上半份,而"硬换行后面的那行"两个份都吃 ——
    按直觉写 spacing 会让整段行距忽紧忽松。只有
    ``spacing1 = ceil(gap/2)``、``spacing2 = gap``、``spacing3 = floor(gap/2)``
    这一组能让下面五种行盒**完全相等**(都等于 字号 × LINE_HEIGHT):

    - 单行段落、多行段落的段首行 / 中间行 / 段末行、硬换行(``\\n``)之后那行。

    实测记录与逐行断言见仓库外的 ``_uicheck/linecheck.py``。
    """
    gap = tk_line_gap_px(font)
    return -(-gap // 2), gap, gap // 2


# ---------------------------------------------------------------------------
# 文本换行(CJK 友好)
# ---------------------------------------------------------------------------

# 每个 token 要么是一个 CJK 字,要么是一段连续的非空白西文,要么是空白
_TOKEN_RE = re.compile(
    r"[\u2e80-\u9fff\u3000-\u303f\uff01-\uff60\uffe0-\uffe6]"
    r"|[^\s\u2e80-\u9fff\u3000-\u303f\uff01-\uff60\uffe0-\uffe6]+"
    r"|\s+"
)


def wrap_lines(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """按像素宽度贪心折行。西文按词断,CJK 按字断,保留显式换行。"""
    scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    out: list[str] = []
    for para in text.split("\n"):
        if not para:
            out.append("")
            continue
        cur = ""
        for tok in _TOKEN_RE.findall(para):
            cand = cur + tok
            if cur and scratch.textlength(cand, font=font) > max_w:
                out.append(cur.rstrip())
                cur = tok if tok.strip() else ""  # 行首不留空白
            else:
                cur = cand
        out.append(cur.rstrip())
    return out


# ---------------------------------------------------------------------------
# 圆角矩形
# ---------------------------------------------------------------------------

_photo_cache: dict[tuple, ImageTk.PhotoImage] = {}
_SUPERSAMPLE = 4  # 先放大再缩小 = 抗锯齿


def _clear_cache() -> None:
    """主题切换后调用,丢弃旧底色的缓存图。"""
    _photo_cache.clear()


def rounded_photo(
    w: int,
    h: int,
    radius: int,
    fill: str,
    bg: str,
    outline: str | None = None,
    outline_w: int = 1,
) -> ImageTk.PhotoImage:
    """抗锯齿圆角矩形。``bg`` 是"挖空"的角所透出的底色(必须是父容器底色)。"""
    w, h = max(1, int(w)), max(1, int(h))
    key = (w, h, radius, fill, bg, outline, outline_w)
    cached = _photo_cache.get(key)
    if cached is not None:
        return cached

    s = _SUPERSAMPLE
    img = Image.new("RGB", (w * s, h * s), bg)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [0, 0, w * s - 1, h * s - 1],
        radius=min(radius * s, w * s // 2, h * s // 2),
        fill=fill,
        outline=outline,
        width=outline_w * s if outline else 0,
    )
    img = img.resize((w, h), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    _photo_cache[key] = photo  # 持有引用,防止 GC 后图上白
    return photo


# ---------------------------------------------------------------------------
# 文本块 / 气泡 / 胶囊按钮
# ---------------------------------------------------------------------------


def render_label_box(
    text: str,
    font_token: str,
    text_fill: str,
    bg: str,
    box_fill: str | None = None,
    radius: int = 0,
    pad_x: int = 12,
    pad_y: int = 8,
    max_w: int | None = None,
    outline: str | None = None,
    align: str = "left",
    line_gap: int | None = None,
    bold: bool = False,
    min_w: int | None = None,
) -> Image.Image:
    """把文本渲染成一张图片。``max_w=None`` 表示不折行(按钮用)。

    ``line_gap=None`` 时按主题行高(``T.LINE_HEIGHT``)自动算,多行文本才有区别 ——
    单行时上下各减一次,最终高度与 line_gap 无关,所以按钮不会因此变胖。

    ``min_w`` 给图片一个最小宽度(图片比它窄时两侧留白,``align="center"`` 仍然居中):
    按钮在「发送 / 生成中」这类不同文案之间切换时宽度不跳,旁边的控件就不会被推来推去。
    """
    bold = bold or T.TYPE[font_token][1] == "bold"
    if line_gap is None:
        line_gap = line_gap_px(font_token, bold=bold)
    font = pil_font(font_px(font_token), bold=bold)
    scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    if max_w is None:
        lines = text.split("\n")
        text_w = int(max(scratch.textlength(ln, font=font) for ln in lines))
    else:
        lines = wrap_lines(text, font, max(40, max_w - 2 * pad_x))
        text_w = int(max(scratch.textlength(ln, font=font) for ln in lines))

    asc, desc = font.getmetrics()
    line_h = asc + desc + line_gap
    w = text_w + 2 * pad_x
    if min_w:
        w = max(w, int(min_w))
    h = line_h * len(lines) + 2 * pad_y - line_gap

    if box_fill is None:
        img = Image.new("RGB", (w, h), bg)
    else:
        # 直接在此图上画圆角(而不是复用 rounded_photo),避免缩放两次糊掉文字
        s = _SUPERSAMPLE
        img = Image.new("RGB", (w * s, h * s), bg)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle(
            [0, 0, w * s - 1, h * s - 1],
            radius=min(radius * s, w * s // 2, h * s // 2),
            fill=box_fill,
            outline=outline,
            width=1 * s if outline else 0,
        )
        img = img.resize((w, h), Image.LANCZOS)

    d = ImageDraw.Draw(img)
    y = pad_y
    for ln in lines:
        if align == "center":
            x = (w - scratch.textlength(ln, font=font)) / 2
        else:
            x = pad_x
        d.text((x, y), ln, font=font, fill=text_fill)
        y += line_h
    return img


def photo_of(img: Image.Image) -> ImageTk.PhotoImage:
    """PIL 图 → tk 图片。调用方必须自己持有返回值,否则会被 GC 成白块。"""
    return ImageTk.PhotoImage(img)


# ---------------------------------------------------------------------------
# 交互绑定
# ---------------------------------------------------------------------------


def bind_hover(
    widget,
    on_enter: Callable[[], None] | None = None,
    on_leave: Callable[[], None] | None = None,
) -> None:
    """正确的 hover:指针移入子控件时 tk 会误发 <Leave>,这里下一帧复查真实位置。"""
    state = {"inside": False}

    def _enter(_e=None):
        if state["inside"]:
            return
        state["inside"] = True
        if on_enter:
            on_enter()

    def _leave(_e=None):
        def check():
            try:
                x, y = widget.winfo_pointerxy()
                wx, wy = widget.winfo_rootx(), widget.winfo_rooty()
                inside = (
                    wx <= x < wx + widget.winfo_width()
                    and wy <= y < wy + widget.winfo_height()
                )
            except Exception:
                inside = False
            if not inside and state["inside"]:
                state["inside"] = False
                if on_leave:
                    on_leave()

        widget.after_idle(check)

    widget.bind("<Enter>", _enter, add="+")
    widget.bind("<Leave>", _leave, add="+")


def bind_click_recursive(widget, handler: Callable[[], None]) -> None:
    """让 widget 及其所有后代都可点击(卡片里的 Label 也要能开 PDF)。"""
    widget.bind("<Button-1>", lambda _e: handler(), add="+")
    for child in widget.winfo_children():
        bind_click_recursive(child, handler)


def set_cursor_recursive(widget, cursor: str = "hand2") -> None:
    try:
        widget.configure(cursor=cursor)
    except Exception:
        pass
    for child in widget.winfo_children():
        set_cursor_recursive(child, cursor)


def set_bg_recursive(widget, bg: str) -> None:
    """把整棵子树的底色换成 ``bg`` —— 列表行做 hover 高亮用。"""
    try:
        widget.configure(bg=bg)
    except Exception:
        pass
    for child in widget.winfo_children():
        set_bg_recursive(child, bg)


def make_focusable(
    widget,
    ring_bg: str,
    on_activate: Callable[[], None] | None = None,
    thickness: int | None = None,
) -> None:
    """让控件能被 Tab 聚焦、用 Enter/Space 激活,并带可见焦点环。

    焦点环用的是 tk 自带的 highlight 描边:失焦时填 ``ring_bg``(即控件所在
    容器的底色,视觉上不可见),因此**描边始终占位**,聚焦前后几何完全一致,
    不会引起布局跳动。聚焦时 tk 自动改用 ``highlightcolor``。

    ``ring_bg`` 必须传控件父容器的底色,否则失焦时会看到一圈异色边框。
    """
    t = thickness if thickness is not None else max(1, round(2 * T.UI_SCALE))
    try:
        widget.configure(
            takefocus=1,
            highlightthickness=t,
            highlightbackground=ring_bg,
            highlightcolor=T.c("focus"),
        )
    except tk.TclError:
        return
    if on_activate is not None:
        widget.bind("<Return>", lambda _e: on_activate(), add="+")
        widget.bind("<KP_Enter>", lambda _e: on_activate(), add="+")
        widget.bind("<space>", lambda _e: on_activate(), add="+")


# ---------------------------------------------------------------------------
# 可选中 / 可复制的只读文本
# ---------------------------------------------------------------------------

# 允许穿透的按键:光标移动 / 选中 / 复制 / 全选。其余按键一律吞掉(只读)。
_PASS_KEYS = frozenset(
    {
        "Left",
        "Right",
        "Up",
        "Down",
        "Home",
        "End",
        "Prior",
        "Next",
        "Shift_L",
        "Shift_R",
        "Control_L",
        "Control_R",
        "Tab",
        "ISO_Left_Tab",
        "Escape",
    }
)
_CTRL = 0x0004


def _readonly_guard(event) -> str | None:
    """只读 Text 的按键闸门:放行导航与复制,拦下一切编辑键。"""
    if event.keysym in _PASS_KEYS:
        return None
    # Ctrl+C / Ctrl+Insert 复制、Ctrl+A 全选放行;Ctrl+V/X 等编辑键继续拦
    if (event.state & _CTRL) and event.keysym.lower() in ("c", "insert", "a"):
        return None
    if event.keysym == "Insert" and not (event.state & _CTRL):
        return "break"
    return "break"


def selectable_text(
    parent,
    font: tuple,
    bg: str,
    fg: str,
    pad_x: int = 0,
    pad_y: int = 0,
    line_gap: int | None = None,
) -> tk.Text:
    """建一个只读但可选、可复制的正文 Text。

    tk.Label 无法选中文字,所以"输出内容要能复制"只能靠 Text 承载。这里不用
    ``state="disabled"``(禁用态在 Windows 上会吃掉一部分选择/复制交互),而是
    保持 normal 并在按键层拦截编辑,行为更接近网页上的只读文本。

    行距按主题行高(``T.LINE_HEIGHT``)铺开,``line_gap=None`` 时自动算
    (算法与理由见 ``line_spacing``)。单行段落的行盒 = linespace + spacing1 +
    spacing3 正好也是 ``fit_height`` 折算 ``-height`` 用的基准,所以行距一改控件
    高度自动跟上。段与段之间的空档由正文里的空行承担(空行本身也是一个"单行段落")。
    """
    sp1, sp2, sp3 = line_spacing(font) if line_gap is None else (0, line_gap, 0)
    txt = tk.Text(
        parent,
        bg=bg,
        fg=fg,
        font=font,
        relief="flat",
        bd=0,
        highlightthickness=0,
        wrap="word",
        height=1,
        padx=pad_x,
        pady=pad_y,
        spacing1=sp1,
        spacing2=sp2,
        spacing3=sp3,
        cursor="xterm",
        insertwidth=0,
        takefocus=1,
    )
    txt.bind("<Key>", _readonly_guard)
    return txt


def _base_line_px(txt: tk.Text) -> int:
    """控件基准行高(像素):字体行距 + 控件级的上下留白。

    tk 的 ``-height`` 就是按这个值折算的,不能只取字体行距 —— 否则算出来的行数
    会偏多。
    """
    ls = tkfont.Font(font=txt.cget("font")).metrics("linespace") or 1
    sp1 = int(txt.cget("spacing1") or 0)
    sp3 = int(txt.cget("spacing3") or 0)
    return int(ls) + sp1 + sp3


def _display_line_px(txt: tk.Text, index: str, base: int) -> int:
    """推算某个显示行的高度:取该行首字符上各 tag 里最高的那套字体 + 上下留白。

    tag 没写 spacing 时用控件级的值(和 tk 的解析规则一致)。
    """
    w1 = int(txt.cget("spacing1") or 0)
    w3 = int(txt.cget("spacing3") or 0)
    base_ls = base - w1 - w3  # 控件基准字体本身的行距
    h = base
    try:
        for tag in txt.tag_names(index):
            spec = txt.tag_cget(tag, "font")
            ls = tkfont.Font(font=spec).metrics("linespace") if spec else base_ls
            sp1 = txt.tag_cget(tag, "spacing1")
            sp3 = txt.tag_cget(tag, "spacing3")
            sp1 = int(sp1) if str(sp1).strip() else w1
            sp3 = int(sp3) if str(sp3).strip() else w3
            h = max(h, int(ls) + sp1 + sp3)
    except Exception:
        pass
    return h


def fit_height(txt: tk.Text, settle: bool = False) -> None:
    """把只读 Text 的高度收到刚好放下内容,免得内部出现第二条滚动条。

    ``settle=True`` 会先 ``update_idletasks()`` 逼出真实宽度;窗口刚建好或宽度
    变过之后必须传 True,否则按旧宽度算出来的行数会偏少。流式刷新时宽度不变,
    可以走 False 这条便宜路径。

    注意 ``-height`` 的单位是"控件基准字体的行数",而代码段换了字体、还带上下
    留白,一行比正文高 —— 只按显示行数设高,末尾几行会落到视口外被切掉(行数
    ≠ 像素高度)。所以这里按 **像素** 累加:已经排版出来的行直接量,还没排到的
    行(在视口下方)按该行的 tag 推算,最后向上取整换算回行数 —— 宁可多一行
    留白,也不切字。
    """
    try:
        if settle:
            txt.update_idletasks()
        n = max(1, int(txt.count("1.0", "end", "displaylines")[0]))
        base = _base_line_px(txt)
    except Exception:
        return

    # 视口外的行只能推算,推算值会比实际略高(末行没有换行符,矮一点),所以定稿
    # 时再量一遍:第二遍整块内容都已排版,能拿到精确值。流式刷新不做第二遍。
    for attempt in range(2 if settle else 1):
        total = 0
        for i in range(n):
            idx = f"1.0 + {i} displaylines"
            info = txt.dlineinfo(idx)
            total += info[3] if info is not None else _display_line_px(txt, idx, base)

        need = max(1, -(-total // base))  # 向上取整
        if int(txt.cget("height")) == need:
            break
        txt.configure(height=need)
        if attempt == 0 and settle:
            txt.update_idletasks()


def attach_text_menu(widget: tk.Text, root, editable: bool = False) -> None:
    """给 Text 挂上右键菜单。``editable=False`` 时只给复制/全选。"""
    menu = tk.Menu(root, tearoff=0, font=T.font("caption"))

    def add(label, accel, fn, state="normal"):
        menu.add_command(label=label, accelerator=accel, command=fn, state=state)

    add("复制", "Ctrl+C", lambda: widget.event_generate("<<Copy>>"))
    if editable:
        add("剪切", "Ctrl+X", lambda: widget.event_generate("<<Cut>>"))
        add("粘贴", "Ctrl+V", lambda: widget.event_generate("<<Paste>>"))
        menu.add_separator()
    else:
        menu.add_separator()

    def select_all():
        widget.tag_add("sel", "1.0", "end-1c")
        widget.mark_set("insert", "1.0")

    add("全选", "Ctrl+A", select_all)

    def popup(event):
        try:
            widget.focus_set()
        except Exception:
            pass
        # 没有选中内容时,"复制"没有意义 —— 置灰而不是点了没反应
        try:
            has_sel = bool(widget.tag_ranges("sel"))
        except Exception:
            has_sel = False
        menu.entryconfigure(0, state="normal" if has_sel else "disabled")
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    widget.bind("<Button-3>", popup, add="+")

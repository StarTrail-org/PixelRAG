"""PixelRAG 桌面客户端设计令牌 —— 对齐 DeepSeek 聊天界面。

这是全项目唯一的"样式表"。所有颜色 / 字号 / 间距 / 圆角 / 动效时长都必须
从这里取,业务代码里不允许再出现硬编码色值或魔法数字。

浅色(light)为默认主题,深色(dark)是等价反转的备选主题。
两个色板必须拥有完全相同的键,否则切换时会 KeyError。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 色板
# ---------------------------------------------------------------------------
PALETTES: dict[str, dict[str, str]] = {
    # 取色原则:正文与提示文字对底色的对比度需 ≥ 4.5:1(WCAG AA 普通文本)。
    # 括号里是对"文字落到自己常见底色上"的实测对比度,改动色值时请一并复核。
    # ---- 浅色:Notion 纸感(默认)----
    "light": {
        "bg": "#FFFFFF",  # 页面底色
        "surface": "#F9F9F9",  # 侧栏 / 顶栏 / 底栏(DeepSeek 冷灰)
        "card": "#FFFFFF",  # 卡片表面
        "card_hover": "#F5F6F8",  # 卡片 hover
        # 侧栏条目(DeepSeek 是浅灰圆角块,不是描边也不是蓝底):
        # 底色是 surface(#F9F9F9),所以这两个值都要比它明显深一档才看得出来。
        "nav_hover": "#F0F0F2",  # 条目 hover(11.9:1 上的文字仍清晰)
        "nav_active": "#E4E6EA",  # 当前打开的那一条(比 hover 再实一点)
        "border": "#E5E7EB",  # 常规分隔线
        "border_strong": "#D1D5DB",  # hover / focus 时的边框
        "text": "#1F2329",  # 正文(15.8:1 on bg)
        "text_muted": "#4B5563",  # 次要信息(7.6:1 on bg)
        "text_faint": "#6B7280",  # 极淡提示(4.8:1 on bg;再淡就跌破 4.5)
        # 主色拆两个键:填充/边框用原值,白底上的**文字**用 accent_text。
        # #4D6BFE 落白底只有 4.33:1,不够正文级的 4.5:1,所以文字另取 #3D5AE0(5.6:1)。
        "accent": "#4D6BFE",  # DeepSeek 蓝:填充 / 边框 / 焦点环
        "accent_text": "#3D5AE0",  # 白底上的强调色文字(5.6:1 on bg)
        "accent_hover": "#3D5AE0",  # 实心按钮 hover(白字落其上 5.6:1)
        "accent_soft": "#F0F3FF",  # 主色淡底(hover 态底,非正文底)
        # 实心按钮 = accent 底 + 白字 = 4.33:1,是全表唯一低于 4.5 的组合;
        # 这是 DeepSeek 自己的做法,按用户指定的原值保留。
        "on_accent": "#FFFFFF",  # 主色之上的文字
        "user_bubble": "#E8EEFF",  # 用户消息浅蓝块(DeepSeek 风格)
        "user_text": "#1F2329",  # 用户消息文字(13.6:1 on user_bubble)
        "code_bg": "#F6F7F9",  # 行内代码 / 代码块底
        "success": "#0F7B6C",  # 5.2:1 on bg
        "warning": "#8A6100",  # 5.5:1 on bg
        "danger": "#C0392B",  # 5.0:1 on danger_bg
        "danger_bg": "#FEF2F2",  # 错误提示底
        "input_bg": "#FFFFFF",
        "input_line": "#E5E7EB",  # 输入框 1px 边框(未聚焦)
        "focus": "#4D6BFE",  # 键盘焦点环
        "scrollbar": "#D1D5DB",
        "skeleton": "#F1F2F4",  # 加载骨架
    },
    # ---- 深色:照 DeepSeek 深色模式推的等价反转 ----
    "dark": {
        "bg": "#1A1A1A",
        "surface": "#1E1E1E",
        "card": "#242424",
        "card_hover": "#2A2A2A",
        "nav_hover": "#2C2C2E",
        "nav_active": "#38383C",
        "border": "#333333",
        "border_strong": "#454545",
        "text": "#E8E8E8",  # 14.2:1 on bg
        "text_muted": "#B4B4BB",
        "text_faint": "#A1A1AA",  # 6.8:1 on bg
        # 深色底上主色要提亮才够对比;同样拆成"填充用 / 文字用"两个键。
        "accent": "#7C93FF",  # 6.2:1 on bg
        "accent_text": "#8FA4FF",  # 7.4:1 on bg
        "accent_hover": "#8FA4FF",
        "accent_soft": "#232B45",
        "on_accent": "#1A1A1A",  # 深色底上主色偏亮,配深墨字才够对比(6.2:1)
        "user_bubble": "#253052",  # 深色下的用户气泡:深蓝而非深灰
        "user_text": "#E8E8E8",  # 10.2:1 on user_bubble
        "code_bg": "#242424",
        "success": "#4ADE80",
        "warning": "#FACC15",
        "danger": "#F87171",
        "danger_bg": "#3A1D1D",
        "input_bg": "#1E1E1E",
        "input_line": "#333333",
        "focus": "#7C93FF",
        "scrollbar": "#4A4A4A",
        "skeleton": "#2A2A2A",
    },
}

DEFAULT_MODE = "light"

_mode = DEFAULT_MODE


def mode() -> str:
    """当前主题名(light / dark)。"""
    return _mode


def set_mode(name: str) -> str:
    """切换主题并返回生效的主题名。"""
    global _mode
    if name not in PALETTES:
        raise ValueError(f"未知主题:{name!r}(可选:{sorted(PALETTES)})")
    _mode = name
    return _mode


def other_mode() -> str:
    return "dark" if _mode == "light" else "light"


def c(key: str) -> str:
    """按语义键取当前主题色。"""
    try:
        return PALETTES[_mode][key]
    except KeyError:
        raise KeyError(f"色板 {_mode!r} 中没有键 {key!r}") from None


# ---------------------------------------------------------------------------
# 排版
# ---------------------------------------------------------------------------
FONT_FAMILY = "Microsoft YaHei UI"  # 中文优先;缺失时 tk 会自行回退
FONT_MONO = "Consolas"

# token -> (字号 px, 字重)。**这里存的是 CSS 语义的像素值**,直接照 DeepSeek 的规格写,
# 肉眼可比。tk 只认点值,由 pt() 换算;Pillow 那边再从 pt() 换回像素,两边严格同源。
#
# DeepSeek 的层级:正文统一 16(提问与答案同号,只靠蓝色气泡区分),
# 标题 20 / 16 / 14 / 12,界面文字最小 12。
TYPE: dict[str, tuple[int, str]] = {
    "title": (20, "bold"),  # 顶栏品牌
    "h2": (16, "bold"),  # 区块标题 / 引用来源
    "body": (16, "normal"),  # 正文(欢迎语 / 提示 / 错误)
    "body_lg": (16, "normal"),  # 大号正文(保留键;DeepSeek 正文统一 16,暂与 body 同)
    "bubble": (16, "normal"),  # 用户提问气泡
    "answer": (16, "normal"),  # 答案正文
    "input": (16, "normal"),  # 输入框
    "code": (14, "normal"),  # 代码块(等宽;字号走 font_mono,行高同样受 LINE_HEIGHT 管)
    "side": (14, "normal"),  # 侧栏条目
    "label": (14, "bold"),  # 卡片标题
    "button": (14, "bold"),  # 按钮
    "caption": (12, "normal"),  # 卡片元信息
    "micro": (12, "normal"),  # 角标
}


# 全局字号倍率:**嫌字小就调大这一个数**。1.0 = 上面那套 DeepSeek 原值。
# tk 绘制与 Pillow 离屏绘制都从这里换算,保证两边的字一样大。
FONT_SCALE = 1.0

# 96dpi 下 1px = 0.75pt(CSS 的 px 与物理点的换算基准)。
PX_PER_PT = 0.75


def pt(token: str) -> int:
    """取某个 token 的点值(tk 字体元组只接受整数,所以在这里取整)。"""
    return max(1, round(TYPE[token][0] * PX_PER_PT * FONT_SCALE))


def font(token: str, weight: str | None = None) -> tuple:
    """取 tk 字体元组,如 ('Microsoft YaHei UI', 12, 'normal')。

    ``weight`` 可以覆盖 token 自带的那一档(取值同 tk:``normal`` / ``bold`` /
    ``italic`` / ``bold italic``)。正文里临时加粗(答案里的小标题)用它 ——
    字号与行高仍然归 token 管,免得为了一个粗体另起一套数字。
    """
    return (FONT_FAMILY, pt(token), weight or TYPE[token][1])


def font_mono(size: int = 14) -> tuple:
    """等宽字体(代码块)。``size`` 是设计像素值。"""
    return (FONT_MONO, max(1, round(size * PX_PER_PT * FONT_SCALE)), "normal")


# 行高倍数:DeepSeek 正文 16px / 行高 1.6。tk 与 Pillow 两边都从这里取,所以屏幕上
# 的正文和离屏画出来的气泡行距完全一致(改行距只动这一个数)。
LINE_HEIGHT = 1.6


def px_size(token: str) -> int:
    """该 token 的字号,设计像素(含 FONT_SCALE)。"""
    return max(1, round(TYPE[token][0] * FONT_SCALE))


def line_box_px(token: str) -> int:
    """该 token 一行该占多高(物理像素)= 字号 × 行高 × 屏幕缩放。

    注意这里含 ``UI_SCALE``:字号走 tk 的 pt→px 换算后实际就是放大了 UI_SCALE 倍,
    行盒必须用同一把尺子,否则高分屏上行距会显得比字紧。
    """
    return max(1, round(px_size(token) * LINE_HEIGHT * UI_SCALE))


# tk 把点值换算成像素的系数。**必须实测,不能按 96dpi 猜成 4/3** —— 本机是 150% 缩放
# (tk scaling ≈ 1.93),猜 4/3 会让 Pillow 画出来的字比 tk 那边小 31%,也就是用户气泡
# 和所有按钮的字一直偏小。这里问一次 tk 再缓存,调用方无需关心。
_TK_SCALING: float | None = None


def tk_scaling() -> float:
    """tk 的点值→像素系数(需要已存在 Tk 根窗口;没有就回退到 96dpi)。"""
    global _TK_SCALING
    if _TK_SCALING is None:
        try:
            import tkinter

            root = tkinter._default_root
            _TK_SCALING = (
                float(root.winfo_fpixels("1i")) / 72 if root else 1 / PX_PER_PT
            )
        except Exception:
            _TK_SCALING = 1 / PX_PER_PT  # 拿不到就退回 96dpi,至少不比以前差
    return _TK_SCALING


# ---------------------------------------------------------------------------
# 间距(4px 基准) / 圆角 / 动效
# ---------------------------------------------------------------------------
_BASE_SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "2xl": 32, "msg": 24}
_BASE_RADIUS = {"sm": 6, "md": 8, "lg": 12}
# 胶囊(整圆端头):给一个"远大于任何按钮高度"的值,绘制时会被自动夹到高度的一半
# (见 widgets.render_label_box),于是按钮高度随字号 / DPI 变化时端头依然是半圆,
# 调用方不需要先知道按钮多高。DeepSeek 的发送/停止按钮都是这个形状。
RADIUS_PILL = 999
# 滚动条宽度(DeepSeek 是细条、无可见槽)。tk 只能给像素值,所以跟着 UI_SCALE 走。
_BASE_SCROLLBAR_W = 8
SCROLLBAR_W = _BASE_SCROLLBAR_W
# 发送按钮的最小宽度(设计像素):它在「发送 / 生成中」两种文案间切换,给个下限
# 让它宽度不变,不然输入框里的文字会跟着一伸一缩。
_BASE_SEND_MIN_W = 72
SEND_MIN_W = _BASE_SEND_MIN_W
# 组件级尺寸(设计像素):不在 4px 栅格上、但 DeepSeek 有明确规格的那几个。
# 单列出来是为了不被"顺手对齐到栅格"改掉 —— 改了就不是 DeepSeek 的比例了。
_BASE_BUBBLE_PAD = (14, 10)  # 用户气泡内边距 (横向, 纵向)
# 消息列的宽度上限(DeepSeek 是 768)。窗口再宽也不跟着变宽 —— 富余的宽度均分成
# 两侧留白,正文始终是一条居中的窄栏,长行才不会横贯整屏。
_BASE_CONTENT_MAX_WIDTH = 768

# 下面几个是"逻辑值 × UI_SCALE"的结果;启动时由 set_ui_scale 按屏幕 DPI 重算,
# 以便在高分屏上字变大的同时,内边距/圆角/行宽等比跟上,不会显得局促。
SPACE: dict[str, int] = dict(_BASE_SPACE)
RADIUS: dict[str, int] = dict(_BASE_RADIUS)
BUBBLE_PAD: tuple[int, int] = _BASE_BUBBLE_PAD
CONTENT_MAX_WIDTH = _BASE_CONTENT_MAX_WIDTH
UI_SCALE = 1.0


def set_ui_scale(factor: float) -> float:
    """按屏幕缩放比例重算所有像素级令牌(0.75x ~ 2.0x)。"""
    global UI_SCALE, CONTENT_MAX_WIDTH, BUBBLE_PAD, SCROLLBAR_W, SEND_MIN_W
    UI_SCALE = max(0.75, min(2.0, float(factor)))
    for k, v in _BASE_SPACE.items():
        SPACE[k] = max(1, round(v * UI_SCALE))
    for k, v in _BASE_RADIUS.items():
        RADIUS[k] = max(2, round(v * UI_SCALE))
    BUBBLE_PAD = tuple(max(1, round(v * UI_SCALE)) for v in _BASE_BUBBLE_PAD)
    CONTENT_MAX_WIDTH = round(_BASE_CONTENT_MAX_WIDTH * UI_SCALE)
    SCROLLBAR_W = max(4, round(_BASE_SCROLLBAR_W * UI_SCALE))
    SEND_MIN_W = max(24, round(_BASE_SEND_MIN_W * UI_SCALE))
    return UI_SCALE


# 动效时长(毫秒)。tkinter 无 CSS,统一用 root.after 驱动。
MOTION: dict[str, int] = {
    "fast": 120,
    "base": 180,
    "stream_throttle": 60,  # 流式重绘节流间隔
    "resize_debounce": 120,  # 窗口尺寸变化防抖
}

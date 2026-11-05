#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PixelRAG 知识库中文查询脚本(交互式)。

用法:先双击 start_serve.bat 启动服务,再双击 query.bat(或运行本脚本)。
输入中文问题,返回最相关的论文页面;输入编号可直接打开该页截图,
输入 p+编号 打开对应论文的 PDF。
"""
import json
import os
import sys
import urllib.request

HOST = "localhost"
PORT = 30001
N_DOCS = 5

_ROOT = os.path.dirname(os.path.abspath(__file__))
TILES_DIR = os.path.join(_ROOT, "knowledge_index", "tiles")


def _safe(s: str) -> str:
    """把控制台无法编码的字符替换掉,避免 UnicodeEncodeError(如 ∕、₃)。"""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        s.encode(enc)
        return s
    except UnicodeEncodeError:
        return s.encode(enc, errors="replace").decode(enc)


def search(text: str, n_docs: int = N_DOCS) -> dict:
    payload = json.dumps(
        {"queries": [{"text": text}], "n_docs": n_docs}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))


def title_of(url: str) -> str:
    name = os.path.basename(url.replace("\\", "/"))
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name


def page_image_path(article_id: int, tile_index: int) -> str:
    d = os.path.join(TILES_DIR, f"{article_id}.png.tiles")
    for ext in ("jpg", "jpeg", "png"):
        p = os.path.join(d, f"tile_{tile_index:04d}.{ext}")
        if os.path.exists(p):
            return p
    return ""


def main() -> None:
    print("=" * 62)
    print("PixelRAG 知识库检索")
    print("输入中文问题后回车查询;直接回车退出。")
    print("=" * 62)

    while True:
        try:
            text = input("\n查询 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not text:
            print("再见。")
            break

        print(f"正在检索「{_safe(text)}」...(约 1 分钟)")
        try:
            resp = search(text)
        except Exception as e:
            print(f"查询失败:{e}")
            print("请确认服务已启动(双击 start_serve.bat)。")
            continue

        hits = resp["results"][0]["hits"]
        if not hits:
            print("没有找到相关结果,换个说法试试。")
            continue

        print(f"\n最相关的 {len(hits)} 个页面:")
        for i, h in enumerate(hits, 1):
            t = title_of(h.get("url", ""))
            score = h.get("score", 0)
            ti = h.get("tile_index", 0)
            print(f"  [{i}] {_safe(t)}  (tile_{ti:04d}, 相似度 {score:.3f})")

        print("\n输入编号打开该页截图;输入 p+编号 打开论文 PDF;直接回车继续:")
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if not choice:
            continue

        open_pdf = choice.startswith("p")
        num = choice[1:] if open_pdf else choice
        try:
            idx = int(num) - 1
            h = hits[idx]
        except (ValueError, IndexError):
            print("编号无效。")
            continue

        if open_pdf:
            pdf = h.get("url", "")
            if pdf and os.path.exists(pdf):
                os.startfile(pdf)
                print(f"已打开论文:{_safe(title_of(pdf))}")
            else:
                print("找不到 PDF 文件。")
        else:
            img = page_image_path(h["article_id"], h["tile_index"])
            if img:
                os.startfile(img)
                print(f"已打开页面截图:{_safe(os.path.basename(img))}")
            else:
                print("找不到该页截图,可能未渲染。")


if __name__ == "__main__":
    main()

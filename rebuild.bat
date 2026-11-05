@echo off
cd /d "%~dp0"
setlocal

rem ---------------------------------------------------------------
rem PixelRAG index rebuild.
rem Run this after dropping new PDFs into papers\.
rem   step 1  rebuild the index
rem   step 2  stop the old search server.  DO NOT SKIP: serve reads
rem           the index into memory once at startup and never reloads
rem           it, so without a restart it keeps answering from the
rem           stale index and the rebuild is wasted.
rem   step 3  start a fresh server
rem NOTE: this file is saved as GBK (cp936), NOT UTF-8, and has no
rem       chcp line.  Under chcp 65001 cmd's batch reader desyncs on
rem       multi-byte characters and starts executing echo lines as
rem       commands.  Keep rem comments ASCII; edit this file as GBK.
rem ---------------------------------------------------------------

set "NPAPERS=0"
for %%f in ("papers\*.pdf") do set /a NPAPERS+=1

echo ============================================================
echo   PixelRAG 索引重建
echo ============================================================
echo.
echo   论文目录 papers\ 有 %NPAPERS% 个 PDF
echo   索引输出 knowledge_index\
echo.
echo   注意: 嵌入阶段是全量重跑,不会因为只加了几篇就少跑。
echo         当前规模在 CPU 上约需 20 小时以上。
echo         中途关掉窗口没关系,重跑时从断点续跑,不会从头来。
echo.
pause

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

echo.
echo ------------------------------------------------------------
echo [1/3] 重建索引(这一步最久,可以放着不管)...
echo ------------------------------------------------------------
.venv\Scripts\pixelrag.exe index build -c pixelrag.yaml
if errorlevel 1 goto :failed

echo.
echo ------------------------------------------------------------
echo [2/3] 停掉旧的检索服务...
echo ------------------------------------------------------------
call :stop_serve
rem 2s wait for the port to free up.  ping, not timeout: timeout wants a
rem console on stdin and errors out when the script runs headless.
ping -n 3 127.0.0.1 >nul

echo.
echo ------------------------------------------------------------
echo [3/3] 启动新的检索服务(加载模型约需 1 分钟)...
echo ------------------------------------------------------------
start "PixelRAG serve" cmd /k ".venv\Scripts\pixelrag.exe serve --index-dir knowledge_index --tiles-dir knowledge_index/tiles --articles-json knowledge_index/articles.json --device cpu --port 30001"

echo.
echo ============================================================
echo   完成。等约 1 分钟后双击 PixelRAG客户端.bat 查询新索引。
echo   新开那个窗口就是检索服务,别关它。
echo ============================================================
pause
exit /b 0


:stop_serve
rem Kill whatever listens on 30001 (the old serve).  Skip if nothing does.
set "KILLED="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":30001" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%p >nul 2>&1
    if not errorlevel 1 (
        echo   已结束旧服务 PID %%p
        set "KILLED=1"
    )
)
if not defined KILLED echo   (端口 30001 上没有在跑的服务,跳过)
exit /b 0


:failed
echo.
echo ============================================================
echo   [X] 重建失败,索引没有更新(旧索引仍是完好的)。
echo       往上翻能看到卡在哪一步;修好后重跑本脚本即可,
echo       已经嵌好的部分会从断点续跑,不会白跑。
echo ============================================================
pause
exit /b 1

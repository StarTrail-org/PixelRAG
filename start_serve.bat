@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 若服务已在运行则不再重复启动
curl -s --max-time 2 http://localhost:30001/health >nul 2>&1
if %errorlevel%==0 (
    echo 服务已在运行(端口 30001),无需重复启动。
    echo 直接双击 query.bat 开始查询即可。
    pause
    exit /b 0
)

echo 正在启动 PixelRAG 检索服务,加载模型约需 1 分钟...
echo 看到 "Uvicorn running on http://0.0.0.0:30001" 即表示启动成功。
echo.
echo 提示:请保持本窗口开启,关闭本窗口即停止服务。
echo.

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
.venv\Scripts\pixelrag.exe serve --index-dir knowledge_index --tiles-dir knowledge_index/tiles --articles-json knowledge_index/articles.json --device cpu --port 30001

pause

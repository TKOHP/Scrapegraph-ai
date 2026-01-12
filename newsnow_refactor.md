# NewsNow 改造计划 - AIpaper2 展示前端

本文件记录将 `newsnow` 项目改造为 `AIpaper2` 数据库前端的方案与步骤。

## 1. 改造目标

*   **前端**: 保留 `newsnow` 的卡片式布局与交互。
*   **后端**: 移除原有 Node.js 后端，替换为 Python (FastAPI) 后端，直接读取 `AIpaper2` 的 SQLite 数据库。
*   **功能**: 按“主题”展示论文，卡片显示题目、日期、URL链接、PDF链接、总结链接。

## 2. 后端改造方案 (AIpaper2)

我们需要在 `AIpaper2` 目录下创建一个轻量级的 API 服务。

*   **文件**: `AIpaper2/api_server.py`
*   **技术栈**: FastAPI, Uvicorn, SQLite
*   **接口**:
    *   `GET /api/subjects`: 获取所有主题列表（用于生成列）。
    *   `GET /api/papers`: 获取论文列表，支持 `subject` 参数过滤。

## 3. 前端改造方案 (newsnow)

需要修改 `newsnow` 以适配新的 API 和数据结构。

### 3.1 配置代理
修改 `vite.config.ts`，将 `/api` 开头的请求代理到 Python 后端 (默认端口 8000)。

### 3.2 数据类型定义
修改 `shared/types.ts` 中的 `NewsItem` 接口，增加 `pdfLink`, `overviewLink` 等字段。

### 3.3 数据获取逻辑
修改 `src/hooks/query.ts`，不再请求 `/s?id=...`，而是请求 `/api/papers?subject=...`。

### 3.4 界面展示
修改 `src/components/column/card.tsx`：
*   移除原有的来源图标逻辑。
*   增加“PDF”、“总结”等链接按钮。
*   适配新的字段显示。

### 3.5 移除冗余功能
*   隐藏/移除登录、设置、MCP 服务相关入口。
*   简化 `pre-sources.ts`，仅保留与 `AIpaper2` 相关的源定义。

## 4. 运行说明

### 4.1 启动后端
在 `Scrapegraph-ai` 根目录下运行：
```bash
# 确保安装了依赖
pip install fastapi uvicorn

# 启动服务
python AIpaper2/api_server.py
```
*注：后端服务默认运行在 8001 端口*

### 4.2 启动前端
在 `newsnow` 目录下运行：
```bash
pnpm dev
```
前端访问地址: `http://localhost:5173`

# 微博舆情分析平台

基于 Streamlit 的中文社交媒体情绪分析作业项目，支持 CSV 评论分析、微博话题采集、SnowNLP/规则增强分析、词云以及 DeepSeek 舆情报告。

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt
cp .env.example .env
streamlit run app.py
```

CSV 文件必须包含 `评论内容` 列。

## 从 GitHub 部署

推荐使用 [Streamlit Community Cloud](https://share.streamlit.io/)：

1. 在 GitHub 创建仓库并推送本项目。
2. 在 Streamlit Community Cloud 选择该仓库、`main` 分支和 `app.py`。
3. Python 版本选择 `3.12`。
4. 在 Advanced settings → Secrets 中配置：

```toml
AI_PROVIDER = "deepseek"
DEEPSEEK_API_KEY = "your-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
CRAWLER_HEADLESS = "true"
SENTIMENT_GPU_ENABLED = "false"
```

5. 点击 Deploy。

云端免费实例默认安装轻量依赖，稳定支持 SnowNLP 与 Hybrid。Paddle/BERT 保留在本地完整依赖中。微博实时采集依赖登录 Cookie 和平台可见窗口，答辩时建议以示例 CSV 作为稳定演示路径。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 数据口径

平台明确区分微博标称评论数、实际抓取数和唯一评论文本数。低覆盖率任务会标记为探索性结论，AI 报告不得将标称评论数误写为实际分析样本。

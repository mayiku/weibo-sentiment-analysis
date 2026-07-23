"""
AI Agent 模块 — 多 Provider 驱动的舆情分析报告生成

用法:
    from src.ai_agent import create_ai_client, ReportGenerator, build_analysis_prompt

    # 使用默认 provider
    client = create_ai_client()
    prompt = build_analysis_prompt(topic, stats, posts, keywords, samples)
    report = client.chat(prompt)

    # 或直接使用 ReportGenerator
    gen = ReportGenerator()
    report = gen.generate(topic, stats, df, posts, keywords)
"""

from src.ai_agent.ai_provider import create_ai_client, AIProvider, SiliconFlowClient, DeepSeekClientWrapper
from src.ai_agent.deepseek_client import DeepSeekClient, LLMClient
from src.ai_agent.prompts import build_analysis_prompt, SYSTEM_PROMPT
from src.ai_agent.report_generator import ReportGenerator, generate_report

__all__ = [
    'create_ai_client', 'AIProvider', 'SiliconFlowClient', 'DeepSeekClientWrapper',
    'DeepSeekClient', 'LLMClient',
    'build_analysis_prompt', 'SYSTEM_PROMPT',
    'ReportGenerator', 'generate_report',
]

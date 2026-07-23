"""
用户体验优化模块 - 专门处理用户交互和错误提示
"""
import streamlit as st
import time
from typing import Callable, Any
from functools import wraps


def user_friendly_error_handler(func: Callable) -> Callable:
    """
    用户友好的错误处理装饰器
    将技术性错误转换为用户容易理解的消息
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()

            # 常见错误映射
            error_messages = {
                'timeout': "请求超时，请检查网络连接或稍后重试",
                'chromedriver': "浏览器驱动问题，请运行 `pip install webdriver-manager`",
                'login': "登录失败，请检查浏览器是否正常打开",
                'cookie': "登录信息过期，请重新扫码登录",
                'connection': "网络连接问题，请检查网络设置",
                'weibo': "微博服务暂时不可用，请稍后重试",
                'element not found': "页面结构变化，系统需要更新",
            }

            # 匹配错误类型
            user_msg = "系统遇到问题，请重试或联系技术支持"
            for key, msg in error_messages.items():
                if key in error_msg:
                    user_msg = msg
                    break

            # 在 Streamlit 中显示友好的错误
            if 'st' in globals():
                st.error(f"[FAILED] {user_msg}")
                st.caption(f"技术细节: {str(e)[:100]}...")

            # 重新抛出错误以供日志记录
            raise RuntimeError(f"{user_msg} (原始错误: {e})")

    return wrapper


def progress_tracker(total_steps: int):
    """
    进度跟踪器 - 优雅的进度显示
    """
    progress_bar = st.progress(0)
    status_container = st.empty()

    def update_step(current_step: int, message: str = ""):
        """更新进度"""
        percentage = int((current_step / total_steps) * 100)
        progress_bar.progress(percentage)

        step_display = f"[{current_step}/{total_steps}]"
        if message:
            status_info = f"{step_display} {message}"
        else:
            status_info = f"{step_display} 处理中..."

        status_container.info(status_info)

    return update_step


def login_status_updater():
    """
    登录状态实时更新器
    """
    status_messages = [
        "正在检查登录状态... 🕵️",
        "打开微博登录页面... 🔗",
        "请扫码登录微博... 📱",
        "检测登录状态中... [SEARCH]",
        "登录成功！开始爬取... [OK]",
    ]

    def update_status(step: int, custom_msg: str = None):
        msg = custom_msg if custom_msg else status_messages[min(step, len(status_messages)-1)]

        # 创建或更新状态显示
        if not hasattr(update_status, 'status_container'):
            update_status.status_container = st.empty()

        emoji = "⏳" if step < 4 else "[OK]"
        update_status.status_container.info(f"{emoji} {msg}")

    return update_status


def retry_with_notification(max_retries: int = 3, delay: float = 5.0):
    """
    带用户通知的重试装饰器
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise e

                    # 通知用户重试
                    if 'st' in globals():
                        st.warning(f"[WARNING] 操作失败，第 {retry_count} 次重试... ({delay}s后)")

                    time.sleep(delay)

            raise RuntimeError(f"操作失败，已达到最大重试次数 ({max_retries})")

        return wrapper

    return decorator


def format_elapsed_time(seconds: float) -> str:
    """将秒数格式化为友好的时间显示"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分"


def show_system_status():
    """显示系统状态面板"""
    from config import (
        CHROMEDRIVER_PATH, DATABASE_PATH,
        find_chinese_font, get_all_tasks
    )

    st.subheader("🔧 系统状态")

    status_cols = st.columns(4)

    with status_cols[0]:
        # ChromeDriver 状态
        driver_ok = CHROMEDRIVER_PATH.exists()
        status = "[OK] 正常" if driver_ok else "[FAILED] 缺失"
        st.metric("ChromeDriver", status)

    with status_cols[1]:
        # 数据库状态
        db_ok = DATABASE_PATH.exists()
        status = "[OK] 就绪" if db_ok else "[FAILED] 未初始化"
        st.metric("数据库", status)

    with status_cols[2]:
        # 中文字体
        font_ok = find_chinese_font() is not None
        status = "[OK] 可用" if font_ok else "[WARNING] 默认字体"
        st.metric("字体支持", status)

    with status_cols[3]:
        # 历史任务
        tasks_count = len(get_all_tasks() or [])
        st.metric("历史任务", f"{tasks_count} 条")


# 常用错误检查函数
@user_friendly_error_handler
def check_browser_availability() -> bool:
    """检查浏览器是否可用"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        # 尝试创建浏览器实例
        driver = webdriver.Chrome(options=options)
        driver.quit()
        return True
    except Exception as e:
        return False


@user_friendly_error_handler
def check_network_connectivity() -> bool:
    """检查网络连通性"""
    import requests

    test_urls = [
        'https://weibo.com',
        'https://www.baidu.com',
        'https://s.weibo.com'
    ]

    for url in test_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return True
        except:
            continue

    return False
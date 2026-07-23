"""
优化的WebDriver管理模块 - 提供更好的用户体验和错误处理
"""
import time
import sys
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException, SessionNotCreatedException

from config import CHROMEDRIVER_PATH, CRAWLER_HEADLESS
from src.logger import get_logger

log = get_logger(__name__)


def create_user_friendly_driver(headless: bool = None, max_retries: int = 3) -> webdriver.Chrome:
    """
    创建WebDriver - 用户友好的版本，包含详细的错误处理和友好的消息
    """
    if headless is None:
        headless = CRAWLER_HEADLESS

    for attempt in range(max_retries):
        try:
            driver = _try_create_driver_attempts(headless, attempt + 1)
            if driver:
                return driver
        except Exception as e:
            log.warning("WebDriver创建尝试 %d/%d 失败: %s", attempt + 1, max_retries, e)

            if attempt == max_retries - 1:
                # 最后一次尝试失败，提供详细的错误信息
                _show_user_friendly_error(e)
                raise

            # 重试前等待
            time.sleep(2)

    raise RuntimeError("无法创建WebDriver")


def _try_create_driver_attempts(headless: bool, attempt: int) -> Optional[webdriver.Chrome]:
    """
    尝试不同的策略创建WebDriver
    """
    strategies = [
        ("webdriver-manager", _create_with_webdriver_manager),
        ("手动路径", _create_with_manual_path),
        ("系统路径", _create_with_system_path),
    ]

    for strategy_name, strategy_func in strategies:
        try:
            log.info("尝试策略 [%d/%d]: %s", attempt, len(strategies), strategy_name)
            driver = strategy_func(headless)
            if driver:
                log.info("[OK] 策略成功: %s", strategy_name)
                _apply_enhanced_stealth(driver)
                return driver
        except Exception as e:
            log.debug("策略 %s 失败: %s", strategy_name, e)
            continue

    return None


def _create_with_webdriver_manager(headless: bool) -> webdriver.Chrome:
    """使用webdriver-manager创建WebDriver"""
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service as ChromeService

        driver_path = ChromeDriverManager().install()
        log.info("[OK] WebDriver路径: %s", driver_path)

        options = _create_chrome_options(headless)
        service = ChromeService(driver_path)
        return webdriver.Chrome(service=service, options=options)

    except ImportError:
        log.info("webdriver-manager 未安装")
        raise
    except Exception as e:
        log.warning("webdriver-manager 错误: %s", e)
        raise


def _create_with_manual_path(headless: bool) -> webdriver.Chrome:
    """使用手动配置路径创建WebDriver"""
    if not CHROMEDRIVER_PATH.exists():
        log.info("手动路径不存在: %s", CHROMEDRIVER_PATH)
        raise FileNotFoundError("ChromeDriver路径不存在")

    options = _create_chrome_options(headless)
    service = Service(str(CHROMEDRIVER_PATH))

    try:
        return webdriver.Chrome(service=service, options=options)
    except SessionNotCreatedException as e:
        if "This version of ChromeDriver only supports Chrome version" in str(e):
            log.error("Chrome版本不匹配")
            raise RuntimeError("Chrome浏览器版本与驱动不匹配，请更新Chrome或ChromeDriver")
        raise


def _create_with_system_path(headless: bool) -> webdriver.Chrome:
    """使用系统PATH中的WebDriver"""
    options = _create_chrome_options(headless)

    try:
        return webdriver.Chrome(options=options)
    except WebDriverException as e:
        if "chromedriver" in str(e).lower() and "path" in str(e).lower():
            log.info("系统PATH中未找到ChromeDriver")
            raise
        raise


def _create_chrome_options(headless: bool) -> Options:
    """创建Chrome选项配置"""
    options = Options()

    # 基础配置
    options.page_load_strategy = 'eager'
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # 用户代理
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )

    # 无头模式
    if headless:
        options.add_argument('--headless=new')
    else:
        # 非无头模式的优化
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-popup-blocking')

    return options


def _apply_enhanced_stealth(driver: webdriver.Chrome):
    """增强的反检测措施"""
    stealth_scripts = [
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})",
        "Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})",
        "Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en']})",
    ]

    for script in stealth_scripts:
        try:
            driver.execute_script(script)
        except Exception:
            pass

    # 设置合理的超时时间
    driver.set_page_load_timeout(60)
    driver.set_script_timeout(30)
    driver.implicitly_wait(10)


def _show_user_friendly_error(error: Exception):
    """显示用户友好的错误信息"""
    error_msg = str(error).lower()

    # 错误映射字典
    error_solutions = {
        "chromedriver": "请安装ChromeDriver: pip install webdriver-manager",
        "version": "Chrome版本不匹配，请更新浏览器或驱动",
        "path": "检查ChromeDriver路径配置",
        "permission": "权限问题，尝试以管理员权限运行",
        "network": "网络连接问题，检查网络设置",
        "chrome": "Chrome浏览器未安装或路径错误",
    }

    # 查找匹配的错误类型
    solution = "请检查系统环境和依赖项"
    for key, msg in error_solutions.items():
        if key in error_msg:
            solution = msg
            break

    log.error("WebDriver创建失败: %s", error)
    log.error("解决方案: %s", solution)

    # 如果可能，在Streamlit中显示错误
    if 'streamlit' in sys.modules:
        import streamlit as st
        st.error(f"[FAILED] 浏览器启动失败")
        st.error(f"问题: {str(error)[:100]}...")
        st.info(f"[INFO] 解决方案: {solution}")


def verify_webdriver_environment() -> dict:
    """
    验证WebDriver环境，返回详细的诊断信息
    """
    result = {
        'chrome_installed': False,
        'webdriver_available': False,
        'manual_path_exists': False,
        'webdriver_manager_installed': False,
        'permission_ok': False,
        'diagnosis': [],
    }

    # 检查Chrome安装
    try:
        import subprocess
        subprocess.run(['chrome', '--version'], capture_output=True)
        result['chrome_installed'] = True
        result['diagnosis'].append("[OK] Chrome浏览器已安装")
    except:
        result['diagnosis'].append("[FAILED] Chrome浏览器未安装或不在PATH中")

    # 检查手动路径
    if CHROMEDRIVER_PATH.exists():
        result['manual_path_exists'] = True
        result['diagnosis'].append("[OK] 手动ChromeDriver路径存在")
    else:
        result['diagnosis'].append("[WARNING] 手动ChromeDriver路径不存在")

    # 检查webdriver-manager
    try:
        import webdriver_manager
        result['webdriver_manager_installed'] = True
        result['diagnosis'].append("[OK] webdriver-manager已安装")
    except ImportError:
        result['diagnosis'].append("[WARNING] webdriver-manager未安装")

    # 测试创建WebDriver
    try:
        driver = create_user_friendly_driver(headless=True)
        driver.quit()
        result['webdriver_available'] = True
        result['diagnosis'].append("[OK] WebDriver创建成功")
    except Exception as e:
        result['diagnosis'].append(f"[FAILED] WebDriver创建失败: {str(e)[:100]}")

    return result
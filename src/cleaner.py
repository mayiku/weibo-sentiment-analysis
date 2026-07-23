"""
数据清洗模块 — 清理CSV中的评论文本
"""
import csv
import re
import unicodedata
import pandas as pd
from src.logger import get_logger

log = get_logger(__name__)


def _is_meaningless_symbol_only(text: str) -> bool:
    """Return True for punctuation-only noise while preserving emoji signals."""
    value = str(text).strip()
    if not value:
        return True
    if any(char.isalnum() for char in value):
        return False
    # Unicode symbols (notably category So) carry sentiment; punctuation and
    # separators alone do not.
    return not any(unicodedata.category(char).startswith("S") for char in value)


def _read_csv_robust(input_path: str) -> pd.DataFrame:
    """
    健壮的 CSV 读取 — 处理纯表头（0行数据）等边缘情况

    某些 pandas 版本在读取只有表头的 utf-8-sig CSV 时会崩溃，
    这里用 csv 模块先行解析，确保始终返回正确的 DataFrame。
    """
    log.info("读取文件: %s", input_path)

    # 先用 csv 模块读取原始数据，避免 pandas 边缘情况
    rows = []
    try:
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
    except UnicodeDecodeError:
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        log.error("文件不存在: %s", input_path)
        raise

    if not rows:
        log.warning("CSV 文件为空 — 创建空 DataFrame")
        return pd.DataFrame({'评论内容': []})

    header = rows[0]
    data = rows[1:] if len(rows) > 1 else []

    if data:
        df = pd.DataFrame(data, columns=header)
    else:
        # pandas 某些版本对空数据 + 中文列名有 bug，用 columns 参数创建
        df = pd.DataFrame(columns=header)
    log.info("读取 %d 行, 列: %s", len(df), list(df.columns))
    return df


def clean_dataframe(df: pd.DataFrame,
                    columns: list[str] = None,
                    remove_empty: bool = True,
                    remove_symbols: bool = True,
                    remove_duplicates: bool = True,
                    strip_whitespace: bool = True) -> pd.DataFrame:
    """
    清理 DataFrame 中的文本数据

    参数:
        df: 输入 DataFrame (必须包含 '评论内容' 列)
        columns: 要清理的列名列表，默认 ['评论内容']
        remove_empty: 是否删除空行
        remove_symbols: 是否删除纯符号行
        remove_duplicates: 是否去重
        strip_whitespace: 是否去除两端空格

    返回:
        清理后的 DataFrame
    """
    if columns is None:
        columns = ['评论内容']

    original_rows = len(df)
    log.info("开始数据清洗 — 原始行数: %d", original_rows)

    # 空 DataFrame 快速返回
    if original_rows == 0:
        log.info("DataFrame 为空 (0 行)，跳过清洗")
        df.attrs["cleaning_metadata"] = {
            "raw_comments": 0, "cleaned_comments": 0, "valid_comments": 0,
            "unique_comments": 0, "removed_comments": 0,
        }
        return df

    cleaned_df = df.copy()
    valid_rows_before_dedup = original_rows

    # 检查列是否存在
    missing = [c for c in columns if c not in cleaned_df.columns]
    if missing:
        raise ValueError(f"DataFrame 中缺少列: {missing}。现有列: {list(cleaned_df.columns)}")

    for col in columns:
        if strip_whitespace:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # 删除空行
    if remove_empty:
        before = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=columns, how='all')
        cleaned_df = cleaned_df[cleaned_df[columns[0]].astype(str).str.len() > 0]
        log.info("删除空行: %d -> %d", before, len(cleaned_df))

    # 空行检查后的快速返回
    if len(cleaned_df) == 0:
        log.info("清洗后无剩余数据")
        cleaned_df.attrs["cleaning_metadata"] = {
            "raw_comments": original_rows,
            "cleaned_comments": 0,
            "valid_comments": 0,
            "unique_comments": 0,
            "removed_comments": original_rows,
        }
        return cleaned_df

    # 删除纯符号行
    if remove_symbols:
        before = len(cleaned_df)
        for col in columns:
            cleaned_df = cleaned_df[
                ~cleaned_df[col].astype(str).apply(
                    _is_meaningless_symbol_only
                )
            ]
        log.info("删除纯符号行: %d -> %d", before, len(cleaned_df))

    valid_rows_before_dedup = len(cleaned_df)

    # 去重
    if remove_duplicates:
        before = len(cleaned_df)
        dedup_columns = list(columns)
        if '帖子ID' in cleaned_df.columns and '帖子ID' not in dedup_columns:
            dedup_columns.append('帖子ID')
        cleaned_df['duplicate_count'] = (
            cleaned_df.groupby(dedup_columns, dropna=False)[columns[0]].transform('size').astype(int)
        )
        cleaned_df = cleaned_df.drop_duplicates(subset=dedup_columns)
        log.info("去重: %d -> %d", before, len(cleaned_df))
    else:
        cleaned_df['duplicate_count'] = 1

    rows_removed = original_rows - len(cleaned_df)
    cleaned_df.attrs["cleaning_metadata"] = {
        "raw_comments": original_rows,
        "cleaned_comments": len(cleaned_df),
        "valid_comments": valid_rows_before_dedup,
        "unique_comments": len(cleaned_df),
        "removed_comments": rows_removed,
    }
    log.info("清洗完成 — 删除 %d 行, 剩余 %d 行", rows_removed, len(cleaned_df))
    return cleaned_df


def clean_csv(input_path: str, output_path: str = None,
              columns: list[str] = None, **kwargs) -> pd.DataFrame:
    """
    读取并清洗 CSV 文件

    参数:
        input_path: 输入 CSV 文件路径
        output_path: 输出 CSV 路径 (可选)
        columns: 要清理的列
        **kwargs: 传递给 clean_dataframe 的参数

    返回:
        清洗后的 DataFrame
    """
    df = _read_csv_robust(input_path)
    cleaned = clean_dataframe(df, columns=columns, **kwargs)

    if output_path:
        cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
        log.info("清洗结果已保存: %s", output_path)

    return cleaned

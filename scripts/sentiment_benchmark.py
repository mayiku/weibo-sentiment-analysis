#!/usr/bin/env python3
"""Create annotation worksheets or evaluate a model on labeled Weibo data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sentiment.benchmark import build_labeling_sample, evaluate_model, load_labeled_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="微博情绪模型人工标注与基准评估")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template", help="从评论 CSV 生成去重标注表")
    template.add_argument("input_csv")
    template.add_argument("output_csv")
    template.add_argument("--size", type=int, default=500)

    evaluate = subparsers.add_parser("evaluate", help="用已标注 CSV 评估模型")
    evaluate.add_argument("labeled_csv")
    evaluate.add_argument("--model", choices=["snownlp", "paddle", "bert", "hybrid"], required=True)
    evaluate.add_argument("--output-json")

    args = parser.parse_args()
    if args.command == "template":
        source = pd.read_csv(args.input_csv, encoding="utf-8-sig")
        worksheet = build_labeling_sample(source, sample_size=args.size)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        worksheet.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"已生成 {len(worksheet)} 条待标注样本: {args.output_csv}")
        return

    labeled = load_labeled_dataset(args.labeled_csv)
    metrics = evaluate_model(labeled, args.model).to_dict()
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

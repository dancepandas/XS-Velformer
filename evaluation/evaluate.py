"""
评估 CLI 入口

用法:
    # 评估单个模型
    python -m evaluation.evaluate \
        --model_type lstm \
        --checkpoint checkpoints/lstm_baseline.pt \
        --test_cases result/fixed_test_cases.pkl \
        --output result/lstm_eval.json

    # 评估多个模型
    python -m evaluation.evaluate \
        --models query_transformer,lstm,gan_z0,gan_mean5,diffusion_mean5 \
        --checkpoints checkpoints/model.pt,checkpoints/lstm_baseline.pt,... \
        --test_cases result/fixed_test_cases.pkl \
        --output result/baseline_comparison.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
import csv
import argparse
import logging

from evaluation.model_adapters import create_adapter
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


def evaluate_single(
    model_type: str,
    checkpoint: str,
    test_cases_path: str,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    adapter = create_adapter(model_type, device=device)
    if adapter.requires_checkpoint and checkpoint and checkpoint.lower() != "none":
        adapter.load(checkpoint)
    elif adapter.requires_checkpoint:
        raise ValueError(f"Model '{model_type}' requires a checkpoint, but none provided")
    evaluator = Evaluator(
        test_cases_path=test_cases_path,
        adapter=adapter,
        batch_size=batch_size,
        device=device,
    )
    return evaluator.run()


def results_to_json(all_results: dict, meta: dict) -> str:
    output = {
        "meta": meta,
        "models": {},
    }
    for model_name, result in all_results.items():
        model_out = {
            "overall": result["overall"],
            "by_strategy": result["by_strategy"],
            "by_mask_ratio": result["by_mask_ratio"],
            "n_cases": result["n_cases"],
        }
        output["models"][model_name] = model_out
    return json.dumps(output, indent=2, ensure_ascii=False)


def results_to_csv(all_results: dict) -> str:
    from evaluation.metrics import METRIC_KEYS
    metric_cols = METRIC_KEYS + ["n_points"]
    rows = []
    for model_name, result in all_results.items():
        overall = result["overall"]
        row = {"model": model_name, "strategy": "overall"}
        for k in metric_cols:
            row[k] = overall.get(k, 0)
        row["n_cases"] = result["n_cases"]
        rows.append(row)

        for strategy, metrics in sorted(result["by_strategy"].items()):
            row = {"model": model_name, "strategy": strategy}
            for k in metric_cols:
                row[k] = metrics.get(k, 0)
            row["n_cases"] = ""
            rows.append(row)

        for ratio, metrics in sorted(result["by_mask_ratio"].items()):
            row = {"model": model_name, "strategy": f"mask_ratio={ratio}"}
            for k in metric_cols:
                row[k] = metrics.get(k, 0)
            row["n_cases"] = ""
            rows.append(row)

    fieldnames = ["model", "strategy"] + metric_cols + ["n_cases"]
    lines = [",".join(fieldnames)]
    for row in rows:
        lines.append(",".join(str(row.get(f, "")) for f in fieldnames))
    return "\n".join(lines)


def save_per_case_pkl(all_results: dict, path: str):
    per_case_data = {}
    for model_name, result in all_results.items():
        per_case_data[model_name] = result["per_case"]
    with open(path, "wb") as f:
        pickle.dump(per_case_data, f)
    logger.info(f"Per-case results saved to {path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="标准测试评估")
    parser.add_argument("--model_type", type=str, default=None,
                        help="单模型评估时的模型类型")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="单模型评估时的 checkpoint 路径")
    parser.add_argument("--models", type=str, default=None,
                        help="多模型评估, 逗号分隔的模型类型列表")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="多模型评估, 逗号分隔的 checkpoint 路径列表")
    parser.add_argument("--test_cases", type=str, required=True,
                        help="fixed_test_cases.pkl 路径")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="result/eval_output.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    all_results = {}

    if args.models:
        model_types = [m.strip() for m in args.models.split(",")]
        ckpt_paths = []
        if args.checkpoints:
            ckpt_paths = [c.strip() for c in args.checkpoints.split(",")]
        while len(ckpt_paths) < len(model_types):
            ckpt_paths.append("none")

        for model_type, ckpt_path in zip(model_types, ckpt_paths):
            logger.info(f"Evaluating {model_type} ...")
            result = evaluate_single(
                model_type, ckpt_path, args.test_cases,
                batch_size=args.batch_size, device=device,
            )
            all_results[model_type] = result
            overall = result["overall"]
            logger.info(f"  {model_type} overall: RMSE={overall['rmse_raw']:.4f} "
                        f"NSE={overall['nse_raw']:.4f} KGE={overall['kge_raw']:.4f}")

    elif args.model_type and args.checkpoint:
        logger.info(f"Evaluating {args.model_type} ...")
        result = evaluate_single(
            args.model_type, args.checkpoint, args.test_cases,
            batch_size=args.batch_size, device=device,
        )
        all_results[args.model_type] = result
        overall = result["overall"]
        logger.info(f"  {args.model_type} overall: RMSE={overall['rmse_raw']:.4f} "
                    f"NSE={overall['nse_raw']:.4f} KGE={overall['kge_raw']:.4f}")
    else:
        parser.error("需要指定 --model_type + --checkpoint 或 --models [+ --checkpoints]")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    meta = {
        "test_cases": args.test_cases,
        "device": device,
        "batch_size": args.batch_size,
    }

    json_str = results_to_json(all_results, meta)
    json_path = args.output if args.output.endswith(".json") else args.output.rsplit(".", 1)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    logger.info(f"JSON results saved to {json_path}")

    csv_str = results_to_csv(all_results)
    csv_path = args.output if args.output.endswith(".csv") else args.output.rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_str)
    logger.info(f"CSV results saved to {csv_path}")

    pkl_path = args.output.rsplit(".", 1)[0] + "_per_case.pkl"
    save_per_case_pkl(all_results, pkl_path)


if __name__ == "__main__":
    main()

"""
Query Transformer 训练入口

本脚本是主模型训练的命令行入口, 支持覆盖 config.py 中的默认参数。

用法:
    python main.py                         # 使用 config.py 全部默认值
    python main.py --epochs 100            # 只覆盖指定参数
    python main.py --help                  # 查看可覆盖参数

训练完成后会自动保存:
    - checkpoint:        由 --save 指定 (默认 checkpoints/model.pt)
    - 训练曲线:          由 --plot_path 指定 (默认 checkpoints/training_curves.png)
    - 逐 step 训练日志:  由 --log_path 指定 (默认 checkpoints/training_log.json)
"""

import argparse
import logging
import torch
from config import ModelConfig, TrainConfig, DataConfig
from train import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Query Transformer 训练 (默认值来自 config.py)")
    # 数据参数
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["stratified", "station", "time", "station_and_time"])
    # 训练参数
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    # 模型参数
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--ctx_layers", type=int, default=None)
    parser.add_argument("--qry_layers", type=int, default=None)
    parser.add_argument("--cross_layers", type=int, default=None)
    # 输出路径
    parser.add_argument("--save", type=str, default="checkpoints/model.pt")
    parser.add_argument("--plot_path", type=str, default="checkpoints/training_curves.png")
    parser.add_argument("--log_path", type=str, default="checkpoints/training_log.json")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # 从 config.py 默认值构建, 命令行参数仅覆盖非 None 的项
    model_kw = {}
    if args.d_model is not None:     model_kw["d_model"] = args.d_model
    if args.num_heads is not None:   model_kw["num_heads"] = args.num_heads
    if args.ctx_layers is not None:  model_kw["context_encoder_layers"] = args.ctx_layers
    if args.qry_layers is not None:  model_kw["query_encoder_layers"] = args.qry_layers
    if args.cross_layers is not None: model_kw["cross_attention_layers"] = args.cross_layers

    train_kw = {}
    if args.epochs is not None:      train_kw["max_epochs"] = args.epochs
    if args.batch_size is not None:  train_kw["batch_size"] = args.batch_size
    if args.lr is not None:          train_kw["learning_rate"] = args.lr

    data_kw = {}
    if args.data is not None:        data_kw["data_path"] = args.data
    if args.split is not None:       data_kw["split_mode"] = args.split

    model_cfg = ModelConfig(**model_kw)
    train_cfg = TrainConfig(**train_kw)
    data_cfg = DataConfig(**data_kw)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Data:   {data_cfg.data_path} (split={data_cfg.split_mode})")
    print(f"Model:  d_model={model_cfg.d_model}, heads={model_cfg.num_heads}, "
          f"ctx={model_cfg.context_encoder_layers}, qry={model_cfg.query_encoder_layers}, "
          f"cross={model_cfg.cross_attention_layers}")
    print(f"Train:  epochs={train_cfg.max_epochs}, batch={train_cfg.batch_size}, lr={train_cfg.learning_rate}")
    print()

    trainer = Trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        device=device,
    )

    trainer.prepare_data()
    trainer.fit(plot_path=args.plot_path, log_path=args.log_path)
    trainer.save(args.save)


if __name__ == "__main__":
    main()

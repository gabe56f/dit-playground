import argparse
import json
from pathlib import Path

args = argparse.ArgumentParser("Trainer")
args.add_argument("data_dir", type=str)
args.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
args.add_argument(
    "-d",
    "--device",
    type=str,
    default="cuda:0",
    help="Device to use for training (e.g., 'cuda:0' or 'cpu')",
)
args.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
args.add_argument(
    "-o",
    "--output-dir",
    type=str,
    default="checkpoints/",
    help="Directory to save checkpoints",
)
args.add_argument(
    "-e", "--epochs", type=int, default=50, help="Number of training epochs"
)
args.add_argument("-lr", type=float, default=4e-4, help="Learning rate")
args.add_argument(
    "--learning-rate-stepping",
    type=str,
    choices=["epoch", "step"],
    default="epoch",
    help="Learning rate stepping strategy",
)
args.add_argument(
    "-wd", "--weight-decay", type=float, default=0.05, help="Weight decay for optimizer"
)
args.add_argument(
    "-b", "--batch-size", type=int, default=4, help="Batch size for training"
)
args.add_argument(
    "-s", "--save-every", type=int, default=1, help="Save checkpoint every N epochs"
)
args.add_argument(
    "-cl",
    "--context-length",
    type=int,
    default=4096,
    help="Context length for the model",
)
args.add_argument(
    "-l",
    "--dataset-length",
    type=int,
    default=4096,
    help="Context length of data to load",
)
args.add_argument(
    "-m", "--mlp-ratio", type=float, default=6.0, help="MLP ratio for the model"
)
args.add_argument(
    "-nh",
    "--n-heads",
    type=int,
    default=12,
    help="Number of attention heads in the model",
)
args.add_argument(
    "-k",
    "--n-kv-heads",
    type=int,
    default=6,
    help="Number of key-value heads in the model",
)
args.add_argument("--depth", type=int, default=32, help="Depth of the model")
args.add_argument(
    "-w", "--width", type=int, default=768, help="Width of the model (d_model)"
)
args.add_argument(
    "--mla-head-dim", type=int, default=32, help="MLA head dimension rank"
)
args.add_argument(
    "--dropout", type=float, default=0.0, help="Dropout rate for the model"
)
args.add_argument(
    "-g",
    "--gradient-accumulation",
    type=int,
    default=1,
    help="Gradient accumulation steps",
)
args.add_argument(
    "--learn-softmax",
    type=bool,
    default=True,
    help="Whether to learn a separate softmax scale",
)
args.add_argument(
    "--softmax-bias",
    type=bool,
    default=False,
    help="Whether to learn a separate bias for the softmax scale",
)
args.add_argument(
    "--softmax",
    type=float,
    default=0.43,
    help="Default scale per-head for the softmax scale",
)
args.add_argument("--load", type=str, default=None, help="Path to a checkpoint to load")
args.add_argument(
    "--position-embedding",
    type=str,
    default="fope",
    choices=["fope", "rope", "none"],
    help="Position embedding type",
)


if __name__ == "__main__":
    args = args.parse_args()

    # Keep big imports here to reduce argparse time
    import torch
    import wandb
    from src.trainer import Trainer

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    config_file = args.config or Path("config.json")
    if isinstance(config_file, str) and Path(config_file).exists():
        config = json.loads(Path(config_file).read_text())
    else:
        config = {
            "device": device,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_accumulation": args.gradient_accumulation,
            "batch_size": args.batch_size,
            "num_timesteps": 1000,
            "save_every": args.save_every,
            "checkpoint_dir": args.output_dir,
            "learning_rate_stepping": args.learning_rate_stepping,
            "dataset_config": {
                "dataset_dir": args.data_dir,
                "context_len": args.dataset_length,
            },
            "model_config": {
                "d_model": args.width,
                "depth": args.depth,
                "n_heads": args.n_heads,
                "n_kv_heads": args.n_kv_heads,
                "mla_dim_rank": args.mla_head_dim,
                "learn_softmax": args.learn_softmax,
                "softmax_bias": args.softmax_bias,
                "softmax_scale_init": args.softmax,
                "softcap": 20.0,
                "context_len": args.context_length,
                "position_embedding": "fope",
                "position_embedding_settings": {},
                "in_channels": 6,
                "dac_codebooks": 18,
                "dac_vocab": 1024,
                "mlp_ratio": args.mlp_ratio,
                "dropout": args.dropout,
            },
        }
    config["device"] = device

    if isinstance(config_file, Path) and not config_file.exists():
        with open("config.json", "x") as f:
            json.dump(config, f, indent=2)

    run = wandb.init(
        project="osu-mmdit-flow",
        config=config,
    )

    print(f"Using device: {config['device']}")
    trainer = Trainer(config, run)
    if args.load is not None:
        trainer._load_checkpoint(args.load)

    trainer.train()

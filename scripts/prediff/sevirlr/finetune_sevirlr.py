import os
import argparse
import platform
from collections import OrderedDict

import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers
from lightning.pytorch.callbacks import (
    Callback, LearningRateMonitor, DeviceStatsMonitor,
    EarlyStopping, ModelCheckpoint,
)
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf

from prediff.utils.path import (
    default_exps_dir,
    default_pretrained_vae_dir,
    default_pretrained_earthformerunet_dir,
)
from prediff.utils.download import (
    download_pretrained_weights,
    pretrained_sevirlr_vae_name,
    pretrained_sevirlr_earthformerunet_name,
)
from prediff.utils.pl_checkpoint import pl_load
from prediff.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule

from train_sevirlr_prediff import PreDiffSEVIRPLModule


def build_dm_from_cfg(cfg: dict, micro_batch_size: int, num_workers: int):
    dm = SEVIRLightningDataModule(
        seq_len=cfg["seq_len"],
        sample_mode=cfg["sample_mode"],
        stride=cfg["stride"],
        batch_size=micro_batch_size,
        layout=cfg["layout"],
        output_type=np.float32,
        preprocess=True,
        rescale_method="01",
        verbose=False,
        aug_mode=cfg["aug_mode"],
        ret_contiguous=False,
        dataset_name=cfg["dataset_name"],
        sevir_dir=cfg.get("sevir_dir", None),
        start_date=cfg.get("start_date", None),
        train_test_split_date=cfg.get("train_test_split_date", None),
        end_date=cfg.get("end_date", None),
        val_ratio=cfg.get("val_ratio", 0.1),
        num_workers=num_workers,
    )
    return dm


def get_parser():
    p = argparse.ArgumentParser("Finetune PreDiff on custom SEVIR-LR dataset")
    p.add_argument("--save", default="finetune_sevirlr_custom", type=str)
    p.add_argument("--cfg", default=None, type=str,
                   help="Base YAML config. Defaults to prediff_sevirlr_v1.yaml")
    p.add_argument("--sevir_dir", default="datasets/train", type=str,
                   help="Path to the SEVIR-LR dataset root (containing CATALOG.csv and data/). If omitted, defaults to datasets/sevirlr.")
    p.add_argument("--pretrained", action="store_true",
                   help="Load pretrained Earthformer-UNet + VAE weights before training.")
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--gpus", default=1, type=int)
    p.add_argument("--nodes", default=1, type=int)
    p.add_argument("--micro_batch_size", default=2, type=int)
    p.add_argument("--val_ratio", default=0.1, type=float)
    p.add_argument("--lr", default=None, type=float, help="Override learning rate (optional)")
    p.add_argument("--accum", default=None, type=int,
                   help="accumulate_grad_batches override (optional). When None, computed from total_batch_size if available.")
    p.add_argument("--sanity_steps", type=int, default=2, help="num_sanity_val_steps for Lightning Trainer")
    p.add_argument("--fast", action="store_true", help="Limit batches for a quick sanity run.")
    return p


def main():
    args = get_parser().parse_args()

    # Resolve cfg path
    if args.cfg is None:
        args.cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "prediff_sevirlr_v1.yaml"))

    # Optionally download pretrained weights (VAE + Earthformer-UNet)
    if args.pretrained:
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_vae_name,
                                    save_dir=default_pretrained_vae_dir,
                                    exist_ok=False)
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_earthformerunet_name,
                                    save_dir=default_pretrained_earthformerunet_dir,
                                    exist_ok=False)

    # Load and override config for finetuning
    oc = OmegaConf.load(open(args.cfg, "r"))
    # Dataset: ensure we use the generated SEVIR-LR dataset entirely for train/val splits
    oc.dataset.dataset_name = "sevirlr"
    oc.dataset.val_ratio = args.val_ratio
    if args.sevir_dir is not None:
        oc.dataset.sevir_dir = os.path.abspath(args.sevir_dir)
    # Use full catalog for training split (no time-based split)
    oc.dataset.start_date = None
    oc.dataset.train_test_split_date = None
    oc.dataset.end_date = None
    # Keep stride >= out_len to avoid leakage
    oc.dataset.stride = oc.dataset.out_len
    # Optim: epochs and (optionally) lr/micro-batch-size
    oc.optim.max_epochs = int(args.epochs)
    oc.optim.micro_batch_size = int(args.micro_batch_size)
    if args.lr is not None:
        oc.optim.lr = float(args.lr)

    # Save merged finetune cfg into the experiment folder
    save_dir = os.path.join(default_exps_dir, args.save)
    os.makedirs(save_dir, exist_ok=True)
    ft_cfg_path = os.path.join(save_dir, "cfg_finetune.yaml")
    with open(ft_cfg_path, "w") as f:
        OmegaConf.save(oc, f)

    # Seed + workers
    seed = oc.optim.get("seed", 0) or 0
    seed_everything(seed, workers=True)
    num_workers = 0 if platform.system() == "Windows" else 8

    # Build datamodule from modified cfg
    dm = build_dm_from_cfg(
        cfg=OmegaConf.to_object(oc.dataset),
        micro_batch_size=int(oc.optim.micro_batch_size),
        num_workers=num_workers,
    )
    dm.prepare_data()
    dm.setup()
    # Log dataset resolution and split sizes for verification
    try:
        print(f"Using dataset: {dm.dataset_name}")
        print(f"SEVIR dir: {dm.sevir_dir}")
        print(f"Catalog: {dm.catalog_path}")
        print(f"Train/Val samples: {dm.num_train_samples} / {dm.num_val_samples}")
    except Exception:
        pass

    # Determine gradient accumulation
    total_batch_size = oc.optim.get("total_batch_size", None)
    if total_batch_size is not None and args.accum is None:
        accum = max(1, int(total_batch_size) // (int(oc.optim.micro_batch_size) * args.gpus * args.nodes))
    else:
        accum = int(args.accum) if args.accum is not None else 1

    # Compute total steps
    total_num_steps = PreDiffSEVIRPLModule.get_total_num_steps(
        epoch=oc.optim.max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=int(oc.optim.micro_batch_size) * args.gpus * args.nodes,
    )

    # Build PL module with finetune cfg
    pl_module = PreDiffSEVIRPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=ft_cfg_path,
    )

    # Optionally load Earthformer-UNet pretrained weights (finetune start)
    if args.pretrained:
        earthformerunet_ckpt_path = os.path.join(default_pretrained_earthformerunet_dir,
                                                 pretrained_sevirlr_earthformerunet_name)
        state_dict = torch.load(earthformerunet_ckpt_path, map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict, strict=False)

    # Trainer kwargs (CPU/GPU auto)
    callbacks = [ModelCheckpoint(
        monitor=pl_module.oc.optim.monitor,
        dirpath=os.path.join(pl_module.save_dir, "checkpoints"),
        filename="{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=pl_module.oc.optim.save_top_k,
        save_last=True,
        mode="min",
    )]
    if pl_module.oc.logging.monitor_lr:
        callbacks += [LearningRateMonitor(logging_interval='step')]
    if pl_module.oc.logging.monitor_device:
        callbacks += [DeviceStatsMonitor()]
    if pl_module.oc.optim.early_stop:
        callbacks += [EarlyStopping(monitor=pl_module.oc.optim.monitor,
                                    patience=pl_module.oc.optim.early_stop_patience,
                                    mode=pl_module.oc.optim.early_stop_mode)]

    logger = [
        pl_loggers.TensorBoardLogger(save_dir=pl_module.save_dir),
        pl_loggers.CSVLogger(save_dir=pl_module.save_dir)
    ]

    trainer_kwargs = dict(
        accelerator="auto",
        devices=args.gpus,
        num_nodes=args.nodes,
        max_epochs=int(oc.optim.max_epochs),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=pl_module.oc.optim.gradient_clip_val,
        precision=pl_module.oc.trainer.precision,
        check_val_every_n_epoch=pl_module.oc.trainer.check_val_every_n_epoch,
        accumulate_grad_batches=accum,
        num_sanity_val_steps=args.sanity_steps,
        log_every_n_steps=max(1, int(pl_module.oc.trainer.log_step_ratio * total_num_steps)),
    )
    if (args.gpus * args.nodes) > 1:
        trainer_kwargs["strategy"] = DDPStrategy(
            find_unused_parameters=pl_module.oc.trainer.find_unused_parameters,
            process_group_backend=("gloo" if platform.system() == "Windows" else "nccl"),
        )
    trainer = Trainer(**trainer_kwargs)

    # Train
    extra = {}
    if args.fast:
        extra.update({'limit_train_batches': 10, 'limit_val_batches': 10})
    trainer.fit(model=pl_module, datamodule=dm, **extra)

    # Save model-only weights (Earthformer-UNet) for future inference/finetune
    # Export finetuned Earthformer-UNet weights from best checkpoint if available, else fall back to last.ckpt
    best_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, 'checkpoint_callback') else None
    export_path = None
    if best_path and os.path.exists(best_path):
        export_path = best_path
    else:
        last_path = os.path.join(pl_module.save_dir, "checkpoints", "last.ckpt")
        if os.path.exists(last_path):
            export_path = last_path
    if export_path is not None:
        pl_ckpt = pl_load(path_or_url=export_path, map_location=torch.device("cpu"))
        pl_state_dict = pl_ckpt
        model_key_prefix = "torch_nn_module."
        state_dict = OrderedDict()
        for key, val in pl_state_dict.items():
            if key.startswith(model_key_prefix):
                state_dict[key.replace(model_key_prefix, "")] = val
        # Derive zero-based epoch number matching checkpoint naming
        base_name = os.path.basename(export_path)
        name_root = os.path.splitext(base_name)[0]
        if name_root.isdigit():
            epoch_num = int(name_root)
        elif name_root == "last":
            epoch_num = int(getattr(trainer, "current_epoch", -1))
        else:
            epoch_num = -1
        if epoch_num >= 0:
            epoch_str = f"{epoch_num:03d}"
            epoch_path = os.path.join(pl_module.save_dir, "checkpoints", f"sevirlr_earthformerunet_finetuned_{epoch_str}.pt")
            torch.save(state_dict, epoch_path)


if __name__ == "__main__":
    main()

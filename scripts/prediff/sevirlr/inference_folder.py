import argparse
import os
import glob
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from prediff.utils.path import (
    default_exps_dir,
    default_pretrained_vae_dir,
    default_pretrained_earthformerunet_dir,
    default_pretrained_alignment_dir,
)
from prediff.utils.download import (
    download_pretrained_weights,
    pretrained_sevirlr_vae_name,
    pretrained_sevirlr_earthformerunet_name,
    pretrained_sevirlr_alignment_name,
)
from prediff.diffusion.latent_diffusion import LatentDiffusion
from prediff.taming import AutoencoderKL
from prediff.models.cuboid_transformer import CuboidTransformerUNet
from prediff.diffusion.knowledge_alignment.sevir import SEVIRAvgIntensityAlignment


def _ensure_4d_thwc(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is THWC. Accepts T H W, adds C=1; or already THWC.
    Casts to float32 without scaling.
    """
    if arr.ndim == 3:
        T, H, W = arr.shape
        arr = arr.reshape(T, H, W, 1)
    elif arr.ndim == 4:
        pass
    else:
        raise ValueError(f"Expected 3D or 4D array (T,H,W[,C]), got shape {arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def _list_files(input_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' in '{input_dir}'.")
    return files


class PreDiffModuleWrapper(LatentDiffusion):
    """
    Minimal wrapper to instantiate the model from the SEVIR config without Lightning Trainer.
    Mirrors the construction in train_sevirlr_prediff.py but strips logging/trainer bits.
    """

    def __init__(self, oc_file: Optional[str], save_dir: str):
        if oc_file is None:
            raise ValueError("A config YAML (--cfg) is required to define model shapes.")

        oc = OmegaConf.load(open(oc_file, "r"))
        self.oc = oc
        self.save_dir = os.path.join(default_exps_dir, save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        # Build latent model (CuboidTransformerUNet)
        latent_model_cfg = OmegaConf.to_object(oc.model.latent_model)
        num_blocks = len(latent_model_cfg["depth"])
        if isinstance(latent_model_cfg["self_pattern"], str):
            block_attn_patterns = [latent_model_cfg["self_pattern"]] * num_blocks
        else:
            block_attn_patterns = list(latent_model_cfg["self_pattern"])  # omegaconf -> list
        latent_model = CuboidTransformerUNet(
            input_shape=latent_model_cfg["input_shape"],
            target_shape=latent_model_cfg["target_shape"],
            base_units=latent_model_cfg["base_units"],
            scale_alpha=latent_model_cfg["scale_alpha"],
            num_heads=latent_model_cfg["num_heads"],
            attn_drop=latent_model_cfg["attn_drop"],
            proj_drop=latent_model_cfg["proj_drop"],
            ffn_drop=latent_model_cfg["ffn_drop"],
            downsample=latent_model_cfg["downsample"],
            downsample_type=latent_model_cfg["downsample_type"],
            upsample_type=latent_model_cfg["upsample_type"],
            upsample_kernel_size=latent_model_cfg["upsample_kernel_size"],
            depth=latent_model_cfg["depth"],
            block_attn_patterns=block_attn_patterns,
            num_global_vectors=latent_model_cfg["num_global_vectors"],
            use_global_vector_ffn=latent_model_cfg["use_global_vector_ffn"],
            use_global_self_attn=latent_model_cfg["use_global_self_attn"],
            separate_global_qkv=latent_model_cfg["separate_global_qkv"],
            global_dim_ratio=latent_model_cfg["global_dim_ratio"],
            ffn_activation=latent_model_cfg["ffn_activation"],
            gated_ffn=latent_model_cfg["gated_ffn"],
            norm_layer=latent_model_cfg["norm_layer"],
            padding_type=latent_model_cfg["padding_type"],
            checkpoint_level=latent_model_cfg["checkpoint_level"],
            pos_embed_type=latent_model_cfg["pos_embed_type"],
            use_relative_pos=latent_model_cfg["use_relative_pos"],
            self_attn_use_final_proj=latent_model_cfg["self_attn_use_final_proj"],
            attn_linear_init_mode=latent_model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=latent_model_cfg["ffn_linear_init_mode"],
            ffn2_linear_init_mode=latent_model_cfg["ffn2_linear_init_mode"],
            attn_proj_linear_init_mode=latent_model_cfg["attn_proj_linear_init_mode"],
            conv_init_mode=latent_model_cfg["conv_init_mode"],
            down_linear_init_mode=latent_model_cfg["down_up_linear_init_mode"],
            up_linear_init_mode=latent_model_cfg["down_up_linear_init_mode"],
            global_proj_linear_init_mode=latent_model_cfg["global_proj_linear_init_mode"],
            norm_init_mode=latent_model_cfg["norm_init_mode"],
            time_embed_channels_mult=latent_model_cfg["time_embed_channels_mult"],
            time_embed_use_scale_shift_norm=latent_model_cfg["time_embed_use_scale_shift_norm"],
            time_embed_dropout=latent_model_cfg["time_embed_dropout"],
            unet_res_connect=latent_model_cfg["unet_res_connect"],
        )

        # First-stage VAE
        vae_cfg = OmegaConf.to_object(oc.model.vae)
        first_stage_model = AutoencoderKL(
            down_block_types=vae_cfg["down_block_types"],
            in_channels=vae_cfg["in_channels"],
            block_out_channels=vae_cfg["block_out_channels"],
            act_fn=vae_cfg["act_fn"],
            latent_channels=vae_cfg["latent_channels"],
            up_block_types=vae_cfg["up_block_types"],
            norm_num_groups=vae_cfg["norm_num_groups"],
            layers_per_block=vae_cfg["layers_per_block"],
            out_channels=vae_cfg["out_channels"],
        )
        pretrained_ckpt_path = vae_cfg.get("pretrained_ckpt_path", None)
        if pretrained_ckpt_path:
            state_dict = torch.load(
                os.path.join(default_pretrained_vae_dir, pretrained_ckpt_path),
                map_location=torch.device("cpu"),
            )
            first_stage_model.load_state_dict(state_dict=state_dict)

        diffusion_cfg = OmegaConf.to_object(oc.model.diffusion)
        # Backward-compatible defaults if some keys are absent in YAML
        if "timesteps" not in diffusion_cfg:
            diffusion_cfg["timesteps"] = 1000
        super().__init__(
            torch_nn_module=latent_model,
            layout=oc.layout.layout,
            data_shape=diffusion_cfg["data_shape"],
            timesteps=diffusion_cfg["timesteps"],
            beta_schedule=diffusion_cfg["beta_schedule"],
            loss_type="l2",
            monitor="valid_loss_epoch",
            use_ema=diffusion_cfg["use_ema"],
            log_every_t=diffusion_cfg["log_every_t"],
            clip_denoised=diffusion_cfg["clip_denoised"],
            linear_start=diffusion_cfg["linear_start"],
            linear_end=diffusion_cfg["linear_end"],
            cosine_s=diffusion_cfg["cosine_s"],
            given_betas=diffusion_cfg["given_betas"],
            original_elbo_weight=diffusion_cfg["original_elbo_weight"],
            v_posterior=diffusion_cfg["v_posterior"],
            l_simple_weight=diffusion_cfg["l_simple_weight"],
            parameterization=diffusion_cfg["parameterization"],
            learn_logvar=diffusion_cfg["learn_logvar"],
            logvar_init=diffusion_cfg["logvar_init"],
            latent_shape=diffusion_cfg["latent_shape"],
            first_stage_model=first_stage_model,
            cond_stage_model=diffusion_cfg["cond_stage_model"],
            num_timesteps_cond=diffusion_cfg["num_timesteps_cond"],
            cond_stage_trainable=diffusion_cfg["cond_stage_trainable"],
            cond_stage_forward=diffusion_cfg["cond_stage_forward"],
            scale_by_std=diffusion_cfg["scale_by_std"],
            scale_factor=diffusion_cfg["scale_factor"],
        )

        # Knowledge alignment (optional)
        align_cfg = OmegaConf.to_object(oc.model.align)
        self.use_alignment = align_cfg.get("alignment_type", None) is not None
        if self.use_alignment:
            alignment_ckpt_path = os.path.join(
                default_pretrained_alignment_dir, align_cfg["model_ckpt_path"]
            )
            self.alignment_obj = SEVIRAvgIntensityAlignment(
                alignment_type=align_cfg["alignment_type"],
                guide_scale=align_cfg["guide_scale"],
                model_type=align_cfg["model_type"],
                model_args=align_cfg["model_args"],
                model_ckpt_path=alignment_ckpt_path,
            )
            self.alignment_model = self.alignment_obj.model
            self.set_alignment(alignment_fn=self.alignment_obj.get_mean_shift)
        else:
            self.set_alignment(alignment_fn=None)


def _load_torch_nn_module_weights(module: PreDiffModuleWrapper, ckpt_path: Optional[str], use_pretrained_if_none: bool):
    """
    Load weights into module.torch_nn_module.
    If ckpt_path is a Lightning checkpoint, extract keys with prefix 'torch_nn_module.'.
    Otherwise treat as plain state_dict.
    If ckpt_path is None and use_pretrained_if_none, load default Earthformer-UNet weights.
    """
    if ckpt_path is None:
        if not use_pretrained_if_none:
            raise ValueError("No --ckpt provided and --pretrained not set to fall back.")
        # Use predefined pretrained Earthformer-UNet
        earthformerunet_ckpt_path = os.path.join(
            default_pretrained_earthformerunet_dir, pretrained_sevirlr_earthformerunet_name
        )
        state_dict = torch.load(earthformerunet_ckpt_path, map_location=torch.device("cpu"))
        module.torch_nn_module.load_state_dict(state_dict=state_dict)
        return

    sd = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if isinstance(sd, dict) and any(k.startswith("torch_nn_module.") for k in sd.keys()):
        filtered = {k.replace("torch_nn_module.", ""): v for k, v in sd.items() if k.startswith("torch_nn_module.")}
        module.torch_nn_module.load_state_dict(filtered)
    else:
        # Treat as raw state_dict for torch_nn_module
        module.torch_nn_module.load_state_dict(sd)


def main():
    parser = argparse.ArgumentParser(description="Run PreDiff inference over a folder of sequences.")
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input .npy sequences")
    parser.add_argument("--pattern", default="*.npy", type=str, help="Glob for inputs inside input_dir")
    parser.add_argument("--save", default="tmp_inference", type=str, help="Experiment subfolder to save outputs")
    parser.add_argument("--cfg", default=None, type=str, help="Path to YAML config (e.g., prediff_sevirlr_v1.yaml)")
    parser.add_argument("--ckpt", default=None, type=str, help="Path to model weights (.pt or Lightning ckpt)")
    parser.add_argument("--pretrained", action="store_true", help="Download and use pretrained weights if --ckpt not set")
    parser.add_argument("--device", default="cuda", type=str, help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--max_samples", default=None, type=int, help="Limit number of files for quick runs")
    parser.add_argument("--rescale", default="none", choices=["none", "01", "255"], help="Input scaling mode")
    parser.add_argument("--save_png", action="store_true", help="Also save PNG visualizations next to outputs")
    parser.add_argument("--align", action="store_true", help="Enable knowledge alignment guidance at sampling")
    parser.add_argument("--align_mode", default="context_mean", choices=["context_mean", "last_frame_mean"],
                        help="How to estimate avg_x target for alignment when ground truth is absent")
    parser.add_argument("--align_multiplier", default=2.0, type=float,
                        help="Multiplier applied to estimated average to mimic training guidance")
    args = parser.parse_args()

    if args.pretrained:
        # Ensure required pretrained assets are present
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_vae_name,
                                    save_dir=default_pretrained_vae_dir,
                                    exist_ok=False)
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_earthformerunet_name,
                                    save_dir=default_pretrained_earthformerunet_dir,
                                    exist_ok=False)
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_alignment_name,
                                    save_dir=default_pretrained_alignment_dir,
                                    exist_ok=False)

    if args.cfg is None:
        # Default to the repository's SEVIR-LR config
        args.cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "prediff_sevirlr_v1.yaml"))

    # Build module and load weights
    module = PreDiffModuleWrapper(oc_file=args.cfg, save_dir=args.save)
    _load_torch_nn_module_weights(module, args.ckpt, use_pretrained_if_none=args.pretrained)

    # Prepare output dirs
    out_npy_dir = os.path.join(module.save_dir, "npy")
    os.makedirs(out_npy_dir, exist_ok=True)
    out_png_dir = os.path.join(module.save_dir, "examples")
    if args.save_png:
        os.makedirs(out_png_dir, exist_ok=True)

    # Device
    dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    module.eval()
    module.to(dev)
    # Ensure alignment model (if any) is on the same device
    if getattr(module, "alignment_model", None) is not None:
        module.alignment_model.to(dev)
        module.alignment_model.eval()

    files = _list_files(args.input_dir, args.pattern)
    if args.max_samples is not None:
        files = files[: args.max_samples]

    # Determine in/out lengths from config
    oc = module.oc
    in_len = int(oc.layout.in_len)
    out_len = int(oc.layout.out_len)

    # Optional visualization import
    if args.save_png:
        try:
            from prediff.datasets.sevir.visualization import vis_sevir_seq
            can_plot = True
        except Exception:
            can_plot = False
    else:
        can_plot = False

    with torch.no_grad():
        # Prefer EMA weights if enabled
        ema_ctx = module.ema_scope() if getattr(module, "use_ema", False) else torch.no_grad()
        with ema_ctx:
            for fp in files:
                base = os.path.splitext(os.path.basename(fp))[0]
                arr = np.load(fp)
                arr = _ensure_4d_thwc(arr)

                # Rescale if requested
                if args.rescale == "255":
                    arr = arr / 255.0
                # "01" assumes already 0..1; "none" does nothing

                T = arr.shape[0]
                if T < in_len:
                    raise ValueError(f"File {fp} has T={T} < in_len={in_len}")

                # Build context (and optional target)
                context = arr[:in_len]  # THWC
                has_target = T >= (in_len + out_len)
                if has_target:
                    target = arr[in_len:in_len + out_len]

                # To torch NTHWC
                y = torch.from_numpy(context[None, ...]).to(dev)
                cond = {"y": y}

                # Sample
                use_alignment = False
                alignment_kwargs = None
                if args.align and getattr(module, "alignment_fn", None) is not None:
                    # Estimate avg_x_gt from context
                    if args.align_mode == "context_mean":
                        avg_est = float(np.mean(context))
                    else:
                        avg_est = float(np.mean(context[-1]))
                    avg_target = torch.tensor([[avg_est * float(args.align_multiplier)]], dtype=torch.float32, device=dev)
                    alignment_kwargs = {"avg_x_gt": avg_target}
                    use_alignment = True

                pred = module.sample(
                    cond=cond,
                    batch_size=1,
                    use_alignment=use_alignment,
                    alignment_kwargs=alignment_kwargs,
                    return_intermediates=False,
                    verbose=False,
                ).contiguous()

                pred_np = pred[0].detach().float().cpu().numpy()  # THWC
                out_path = os.path.join(out_npy_dir, f"{base}_pred.npy")
                np.save(out_path, pred_np)

                if can_plot:
                    try:
                        label = ["context", "pred"] if not has_target else ["context", "target", "pred"]
                        seq = [context, pred_np] if not has_target else [context, target, pred_np]
                        vis_sevir_seq(
                            save_path=os.path.join(out_png_dir, f"{base}.png"),
                            seq=seq,
                            label=label,
                            interval_real_time=getattr(oc.dataset, "interval_real_time", 10),
                            plot_stride=1,
                            fs=getattr(oc.eval, "fs", 20),
                            label_offset=getattr(oc.eval, "label_offset", (-0.5, 0.5)),
                            label_avg_int=getattr(oc.eval, "label_avg_int", False),
                        )
                    except Exception:
                        # Non-fatal if plotting fails
                        pass

    print(f"Saved predictions to: {out_npy_dir}")
    if args.save_png:
        print(f"Saved PNGs to: {out_png_dir}")


if __name__ == "__main__":
    main()

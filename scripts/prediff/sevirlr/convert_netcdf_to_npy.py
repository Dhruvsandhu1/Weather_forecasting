import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch


def _choose_variable(ds) -> str:
    """Pick a data variable with >=2 dims (spatial). Prefer names containing rain/precip."""
    candidates = list(ds.data_vars)
    priority = [v for v in candidates if any(k in v.lower() for k in ["rain", "precip", "rf", "rate"])]
    for v in priority + candidates:
        da = ds[v]
        if da.ndim >= 2:
            return v
    raise ValueError("No suitable data variable found in the NetCDF file.")


def _get_dims(da) -> Tuple[str, str, str]:
    """Return (time, height, width) dim names inferred from DataArray dims."""
    dims = list(da.dims)
    # Heuristics
    time_dim = None
    for d in dims:
        if d.lower() in ["time", "t"]:
            time_dim = d
            break
    if time_dim is None and len(dims) >= 3:
        time_dim = dims[0]
    # spatial dims are the last two by default
    spatial = [d for d in dims if d != time_dim]
    if len(spatial) < 2:
        raise ValueError(f"Cannot infer spatial dims from {dims}")
    h_dim, w_dim = spatial[-2], spatial[-1]
    return time_dim, h_dim, w_dim


def _to_thw(da, time_dim: str, h_dim: str, w_dim: str) -> np.ndarray:
    arr = da.transpose(time_dim, h_dim, w_dim).values
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def _scale(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return arr
    if mode == "255":
        return arr / 255.0
    if mode == "01":
        a_min = float(np.nanmin(arr))
        a_max = float(np.nanmax(arr))
        if a_max > a_min:
            return (arr - a_min) / (a_max - a_min)
        return np.zeros_like(arr, dtype=np.float32)
    raise ValueError(f"Unknown scale mode: {mode}")


def _resize_hw(arr_thw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize THW to THW' using torch interpolate (bilinear)."""
    if arr_thw.shape[1] == target_h and arr_thw.shape[2] == target_w:
        return arr_thw
    t, h, w = arr_thw.shape
    ten = torch.from_numpy(arr_thw).unsqueeze(1)  # T 1 H W
    ten = torch.nn.functional.interpolate(
        ten, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return ten.squeeze(1).numpy()


def main():
    parser = argparse.ArgumentParser(description="Convert NetCDF variable to sliding-window THWC .npy files.")
    parser.add_argument("--nc", required=True, type=str, help="Path to NetCDF file")
    parser.add_argument("--var", default=None, type=str, help="Variable name to extract (auto if omitted)")
    parser.add_argument("--out_dir", default="finetuning_data", type=str, help="Output folder for .npy files")
    parser.add_argument("--in_len", default=7, type=int, help="Context length for inference inputs")
    parser.add_argument("--window_len", default=None, type=int, help="Total frames per file; defaults to in_len")
    parser.add_argument("--stride", default=1, type=int, help="Stride between windows along time")
    parser.add_argument("--target_h", default=128, type=int, help="Resize height")
    parser.add_argument("--target_w", default=128, type=int, help="Resize width")
    parser.add_argument("--channels", default=1, type=int, help="Replicate single channel to this count (model default: 1)")
    parser.add_argument("--scale", default="none", choices=["none", "01", "255"], help="Scaling mode")
    parser.add_argument("--temporal_factor", default=1, type=int, help="Upsample time by this integer factor (e.g., 3 for 30→10 min)")
    parser.add_argument("--start", default=0, type=int, help="Start index along time")
    parser.add_argument("--limit", default=None, type=int, help="Max number of windows to export")
    args = parser.parse_args()

    try:
        import xarray as xr
    except Exception as e:
        raise RuntimeError(
            "xarray is required. Install with: python -m pip install xarray netcdf4"
        ) from e

    ds = xr.open_dataset(args.nc)
    var = args.var or _choose_variable(ds)
    da = ds[var]
    t_dim, h_dim, w_dim = _get_dims(da)
    arr_thw = _to_thw(da, t_dim, h_dim, w_dim)  # T H W
    arr_thw = _scale(arr_thw, args.scale)
    arr_thw = _resize_hw(arr_thw, args.target_h, args.target_w)

    # Temporal upsampling (simple linear interpolation)
    if args.temporal_factor and args.temporal_factor > 1:
        f = int(args.temporal_factor)
        T, H, W = arr_thw.shape
        new_T = (T - 1) * f + 1
        out = np.empty((new_T, H, W), dtype=np.float32)
        out[0] = arr_thw[0]
        for t in range(T - 1):
            a = arr_thw[t]
            b = arr_thw[t + 1]
            for i in range(1, f + 1):
                alpha = i / float(f)
                out[t * f + i] = (1.0 - alpha) * a + alpha * b
        arr_thw = out

    T = arr_thw.shape[0]
    window_len = args.window_len or args.in_len
    os.makedirs(args.out_dir, exist_ok=True)

    count = 0
    for s in range(args.start, max(0, T - window_len + 1), args.stride):
        e = s + window_len
        if e > T:
            break
        win = arr_thw[s:e]  # WLEN H W
        # THWC with channel replication
        thwc = np.repeat(win[..., None], args.channels, axis=-1)
        out_name = f"{os.path.splitext(os.path.basename(args.nc))[0]}_t{s:05d}_len{window_len}.npy"
        np.save(os.path.join(args.out_dir, out_name), thwc.astype(np.float32))
        count += 1
        if args.limit is not None and count >= args.limit:
            break

    print(f"Exported {count} files to {args.out_dir} using var '{var}'.")


if __name__ == "__main__":
    main()

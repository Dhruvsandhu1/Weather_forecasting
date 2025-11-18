import argparse
import os
import csv
from typing import Optional, Tuple, List

import numpy as np
import torch
import h5py
from datetime import datetime, timedelta

try:
    # Prefer project-defined default path if package is installed
    from prediff.utils.path import default_dataset_sevirlr_dir  # type: ignore
except Exception:
    # Fallback to local datasets/sevirlr under current working directory
    default_dataset_sevirlr_dir = os.path.join(os.getcwd(), "datasets", "sevirlr")


def _choose_variable(ds) -> str:
    candidates = list(ds.data_vars)
    priority = [v for v in candidates if any(k in v.lower() for k in ["vil", "rain", "precip", "rf", "rate"]) ]
    for v in priority + candidates:
        da = ds[v]
        if da.ndim >= 2:
            return v
    raise ValueError("No suitable data variable found in the NetCDF file.")


def _get_dims(da) -> Tuple[Optional[str], str, str]:
    dims = list(da.dims)
    t_dim = None
    for d in dims:
        if d.lower() in ["time", "t"]:
            t_dim = d
            break
    if t_dim is None and len(dims) >= 3:
        t_dim = dims[0]
    spatial = [d for d in dims if d != t_dim]
    if len(spatial) < 2:
        raise ValueError(f"Cannot infer spatial dims from {dims}")
    h_dim, w_dim = spatial[-2], spatial[-1]
    return t_dim, h_dim, w_dim


def _to_thw(da, time_dim: Optional[str], h_dim: str, w_dim: str) -> np.ndarray:
    if time_dim is None:
        arr = da.transpose(h_dim, w_dim).values[None, ...]
    else:
        arr = da.transpose(time_dim, h_dim, w_dim).values
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def _resize_hw(arr_thw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if arr_thw.shape[1] == target_h and arr_thw.shape[2] == target_w:
        return arr_thw
    ten = torch.from_numpy(arr_thw).unsqueeze(1)  # T 1 H W
    ten = torch.nn.functional.interpolate(ten, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return ten.squeeze(1).numpy()


def _temporal_upsample(arr_thw: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr_thw
    T, H, W = arr_thw.shape
    new_T = (T - 1) * factor + 1
    out = np.empty((new_T, H, W), dtype=np.float32)
    out[0] = arr_thw[0]
    for t in range(T - 1):
        a = arr_thw[t]
        b = arr_thw[t + 1]
        for i in range(1, factor + 1):
            alpha = i / float(factor)
            out[t * factor + i] = (1.0 - alpha) * a + alpha * b
    return out


def _scale_to_uint8(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        pass
    elif mode == "01":
        a_min = float(np.nanmin(arr))
        a_max = float(np.nanmax(arr))
        if a_max > a_min:
            arr = (arr - a_min) / (a_max - a_min)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    elif mode == "255":
        arr = arr / 255.0
    else:
        raise ValueError(f"Unknown scale mode: {mode}")
    arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    return arr


def _ensure_dirs(base_out: str):
    cat_path = os.path.join(base_out, "CATALOG.csv")
    data_dir = os.path.join(base_out, "data", "vil")
    os.makedirs(data_dir, exist_ok=True)
    return cat_path, data_dir


def _write_h5(file_path: str, data_nhwt_uint8: np.ndarray):
    # data_nhwt_uint8: shape (N, H, W, T)
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('vil', data=data_nhwt_uint8, maxshape=(None,) + data_nhwt_uint8.shape[1:])


def _write_catalog(cat_path: str, rows: List[dict]):
    fieldnames = ["id", "img_type", "file_name", "file_index", "time_utc", "pct_missing"]
    with open(cat_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Build a SEVIR-LR compatible dataset from a NetCDF file.")
    parser.add_argument("--nc", required=True, type=str, help="Path to NetCDF file")
    parser.add_argument("--var", default=None, type=str, help="Variable name to extract")
    parser.add_argument("--out_dir", default=None, type=str, help="Target dataset root (defaults to datasets/sevirlr)")
    parser.add_argument("--target_h", default=128, type=int, help="Resize height")
    parser.add_argument("--target_w", default=128, type=int, help="Resize width")
    parser.add_argument("--temporal_factor", default=3, type=int, help="Upsample time factor (e.g., 3 for 30→10min)")
    parser.add_argument("--window_len", default=25, type=int, help="Raw sequence length per event (SEVIR-LR=25)")
    parser.add_argument("--stride", default=25, type=int, help="Stride between events (use 25 to avoid overlaps)")
    parser.add_argument("--scale", default="01", choices=["none", "01", "255"], help="Scaling before uint8")
    parser.add_argument("--h5_name", default="custom.h5", type=str, help="Output HDF5 file name under data/vil")
    parser.add_argument("--start_datetime", default="2023-07-07T00:00:00", type=str, help="Base time_utc for events if source lacks time")
    parser.add_argument("--limit", default=None, type=int, help="Max events to create")
    args = parser.parse_args()

    try:
        import xarray as xr
    except Exception as e:
        raise RuntimeError("xarray is required. Install with: python -m pip install xarray netcdf4") from e

    out_root = args.out_dir if args.out_dir is not None else default_dataset_sevirlr_dir
    cat_path, data_dir = _ensure_dirs(out_root)
    h5_rel = os.path.join("vil", args.h5_name)
    h5_path = os.path.join(out_root, "data", h5_rel)
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    ds = xr.open_dataset(args.nc)
    var = args.var or _choose_variable(ds)
    da = ds[var]
    t_dim, h_dim, w_dim = _get_dims(da)
    arr_thw = _to_thw(da, t_dim, h_dim, w_dim)  # T H W
    arr_thw = _resize_hw(arr_thw, args.target_h, args.target_w)
    arr_thw = _temporal_upsample(arr_thw, args.temporal_factor)
    arr_thw_u8 = _scale_to_uint8(arr_thw, args.scale)  # T H W -> uint8 (0..255)

    T = arr_thw_u8.shape[0]
    win = int(args.window_len)
    stride = int(args.stride)
    windows = []
    time_base = None
    has_time = (t_dim is not None) and ("time" in ds.coords or t_dim in ds.coords)
    time_var_name = None
    if has_time:
        if "time" in ds:
            time_var_name = "time"
        elif t_dim in ds:
            time_var_name = t_dim
    base_dt = datetime.fromisoformat(args.start_datetime)
    event_rows = []
    idx = 0
    for s in range(0, max(0, T - win + 1), stride):
        e = s + win
        if e > T:
            break
        seq = arr_thw_u8[s:e]  # THW
        windows.append(np.transpose(seq, (1, 2, 0)))  # H W T
        # time_utc for event: derive from coordinate if available else synthetic
        if has_time and time_var_name is not None:
            try:
                # Map upsampled frame index back to original time index
                orig_len = ds[time_var_name].shape[0]
                if args.temporal_factor > 1:
                    base_idx = min(int(s // args.temporal_factor), orig_len - 1)
                else:
                    base_idx = min(s, orig_len - 1)
                t0 = ds[time_var_name].values[base_idx]
                time_str = np.datetime_as_string(t0, unit='s')
            except Exception:
                # Fallback to synthetic time if anything goes wrong
                time_str = (base_dt + timedelta(minutes=10 * s)).isoformat()
        else:
            time_str = (base_dt + timedelta(minutes=10 * s)).isoformat()
        event_rows.append({
            "id": f"E{idx:08d}",
            "img_type": "vil",
            "file_name": h5_rel.replace("\\", "/"),
            "file_index": idx,
            "time_utc": time_str,
            "pct_missing": 0,
        })
        idx += 1
        if args.limit is not None and idx >= args.limit:
            break

    if len(windows) == 0:
        raise RuntimeError("No windows were created; check --window_len, --stride and temporal upsampling.")

    data_nhwt = np.stack(windows, axis=0)  # N H W T
    _write_h5(h5_path, data_nhwt)
    _write_catalog(cat_path, event_rows)

    print(f"Wrote HDF5: {h5_path} [shape={data_nhwt.shape}, dtype=uint8]")
    print(f"Wrote catalog: {cat_path} [rows={len(event_rows)}]")
    print(f"SEVIR-LR dataset ready at: {out_root}")


if __name__ == "__main__":
    main()

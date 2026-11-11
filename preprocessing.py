"""Preprocesamiento DTEASN + voxelización de eventos (intacto)."""
import numpy as np
import torch

from config import Config


def compute_event_attention(events: np.ndarray, cfg: Config) -> np.ndarray:
    if not cfg.use_attention:
        return np.ones(len(events), dtype=np.float32)

    K = cfg.K_neighbors
    tau_t = cfg.tau_time_ms * 1000.0
    tau_s = cfg.tau_space_px
    lam = cfg.lambda_pol
    alpha, beta, gamma = cfg.alpha, cfg.beta, cfg.gamma

    N = len(events)
    if N < 2:
        return np.ones(N, dtype=np.float32)

    order = np.argsort(events["t"])
    t = events["t"][order].astype(np.float64)
    x = events["x"][order].astype(np.float64)
    y = events["y"][order].astype(np.float64)
    p = events["p"][order].astype(np.int64)

    attn_sum = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.float64)
    idx = np.arange(N)

    for k in range(1, K + 1):
        valid = (idx + k) < N
        i_idx = idx[valid]
        j_idx = i_idx + k
        dt = np.abs(t[j_idx] - t[i_idx])
        dist = np.sqrt((x[j_idx] - x[i_idx]) ** 2 + (y[j_idx] - y[i_idx]) ** 2)
        same = p[j_idx] == p[i_idx]
        pair = (alpha * np.exp(-dt / tau_t)
                + beta * np.exp(-dist / tau_s)
                + gamma * np.where(same, 1.0, lam))
        attn_sum[i_idx] += pair
        attn_sum[j_idx] += pair
        cnt[i_idx] += 1
        cnt[j_idx] += 1

    cnt = np.maximum(cnt, 1.0)
    attn = attn_sum / cnt
    m, s = attn.mean(), attn.std() + 1e-6
    attn = 1.0 + 0.3 * np.clip((attn - m) / s, -2.0, 2.0)
    attn = np.clip(attn, cfg.attn_clip_min, cfg.attn_clip_max)

    out = np.zeros(N, dtype=np.float32)
    out[order] = attn.astype(np.float32)
    return out


def create_frame_sequence(events: np.ndarray, cfg: Config) -> torch.Tensor:
    H, W, C = cfg.sensor_size
    T = cfg.n_time_bins
    max_count = cfg.voxel_max_count

    t = events["t"].astype(np.int64)
    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    p = np.clip(events["p"].astype(np.int64), 0, C - 1)

    w = (compute_event_attention(events, cfg) if cfg.use_attention
         else np.ones(len(events), dtype=np.float32))

    t_min, t_max = int(t.min()), int(t.max())
    dt = max((t_max - t_min) // T, 1)
    frames = np.zeros((T, C, H, W), dtype=np.float32)

    for b in range(T):
        t0 = t_min + b * dt
        t1 = t0 + dt if b < T - 1 else t_max + 1
        mask = (t >= t0) & (t < t1)
        if not mask.any():
            continue
        xw, yw, pw, ww = x[mask], y[mask], p[mask], w[mask]
        valid = (xw >= 0) & (xw < W) & (yw >= 0) & (yw < H)
        if not valid.any():
            continue
        np.add.at(frames[b], (pw[valid], yw[valid], xw[valid]),
                  ww[valid].astype(np.float32))

    frames = np.clip(frames, 0.0, float(max_count)) / float(max_count)
    return torch.from_numpy(frames).float()
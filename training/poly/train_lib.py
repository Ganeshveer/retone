"""Model, data, and config for polyphonic instrument conversion (Stage 2).

Piano roll -> mel spectrogram, one model per target instrument. The mel is decoded
to waveform by a FROZEN pretrained BigVGAN, so this model only ever learns the
note-events -> timbre mapping.
"""
import os, sys, math, random, pathlib
import numpy as np, torch, torch.nn as nn

INSTRUMENT = os.environ.get("RETONE_INSTRUMENT", "piano")
ROOT       = pathlib.Path("/workspace/retone_poly")
CACHE_DIR  = ROOT / "cache" / INSTRUMENT
CKPT_DIR   = ROOT / "ckpt" / INSTRUMENT
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def _cpu_quota():
    """Real core budget. nproc reports the HOST's cores (96 here) but the pod's
    cgroup quota is 7.65 — sizing workers from nproc starves the GPU."""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(float(q) / float(p)))
    except Exception:
        pass
    for base in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
        try:
            q = int(open(base + "/cpu.cfs_quota_us").read())
            p = int(open(base + "/cpu.cfs_period_us").read())
            if q > 0:
                return max(1, int(q / p))
        except Exception:
            pass
    return os.cpu_count() or 4


_CORES = _cpu_quota()

CFG = dict(
    sample_rate=44100, n_fft=2048, hop_length=512, win_length=2048,
    n_mels=128, fmin=0, fmax=None,
    n_pitches=128, n_cond_ch=3,
    # ~57M params: inside the published 20-60M sweet spot for a single instrument
    hidden=768, n_layers=8, n_heads=12, dropout=0.1,
    seq_frames=512,          # ~5.9 s per example at 86.13 fps
    batch_size=64,           # A40 48GB
    lr=3e-4, max_steps=200_000, warmup=2_000,
    val_every=2_000, save_every=2_500,
    num_workers=max(2, _CORES - 2),
    seed=940513,
    amp_dtype="bf16",        # Ampere: native bf16, no GradScaler needed
    compile=True, tf32=True, prefetch=4,
)
CFG["frame_rate"] = CFG["sample_rate"] / CFG["hop_length"]

MEL_MEAN, MEL_STD = -6.0, 2.5    # log-mel normalization; inverted before vocoding

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])


# ───────────────────────────── data ─────────────────────────────
from torch.utils.data import Dataset, DataLoader

class PairDataset(Dataset):
    def __init__(self, cache_dir, seq=CFG["seq_frames"]):
        self.files = sorted(pathlib.Path(cache_dir).glob("pair_*.npz"))
        self.seq = seq
        if not self.files:
            raise RuntimeError("no cached pairs in %s" % cache_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        roll, mel = d["roll"].astype(np.float32), d["mel"].astype(np.float32)
        T = min(roll.shape[2], mel.shape[1])
        # Random crop offset. Never crop on note boundaries, or the model only ever
        # sees clean attacks and never learns to continue an already-sounding note.
        s = random.randint(0, max(0, T - self.seq))
        roll, mel = roll[:, :, s:s + self.seq], mel[:, s:s + self.seq]
        if roll.shape[2] < self.seq:
            pad = self.seq - roll.shape[2]
            roll = np.pad(roll, ((0, 0), (0, 0), (0, pad)))
            mel = np.pad(mel, ((0, 0), (0, pad)), constant_values=MEL_MEAN - 2 * MEL_STD)
        return torch.from_numpy(roll), torch.from_numpy((mel - MEL_MEAN) / MEL_STD)


def make_loaders(cache_dir=None, val_frac=0.05):
    ds = PairDataset(cache_dir or CACHE_DIR)
    n_val = max(1, int(len(ds) * val_frac))
    tr, va = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(CFG["seed"]))
    nw = CFG["num_workers"]
    kw = dict(batch_size=CFG["batch_size"], num_workers=nw, pin_memory=True,
              persistent_workers=nw > 0)
    if nw > 0:
        kw["prefetch_factor"] = CFG["prefetch"]
    return (DataLoader(tr, shuffle=True, drop_last=True, **kw),
            DataLoader(va, shuffle=False, **kw))


# ───────────────────────────── model ─────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PianoRollToMel(nn.Module):
    """(B, 3, 128, T) conditioning -> (B, 128, T) log-mel. One per instrument."""

    def __init__(self, cfg=CFG):
        super().__init__()
        d_in, h = cfg["n_cond_ch"] * cfg["n_pitches"], cfg["hidden"]
        # Local temporal context before attention. Cheap, and sharpens attacks.
        self.stem = nn.Sequential(
            nn.Conv1d(d_in, h, 5, padding=2), nn.GELU(),
            nn.Conv1d(h, h, 5, padding=2), nn.GELU(),
        )
        self.pos = PositionalEncoding(h)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h, nhead=cfg["n_heads"], dim_feedforward=h * 4,
                dropout=cfg["dropout"], batch_first=True, norm_first=True,
                activation="gelu"),
            num_layers=cfg["n_layers"])
        self.head = nn.Sequential(
            nn.Conv1d(h, h, 5, padding=2), nn.GELU(),
            nn.Conv1d(h, cfg["n_mels"], 1),
        )

    def forward(self, roll):
        B, C, P, T = roll.shape
        x = self.stem(roll.reshape(B, C * P, T))
        x = self.encoder(self.pos(x.transpose(1, 2)))
        return self.head(x.transpose(1, 2))


def build_model(cfg=CFG, device="cuda"):
    m = PianoRollToMel(cfg).to(device)
    print("model: %.1fM params" % (sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
    return m


def mel_loss(pred, target):
    """L1 on log-mel, plus a delta term. L1 (not L2) because it is less dominated by
    the loud low bins and gives sharper spectrograms. The delta term weights
    frame-to-frame CHANGE, which is where attacks live -- they matter perceptually far
    more than their pixel count suggests."""
    l1 = torch.nn.functional.l1_loss(pred, target)
    d = torch.nn.functional.l1_loss(pred[..., 1:] - pred[..., :-1],
                                    target[..., 1:] - target[..., :-1])
    return l1 + 0.5 * d

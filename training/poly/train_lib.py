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


def _mem_limit_gb():
    """Real memory ceiling for THIS pod. free/psutil report the HOST's RAM (503 GB
    here) while the cgroup allows 50 GB — the same host-vs-pod trap as nproc."""
    for f in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v != "max":
                b = int(v)
                if b < (1 << 62):          # sentinel for "unlimited"
                    return b / 1e9
        except Exception:
            pass
    try:
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return 8.0


_CORES = _cpu_quota()
_MEM_GB = _mem_limit_gb()

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
    num_workers=int(os.environ.get("RETONE_WORKERS", 0)),
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
    """All pairs held in RAM.

    The .npz files are ~3 MB each decompressed; decompressing 64 of them per batch
    made us completely dataloader-bound (0.3 it/s with the GPU at 0%). The whole
    corpus is only ~1.2 GB and the pod has 503 GB, so we load once and slice.

    Each file is also worth MANY training crops, not one: a 60 s file holds ~10
    non-overlapping 5.9 s windows. Yielding one crop per file made an "epoch" just
    3 batches, so the loader spent its life at epoch boundaries.
    """

    def __init__(self, cache_dir, seq=CFG["seq_frames"], crops_per_file=None, augment=False):
        files = sorted(pathlib.Path(cache_dir).glob("pair_*.npz"))
        if not files:
            raise RuntimeError("no cached pairs in %s" % cache_dir)
        self.seq = seq
        self.augment = augment

        # Estimate the footprint before committing to a preload. Exceeding the pod's
        # cgroup limit gets the process OOM-killed with no useful traceback.
        probe = np.load(files[0])
        per_file_mb = (probe["roll"].nbytes + probe["mel"].nbytes) / 1e6
        est_gb = per_file_mb * len(files) / 1000
        budget_gb = _MEM_GB * 0.5          # leave room for torch, CUDA context, compile cache
        if est_gb > budget_gb:
            raise RuntimeError(
                "dataset needs ~%.1f GB but only ~%.1f GB of the pod's %.0f GB is safe to use. "
                "Either shard the cache, shorten `seconds` in dataprep, or switch to a "
                "memmap/lazy loader." % (est_gb, budget_gb, _MEM_GB))

        self.rolls, self.mels = [], []
        for f in files:
            d = np.load(f)
            roll, mel = d["roll"], d["mel"]           # keep fp16 in RAM
            T = min(roll.shape[2], mel.shape[1])
            if T < seq:
                continue
            self.rolls.append(np.ascontiguousarray(roll[:, :, :T]))
            self.mels.append(np.ascontiguousarray(mel[:, :T]))
        if not self.rolls:
            raise RuntimeError("every pair in %s was shorter than %d frames" % (cache_dir, seq))
        med = int(np.median([r.shape[2] for r in self.rolls]))
        self.crops = crops_per_file or max(1, med // seq)
        mb = sum(r.nbytes + m.nbytes for r, m in zip(self.rolls, self.mels)) / 1e6
        print("  dataset: %d files, %.0f MB in RAM (%.0f GB pod limit), %d crops/file "
              "-> %d samples/epoch"
              % (len(self.rolls), mb, _MEM_GB, self.crops,
                 len(self.rolls) * self.crops), flush=True)

    def __len__(self):
        return len(self.rolls) * self.crops

    def __getitem__(self, i):
        j = i // self.crops
        roll, mel = self.rolls[j], self.mels[j]
        T = roll.shape[2]
        # Random offset, never aligned to note boundaries.
        s = random.randint(0, max(0, T - self.seq))
        r = roll[:, :, s:s + self.seq].astype(np.float32)
        m = mel[:, s:s + self.seq].astype(np.float32)
        if r.shape[2] < self.seq:
            pad = self.seq - r.shape[2]
            r = np.pad(r, ((0, 0), (0, 0), (0, pad)))
            m = np.pad(m, ((0, 0), (0, pad)), constant_values=MEL_MEAN - 2 * MEL_STD)

        if self.augment:
            m, r = self._augment(m, r)

        return torch.from_numpy(r), torch.from_numpy((m - MEL_MEAN) / MEL_STD)

    def _augment(self, m, r):
        """Break the determinism of soundfont rendering.

        audio = FluidR3(MIDI) is an EXACT function: same note+velocity gives
        byte-identical samples. A model trained on that learns to imitate one
        soundfont rather than to generalize, and the loss looks great while the
        model is quietly narrow. These perturbations are all things that vary in
        real recordings but never vary in our renders.

        Mel is log-magnitude, so a gain change is an ADDITIVE offset, not a multiply.
        """
        # overall level (log domain: +-3 dB-ish)
        m = m + np.float32(random.uniform(-0.35, 0.35))

        # spectral tilt — stands in for mic/room/EQ differences
        if random.random() < 0.6:
            tilt = np.linspace(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3),
                               m.shape[0], dtype=np.float32)
            m = m + tilt[:, None]

        # noise floor — real recordings have one, soundfont renders do not
        if random.random() < 0.5:
            m = m + np.random.randn(*m.shape).astype(np.float32) * random.uniform(0.01, 0.06)

        # SpecAugment-style masking: forces reliance on the CONDITIONING rather than
        # on neighbouring mel context, which is what we actually want at inference.
        if random.random() < 0.3:
            f = random.randint(1, 8); f0 = random.randint(0, max(1, m.shape[0] - f))
            m[f0:f0 + f, :] = MEL_MEAN - 2 * MEL_STD

        # velocity jitter on the CONDITIONING, so the model does not treat the
        # soundfont's exact velocity->timbre curve as gospel
        if random.random() < 0.4:
            v = r[2] > 0
            r[2][v] = np.clip(r[2][v] * random.uniform(0.85, 1.15), 0.01, 1.0)

        return m, r


def make_loaders(cache_dir=None, val_frac=0.05):
    # Two views of the same cache: augmented for train, clean for val.
    # Validating on augmented data would measure the wrong thing.
    ds_tr = PairDataset(cache_dir or CACHE_DIR, augment=True)
    ds_va = PairDataset(cache_dir or CACHE_DIR, augment=False)
    n = len(ds_tr)
    n_val = max(1, int(n * val_frac))
    g = torch.Generator().manual_seed(CFG["seed"])
    perm = torch.randperm(n, generator=g).tolist()
    tr = torch.utils.data.Subset(ds_tr, perm[n_val:])
    va = torch.utils.data.Subset(ds_va, perm[:n_val])
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

"""Build-time model prefetch.

Run during `docker build` so model weights are baked into the image — then a cold start is
just weights -> VRAM (no network download at runtime), which FlashBoot can snapshot.

Best-effort: if a model download fails at build (e.g. registry hiccup), the build still
succeeds and the handler downloads the model lazily on first use.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from separator_engine import load_model_config  # noqa: E402


def main() -> None:
    cfg = load_model_config()
    tiers = [t for t in ("2stem", "4stem", "6stem") if t in cfg]
    try:
        from audio_separator.separator import Separator

        model_dir = str(Path(__file__).parent / "models")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        sep = Separator(model_file_dir=model_dir, output_dir="/tmp/dl")
        for tier in tiers:
            model = cfg[tier]["model"]
            try:
                print(f"[prefetch] {tier} -> {model}", flush=True)
                sep.load_model(model_filename=model)
            except Exception as exc:
                print(f"[prefetch] WARN {tier} {model}: {exc}", flush=True)
    except Exception as exc:
        print(f"[prefetch] skipped ({exc}); models will download at runtime", flush=True)


if __name__ == "__main__":
    main()

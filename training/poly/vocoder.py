"""BigVGAN loader that bypasses the HF PyTorchModelHubMixin.

BigVGAN._from_pretrained() is declared with keyword-only "proxies" and
"resume_download" args. huggingface_hub >= 1.x no longer passes those, so
BigVGAN.from_pretrained() raises TypeError. Rather than pin an old hub version
(which drags in other conflicts), we replicate what the mixin does: fetch
config.json + the generator weights, build the model, load the state dict.
"""
import json, os, sys, torch

BIGVGAN_DIR = "/workspace/retone_poly/BigVGAN"
if BIGVGAN_DIR not in sys.path:
    sys.path.insert(0, BIGVGAN_DIR)


def load_bigvgan(model_id="nvidia/bigvgan_v2_44khz_128band_512x",
                 device="cuda", use_cuda_kernel=False):
    """Return a frozen, eval-mode BigVGAN generator with .h config attached."""
    from huggingface_hub import hf_hub_download
    import bigvgan as bigvgan_mod
    from env import AttrDict

    cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(cfg_path) as f:
        h = AttrDict(json.load(f))

    model = bigvgan_mod.BigVGAN(h, use_cuda_kernel=use_cuda_kernel)

    ckpt_path = hf_hub_download(repo_id=model_id, filename="bigvgan_generator.pt")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd["generator"] if "generator" in sd else sd, strict=False)

    model.remove_weight_norm()
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.h = h
    return model

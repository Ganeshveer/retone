"""Train one polyphonic instrument renderer. Detached-friendly, resumable."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/workspace/retone_poly")
from train_lib import CFG, make_loaders, build_model, mel_loss, CACHE_DIR, CKPT_DIR, INSTRUMENT


def unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m


def main():
    dev = "cuda"
    if CFG["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = make_loaders()
    print("instrument=%s  train_batches=%d" % (INSTRUMENT, len(train_loader)), flush=True)

    model = build_model(device=dev)
    if CFG["compile"]:
        try:
            model = torch.compile(model)
            print("torch.compile ON", flush=True)
        except Exception as e:
            print("compile unavailable (%s) - eager" % e, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-2,
                            betas=(0.9, 0.99), fused=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=CFG["lr"], total_steps=CFG["max_steps"],
        pct_start=CFG["warmup"] / CFG["max_steps"], anneal_strategy="cos")
    # bf16 on Ampere needs no GradScaler: wider exponent range than fp16, so no
    # loss-scaling dance and no overflow babysitting.
    amp = torch.bfloat16 if CFG["amp_dtype"] == "bf16" else torch.float16

    step, best = 0, float("inf")
    last = CKPT_DIR / "latest.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev)
        unwrap(model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step = ck["step"]
        print("resumed @ %d" % step, flush=True)
    else:
        # No latest.pt — fall back to warm-starting from best.pt if present.
        # best.pt has only {model, cfg, step, val}, no optimizer/scheduler state,
        # so we load model weights only and start opt/sched fresh at step 0.
        # Enables per-instrument fine-tuning: seed cache/violin/ckpt/best.pt
        # from a related model (e.g. the strings ensemble) and this branch
        # picks it up automatically.
        warm = CKPT_DIR / "best.pt"
        if warm.exists():
            try:
                ck = torch.load(warm, map_location=dev, weights_only=True)
                unwrap(model).load_state_dict(ck["model"])
                src_step = ck.get("step", "?")
                src_val = ck.get("val", None)
                val_str = f"{src_val:.4f}" if isinstance(src_val, float) else "n/a"
                print("warm-start from best.pt (source step=%s val=%s) — fresh opt/sched, step reset to 0"
                      % (src_step, val_str), flush=True)
            except Exception as e:
                print("warm-start FAILED: %s — starting fully fresh" % e, flush=True)
    # Seed `best` from the existing best.pt or the first val after resume/warm-start
    # is auto-crowned and overwrites a genuinely-better checkpoint.
    best_ckpt = CKPT_DIR / "best.pt"
    if best_ckpt.exists():
        try:
            prev = torch.load(best_ckpt, map_location="cpu", weights_only=True)
            if isinstance(prev, dict) and "val" in prev:
                best = float(prev["val"])
                print("resumed best-val=%.4f from best.pt" % best, flush=True)
        except Exception as e:
            print("could not read prior best.pt val: %s" % e, flush=True)

    # Rotator support: stop this run after N additional steps for round-robin
    # training across instruments. Overall max_steps still governs total budget.
    steps_this_run = int(os.environ.get("RETONE_MAX_STEPS_THIS_RUN", 0))
    stop_at = (step + steps_this_run) if steps_this_run > 0 else CFG["max_steps"]

    model.train(); t0 = time.time(); tlast = t0
    while step < CFG["max_steps"] and step < stop_at:
        for roll, mel in train_loader:
            if step >= CFG["max_steps"] or step >= stop_at:
                break
            roll = roll.to(dev, non_blocking=True); mel = mel.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=amp):
                loss = mel_loss(model(roll), mel)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1

            if step % 100 == 0:
                ips = 100 / (time.time() - tlast); tlast = time.time()
                print("step %d  loss %.4f  lr %.2e  %.1f it/s  eta %.1fh  elapsed %.2fh" % (
                    step, loss.item(), sched.get_last_lr()[0], ips,
                    (CFG["max_steps"] - step) / max(ips, 1e-9) / 3600,
                    (time.time() - t0) / 3600), flush=True)

            if step % CFG["val_every"] == 0:
                model.eval(); vs = []
                with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                    for vr, vm in val_loader:
                        vs.append(mel_loss(model(vr.to(dev)), vm.to(dev)).item())
                v = float(np.mean(vs)); model.train()
                print("  VAL %d  loss %.4f" % (step, v), flush=True)
                if v < best:
                    best = v
                    torch.save({"model": unwrap(model).state_dict(), "cfg": CFG,
                                "step": step, "val": v}, CKPT_DIR / "best.pt")
                    print("  new best %.4f" % v, flush=True)

            if step % CFG["save_every"] == 0:
                sd = unwrap(model).state_dict()
                torch.save({"model": sd, "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step}, CKPT_DIR / "latest.pt")
                # Timestamped snapshot so a mid-run listen is always possible.
                torch.save({"model": sd, "cfg": CFG, "step": step},
                           CKPT_DIR / ("step_%d.pt" % step))

    torch.save({"model": unwrap(model).state_dict(), "cfg": CFG, "step": step},
               CKPT_DIR / "final.pt")
    print("COMPLETE in %.2fh" % ((time.time() - t0) / 3600), flush=True)


if __name__ == "__main__":
    main()

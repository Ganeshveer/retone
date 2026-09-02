# ReTone Direct Instrument Conversion — quick reference

Change-instrument pipeline that renders audio → MIDI → arranged MIDI → SF2
synthesis → WAV. No neural network at inference; only the transcriber is a
model. See [`../../notebooks/direct_instrument_conversion.ipynb`](../../notebooks/direct_instrument_conversion.ipynb)
for the full walkthrough and the ML-approach postmortem.

## Files

| File | Role |
|---|---|
| `instruments.py`             | Catalog of 65 instruments across 10 categories (piano, keys, organ, guitar, bass, strings, brass, wind, voice, harp, synth). Data-only — add your own by appending to `_ENTRIES`. |
| `arrange.py`                 | Per-target MIDI transforms: `piano_sustain` (extend held notes), `strings` (sostenuto extension), `none` (passthrough for pizz/staccato). |
| `render_direct.py`           | Main entrypoint. CLI + reusable `render_one()` API + transcriber dispatch (`basic_pitch`, `bytedance_piano`, `transkun`). |
| `render_multi_song.py`       | Batch: several source songs × the same target set. Auto-picks transcriber per source. |
| `render_transcriber_ab.py`   | Same source, same target, different transcriber — for A/B'ing which transcriber to standardize on. |
| `render_variety.py`          | Wider variety batch: 8 sources × 12 diverse targets, transcription cached per source (12× fewer transcribe calls than naïve `render_one` loops). Reads/adapts from anywhere — set `SOURCES` / `TARGETS` at the top. |

## Install

```bash
# System (Ubuntu / Debian / RunPod pod)
apt-get install -y fluidsynth fluid-soundfont-gm ffmpeg unzip

# System (macOS)
brew install fluidsynth ffmpeg
# then download FluidR3_GM.sf2 (150 MB, MIT-licensed) from
# https://musical-artifacts.com/artifacts/738 and point instruments.py at it

# Python — always
pip install librosa soundfile pretty_midi scipy basic-pitch

# Piano-specific transcribers (recommend both — they're cheap installs)
pip install piano_transcription_inference    # F1 96.72, real velocity + pedal
pip install transkun                          # F1 ~97.5, best on clean piano
```

**Sonatina Symphonic Orchestra** (500 MB, unlocks 30 real-sampled orchestral
instruments) — one-time setup:

```bash
mkdir -p /workspace/sf2 && cd /workspace/sf2
curl -sL -o sonatina.zip \
  "https://archive.org/download/SonatinaSymphonicOrchestraSF2/Sonatina%20Symphonic%20Orchestra%20SF2.zip"
unzip -q sonatina.zip && rm sonatina.zip
```

Edit `SONATINA_DIR` in `instruments.py` if you use a different path.

## Run — single render

```bash
python render_direct.py \
    --input   /path/to/song.wav \
    --instrument cello_sustain \
    --transcriber bytedance_piano \
    --out     /tmp/song_as_cello.wav
```

Or as a library:

```python
from render_direct import render_one
render_one(
    input_audio="/path/to/song.wav",
    instrument_name="cello_sustain",
    out_wav="/tmp/song_as_cello.wav",
    transcriber="bytedance_piano",   # or 'basic_pitch' | 'transkun'
    seconds=15,                       # clip to first N s (optional)
    reverb_wet=0.15,                  # 0.0 = dry, 0.3 = wet
)
```

## Monophonic targets and accompaniment (Aug 2026)

A real violin cannot bow six pitches at once. When you route polyphonic
material into a monophonic target (`violin_solo`, `trumpet_solo`,
`flute_solo`, sax, …), the pipeline splits into:

- **Lead**  — top pitch of each onset cluster (skyline) → target instrument.
- **Accompaniment** — everything else → a complementary polyphonic
  instrument (harp / piano / guitar / organ, mixed ≈ -5 dB under the lead).

Auto-pick uses a musical "color wheel": prefer **opposite attack** (percussive
under sustained, or vice-versa) and **different family**, with wide-register
instruments (piano, harp, organ) favored. Override with `--accompaniment`:

```bash
python render_direct.py --input song.wav --instrument violin_solo \
    --accompaniment guitar_nylon --out out.wav        # user override
python render_direct.py --input song.wav --instrument violin_solo \
    --accompaniment none --out out.wav                # no accompaniment
```

If the source is already monophonic (a vocal, a solo flute) the split is
skipped and any `--accompaniment` is silently ignored — there's no
polyphonic content to route.

## Density limiting for plucked / decaying targets

Harp, guitar, harpsichord, marimba, celesta, and pizzicato patches all carry
a `min_ioi_s` in `instruments.py`: a per-pitch minimum inter-onset gap. A
real harpist doesn't strike the same string 15×/second; the renderer thins
retriggered notes before FluidSynth sees them, and a 1-second sliding
window also caps global onset rate at 14/sec (Fletcher-Rossing perceptual
continuous-tone threshold). Values live in the catalog — no code changes to
retune.

## Run — batch (a song through every instrument in a category)

```bash
python render_direct.py --input song.wav --category strings --outdir /tmp/strings/
python render_direct.py --list                                     # browse the catalog
```

## Run — bundled batches

```bash
# 6 songs × 6 targets = 42 renders. Edit SOURCES/TARGETS in the script for your set.
python render_multi_song.py

# Transcriber A/B on 3 piano songs × 3 transcribers × 6 targets = 54 renders.
python render_transcriber_ab.py

# Variety pack: 8 popular instrumentals × 12 targets across all families = 104 renders.
# Uses cached transcription per source — one transcribe call reused across 12 targets.
python render_variety.py
```

## Transcriber cheatsheet

| Transcriber        | F1 (MAESTRO) | Best for                | Notes |
|--------------------|-------------|-------------------------|-------|
| `basic_pitch`      | ~85         | any polyphonic          | always available, tiny weights, ~1× real-time on CPU |
| `bytedance_piano`  | 96.72       | piano                   | real velocity + pedal, ~2× real-time on GPU |
| `transkun`         | ~97.5       | piano (clean studio)    | slowest (~15× real-time on CPU), WAV input only |

## Add a new instrument

1. Drop the SF2 file on disk.
2. Append one row to `instruments.py::_ENTRIES`:
   ```python
   ("my_koto",  "Koto (My SF2)",  "strings",
     "/path/to/koto.sf2",  0,  "strings"),
   ```
3. Confirm: `python render_direct.py --list | grep my_koto`.

No retrain, no cache rebuild — the next `render_one` call sees it.

## Roadmap

Ranked next steps are in the notebook (§11). Highlights: `mt3-infer` for
non-piano polyphonic sources, real-audio velocity envelope for
Basic Pitch, real convolution reverb IRs, sfizz backend for SFZ libraries.

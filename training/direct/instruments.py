"""Instrument catalog for the DIRECT pipeline.

Each instrument is a target the pipeline can render into: a soundfont path,
a program number to select inside that soundfont, and which per-target
arrangement stage to apply (piano_sustain / strings / none).

All FluidR3 GM instruments share one SF2 (128 GM programs). Sonatina files
are per-instrument SF2s; program 0 selects the file's sole preset. Add new
entries by dropping SF2 files on disk and appending here — no code changes.

Since Aug 2026 each row also carries polyphony metadata used by the direct
renderer to decide when a piece needs a companion instrument (mono targets
on poly source), how far to fold notes back into the instrument's playable
range, and how densely a plucked/decaying target can be retriggered.
"""
from dataclasses import dataclass
from typing import Literal, Optional

# ─────────────── soundfont paths (edit for your environment) ───────────────

FLUIDR3       = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SONATINA_DIR  = "/workspace/sf2"                                # holds Sonatina Symphonic Orchestra split SF2s
GENERAL_USER  = "/usr/share/sounds/sf2/GeneralUser_GS.sf2"       # optional, install via package or download

Arranger  = Literal["piano_sustain", "strings", "none"]
Polyphony = Literal["mono", "poly"]


@dataclass(frozen=True)
class Instrument:
    name: str                 # short slug used at the CLI (e.g. "piano_grand")
    display: str              # human-readable name shown in UIs
    category: str             # "piano" | "strings" | "brass" | ...
    sf2: str                  # soundfont path
    program: int              # program number inside sf2. Sonatina single-preset files: 0
    arranger: Arranger        # which arrangement stage to apply before render
    polyphony: Polyphony = "poly"
    range_lo: int = 21        # MIDI low  — A0 (piano range default)
    range_hi: int = 108       # MIDI high — C8
    min_ioi_s: float = 0.0    # per-pitch minimum inter-onset (0 = no thinning)
    default_accompaniment: Optional[str] = None  # slug in INSTRUMENTS, used when polyphony="mono"

    @property
    def id(self):
        return self.name


# ────────────────────────────── catalog ────────────────────────────────────
#
# Column meaning (positional args to Instrument):
#     name, display, category, sf2, program, arranger,
#     polyphony, range_lo, range_hi, min_ioi_s, default_accompaniment
#
# Ranges are lifted from the notebook's INSTRUMENT_RANGES (Part III of
# train_poly_instrument_conversion.ipynb) plus new rows for solos.
# min_ioi_s targets come from the density research summarized in
# `plans/adaptive-shimmying-graham.md`.

_ENTRIES = [
    # ── pianos ───────────────────────────────────────────────────────────
    ("piano_grand",             "Acoustic Grand Piano",       "piano",  FLUIDR3, 0,  "piano_sustain",  "poly", 21, 108, 0.0,  None),
    ("piano_bright",            "Bright Acoustic Piano",      "piano",  FLUIDR3, 1,  "piano_sustain",  "poly", 21, 108, 0.0,  None),
    ("piano_electric_grand",    "Electric Grand Piano",       "piano",  FLUIDR3, 2,  "piano_sustain",  "poly", 21, 108, 0.0,  None),
    ("piano_honky_tonk",        "Honky-Tonk Piano",           "piano",  FLUIDR3, 3,  "piano_sustain",  "poly", 21, 108, 0.0,  None),
    ("piano_ep1_rhodes",        "Electric Piano 1 (Rhodes)",  "piano",  FLUIDR3, 4,  "piano_sustain",  "poly", 28, 100, 0.0,  None),
    ("piano_ep2_chorus",        "Electric Piano 2 (Chorus)",  "piano",  FLUIDR3, 5,  "piano_sustain",  "poly", 28, 100, 0.0,  None),
    ("piano_sonatina_grand",    "Sonatina Grand Piano",       "piano",  f"{SONATINA_DIR}/Keys - Grand Piano.sf2", 0, "piano_sustain", "poly", 21, 108, 0.0, None),

    # ── keys / chromatic percussion (short decay → min_ioi) ─────────────
    ("harpsichord",             "Harpsichord",                "keys",   FLUIDR3, 6,  "piano_sustain",  "poly", 29, 89,  0.08, None),
    ("clavinet",                "Clavinet",                   "keys",   FLUIDR3, 7,  "piano_sustain",  "poly", 36, 84,  0.10, None),
    ("celesta",                 "Celesta",                    "keys",   FLUIDR3, 8,  "piano_sustain",  "poly", 60, 108, 0.10, None),
    ("vibraphone",              "Vibraphone",                 "keys",   FLUIDR3, 11, "piano_sustain",  "poly", 53, 89,  0.10, None),
    ("marimba",                 "Marimba",                    "keys",   FLUIDR3, 12, "piano_sustain",  "poly", 45, 96,  0.10, None),

    # ── organ ───────────────────────────────────────────────────────────
    ("organ_drawbar",           "Drawbar Organ",              "organ",  FLUIDR3, 16, "strings",        "poly", 24, 108, 0.0,  None),
    ("organ_church",            "Church Organ",               "organ",  FLUIDR3, 19, "strings",        "poly", 24, 108, 0.0,  None),
    ("accordion",               "Accordion",                  "organ",  FLUIDR3, 21, "strings",        "poly", 41, 89,  0.0,  None),

    # ── guitars (plucked → min_ioi) ─────────────────────────────────────
    ("guitar_nylon",            "Acoustic Guitar (Nylon)",    "guitar", FLUIDR3, 24, "piano_sustain",  "poly", 40, 88,  0.20, None),
    ("guitar_steel",            "Acoustic Guitar (Steel)",    "guitar", FLUIDR3, 25, "piano_sustain",  "poly", 40, 88,  0.20, None),
    ("guitar_jazz",             "Electric Guitar (Jazz)",     "guitar", FLUIDR3, 26, "piano_sustain",  "poly", 40, 88,  0.30, None),
    ("guitar_clean",            "Electric Guitar (Clean)",    "guitar", FLUIDR3, 27, "piano_sustain",  "poly", 40, 88,  0.20, None),
    ("guitar_overdrive",        "Electric Guitar (Overdrive)","guitar", FLUIDR3, 29, "piano_sustain",  "poly", 40, 88,  0.15, None),
    ("guitar_distortion",       "Electric Guitar (Distortion)","guitar",FLUIDR3, 30, "piano_sustain",  "poly", 40, 88,  0.15, None),

    # ── bass ─────────────────────────────────────────────────────────────
    ("bass_acoustic",           "Acoustic Bass",              "bass",   FLUIDR3, 32, "piano_sustain",  "poly", 28, 67,  0.15, None),
    ("bass_electric_fingered",  "Fingered Electric Bass",     "bass",   FLUIDR3, 33, "piano_sustain",  "poly", 28, 67,  0.15, None),
    ("bass_electric_picked",    "Picked Electric Bass",       "bass",   FLUIDR3, 34, "piano_sustain",  "poly", 28, 67,  0.15, None),
    ("bass_fretless",           "Fretless Bass",              "bass",   FLUIDR3, 35, "piano_sustain",  "poly", 28, 67,  0.0,  None),

    # ── strings (FluidR3) ───────────────────────────────────────────────
    ("violin_fluidr3",          "Violin (FluidR3 GM)",        "strings", FLUIDR3, 40, "strings",       "mono", 55, 105, 0.0,  "harp_sonatina"),
    ("viola_fluidr3",           "Viola (FluidR3 GM)",         "strings", FLUIDR3, 41, "strings",       "mono", 48, 88,  0.0,  "harp_sonatina"),
    ("cello_fluidr3",           "Cello (FluidR3 GM)",         "strings", FLUIDR3, 42, "strings",       "mono", 36, 84,  0.0,  "harp_sonatina"),
    ("contrabass_fluidr3",      "Contrabass (FluidR3 GM)",    "strings", FLUIDR3, 43, "strings",       "mono", 28, 67,  0.0,  "harp_sonatina"),
    ("strings_tremolo",         "Tremolo Strings",            "strings", FLUIDR3, 44, "strings",       "poly", 28, 105, 0.0,  None),
    ("strings_pizzicato",       "Pizzicato Strings",          "strings", FLUIDR3, 45, "none",          "poly", 28, 105, 0.15, None),
    ("strings_ensemble_1",      "String Ensemble 1",          "strings", FLUIDR3, 48, "strings",       "poly", 28, 105, 0.0,  None),
    ("strings_ensemble_2",      "String Ensemble 2",          "strings", FLUIDR3, 49, "strings",       "poly", 28, 105, 0.0,  None),
    ("synth_strings_1",         "SynthStrings 1",             "strings", FLUIDR3, 50, "strings",       "poly", 28, 105, 0.0,  None),
    ("synth_strings_2",         "SynthStrings 2",             "strings", FLUIDR3, 51, "strings",       "poly", 28, 105, 0.0,  None),

    # ── strings (Sonatina — real sampled orchestral) ────────────────────
    ("violin1_sustain",         "1st Violins Sustain (Sonatina)",   "strings", f"{SONATINA_DIR}/Strings - 1st Violins Sustain.sf2",   0, "strings", "poly", 55, 105, 0.0,  None),
    ("violin1_pizzicato",       "1st Violins Pizzicato (Sonatina)", "strings", f"{SONATINA_DIR}/Strings - 1st Violins Pizzicato.sf2", 0, "none",    "poly", 55, 105, 0.15, None),
    ("violin1_staccato",        "1st Violins Staccato (Sonatina)",  "strings", f"{SONATINA_DIR}/Strings - 1st Violins Staccato.sf2",  0, "none",    "poly", 55, 105, 0.10, None),
    ("violin2_sustain",         "2nd Violins Sustain (Sonatina)",   "strings", f"{SONATINA_DIR}/Strings - 2nd Violins Sustain.sf2",   0, "strings", "poly", 55, 105, 0.0,  None),
    ("violin_solo",             "Violin Solo (Sonatina)",           "strings", f"{SONATINA_DIR}/Strings - Violin Solo.sf2",           0, "strings", "mono", 55, 105, 0.0,  "harp_sonatina"),
    ("viola_sustain",           "Violas Sustain (Sonatina)",        "strings", f"{SONATINA_DIR}/Strings - Violas Sustain.sf2",        0, "strings", "poly", 48, 88,  0.0,  None),
    ("viola_pizzicato",         "Violas Pizzicato (Sonatina)",      "strings", f"{SONATINA_DIR}/Strings - Violas Pizzicato.sf2",      0, "none",    "poly", 48, 88,  0.15, None),
    ("cello_sustain",           "Celli Sustain (Sonatina)",         "strings", f"{SONATINA_DIR}/Strings - Celli Sustain.sf2",         0, "strings", "poly", 36, 84,  0.0,  None),
    ("cello_pizzicato",         "Celli Pizzicato (Sonatina)",       "strings", f"{SONATINA_DIR}/Strings - Celli Pizzicato.sf2",       0, "none",    "poly", 36, 84,  0.15, None),
    ("cello_solo",              "Cello Solo (Sonatina)",            "strings", f"{SONATINA_DIR}/Strings - Cello Solo.sf2",            0, "strings", "mono", 36, 84,  0.0,  "harp_sonatina"),
    ("bass_sustain",            "Basses Sustain (Sonatina)",        "strings", f"{SONATINA_DIR}/Strings - Basses Sustain.sf2",        0, "strings", "poly", 28, 67,  0.0,  None),
    ("bass_pizzicato",          "Basses Pizzicato (Sonatina)",      "strings", f"{SONATINA_DIR}/Strings - Basses Pizzicato.sf2",      0, "none",    "poly", 28, 67,  0.15, None),

    # ── brass (FluidR3) ─────────────────────────────────────────────────
    ("trumpet_fluidr3",         "Trumpet (FluidR3 GM)",       "brass",  FLUIDR3, 56, "strings",        "mono", 52, 84,  0.0,  "piano_ep1_rhodes"),
    ("trombone_fluidr3",        "Trombone (FluidR3 GM)",      "brass",  FLUIDR3, 57, "strings",        "mono", 40, 77,  0.0,  "piano_ep1_rhodes"),
    ("tuba_fluidr3",            "Tuba (FluidR3 GM)",          "brass",  FLUIDR3, 58, "strings",        "mono", 26, 65,  0.0,  "piano_ep1_rhodes"),
    ("french_horn_fluidr3",     "French Horn (FluidR3 GM)",   "brass",  FLUIDR3, 60, "strings",        "mono", 34, 77,  0.0,  "piano_ep1_rhodes"),
    ("brass_ensemble",          "Brass Ensemble",             "brass",  FLUIDR3, 61, "strings",        "poly", 34, 84,  0.0,  None),

    # ── brass (Sonatina) ────────────────────────────────────────────────
    ("trumpet_solo",            "Trumpet Solo (Sonatina)",    "brass",  f"{SONATINA_DIR}/Brass - Trumpet Solo.sf2",   0, "strings", "mono", 52, 84,  0.0,  "piano_ep1_rhodes"),
    ("horns_sustain",           "Horns Sustain (Sonatina)",   "brass",  f"{SONATINA_DIR}/Brass - Horns Sustain.sf2",  0, "strings", "poly", 34, 77,  0.0,  None),
    ("trombones_sustain",       "Trombones Sustain (Sonatina)","brass", f"{SONATINA_DIR}/Brass - Trombones Sustain.sf2", 0, "strings","poly", 40, 77,  0.0,  None),
    ("tuba_sonatina",           "Tuba Sustain (Sonatina)",    "brass",  f"{SONATINA_DIR}/Brass - Tuba Sustain.sf2",   0, "strings", "mono", 26, 65,  0.0,  "piano_ep1_rhodes"),

    # ── saxophones (mono) ───────────────────────────────────────────────
    ("soprano_sax",             "Soprano Sax",                "wind",   FLUIDR3, 64, "strings",        "mono", 56, 87,  0.0,  "piano_ep1_rhodes"),
    ("alto_sax",                "Alto Sax",                   "wind",   FLUIDR3, 65, "strings",        "mono", 49, 80,  0.0,  "piano_ep1_rhodes"),
    ("tenor_sax",               "Tenor Sax",                  "wind",   FLUIDR3, 66, "strings",        "mono", 44, 75,  0.0,  "piano_ep1_rhodes"),
    ("baritone_sax",            "Baritone Sax",               "wind",   FLUIDR3, 67, "strings",        "mono", 36, 68,  0.0,  "piano_ep1_rhodes"),

    # ── woodwinds (mono) ────────────────────────────────────────────────
    ("oboe_fluidr3",            "Oboe (FluidR3 GM)",          "wind",   FLUIDR3, 68, "strings",        "mono", 58, 91,  0.0,  "guitar_nylon"),
    ("clarinet_fluidr3",        "Clarinet (FluidR3 GM)",      "wind",   FLUIDR3, 71, "strings",        "mono", 50, 91,  0.0,  "guitar_nylon"),
    ("flute_fluidr3",           "Flute (FluidR3 GM)",         "wind",   FLUIDR3, 73, "strings",        "mono", 60, 98,  0.0,  "guitar_nylon"),
    ("recorder",                "Recorder",                   "wind",   FLUIDR3, 74, "strings",        "mono", 60, 96,  0.0,  "guitar_nylon"),
    ("pan_flute",               "Pan Flute",                  "wind",   FLUIDR3, 75, "strings",        "mono", 60, 96,  0.0,  "guitar_nylon"),
    ("flute_solo",              "Flute Solo (Sonatina)",      "wind",   f"{SONATINA_DIR}/Woodwinds - Flute Solo.sf2",    0, "strings", "mono", 60, 98,  0.0, "guitar_nylon"),
    ("oboe_solo",               "Oboe Solo (Sonatina)",       "wind",   f"{SONATINA_DIR}/Woodwinds - Oboe Solo.sf2",     0, "strings", "mono", 58, 91,  0.0, "guitar_nylon"),
    ("clarinet_solo",           "Clarinet Solo (Sonatina)",   "wind",   f"{SONATINA_DIR}/Woodwinds - Clarinet Solo.sf2", 0, "strings", "mono", 50, 91,  0.0, "guitar_nylon"),
    ("bassoon_solo",            "Bassoon Solo (Sonatina)",    "wind",   f"{SONATINA_DIR}/Woodwinds - Bassoon Solo.sf2",  0, "strings", "mono", 34, 72,  0.0, "guitar_nylon"),

    # ── choir / voice ───────────────────────────────────────────────────
    ("choir_aahs",              "Choir Aahs",                 "voice",  FLUIDR3, 52, "strings",        "poly", 40, 84,  0.0,  None),
    ("voice_oohs",              "Voice Oohs",                 "voice",  FLUIDR3, 53, "strings",        "poly", 40, 84,  0.0,  None),
    ("synth_voice",             "Synth Voice",                "voice",  FLUIDR3, 54, "strings",        "poly", 40, 84,  0.0,  None),

    # ── harp (plucked → strong min_ioi) ─────────────────────────────────
    ("harp_fluidr3",            "Orchestral Harp (FluidR3)",  "harp",   FLUIDR3, 46, "piano_sustain",  "poly", 24, 103, 1.2,  None),
    ("harp_sonatina",           "Concert Harp (Sonatina)",    "harp",   f"{SONATINA_DIR}/Concert Harp.sf2", 0, "piano_sustain", "poly", 24, 103, 1.2, None),

    # ── synths + pads (FluidR3 80-95) ──────────────────────────────────
    ("synth_square",            "Lead: Square",               "synth",  FLUIDR3, 80, "strings",        "mono", 36, 96,  0.0,  "pad_warm"),
    ("synth_saw",               "Lead: Saw",                  "synth",  FLUIDR3, 81, "strings",        "mono", 36, 96,  0.0,  "pad_warm"),
    ("pad_new_age",             "Pad: New Age",               "synth",  FLUIDR3, 88, "strings",        "poly", 28, 96,  0.0,  None),
    ("pad_warm",                "Pad: Warm",                  "synth",  FLUIDR3, 89, "strings",        "poly", 28, 96,  0.0,  None),
    ("pad_bowed",               "Pad: Bowed",                 "synth",  FLUIDR3, 92, "strings",        "poly", 28, 96,  0.0,  None),
]

INSTRUMENTS = {e[0]: Instrument(*e) for e in _ENTRIES}
"""
Lookup: INSTRUMENTS['piano_grand'] -> Instrument(...)
Iterate: for inst in INSTRUMENTS.values(): ...
"""


def by_category(category: str) -> list[Instrument]:
    return [i for i in INSTRUMENTS.values() if i.category == category]


def list_categories() -> list[str]:
    return sorted({i.category for i in INSTRUMENTS.values()})


if __name__ == "__main__":
    # dump the catalog grouped by category
    for cat in list_categories():
        entries = by_category(cat)
        print(f"\n═══ {cat.upper()} ({len(entries)}) ═══")
        for inst in entries:
            tag = f"[{inst.polyphony}]" if inst.polyphony == "mono" else "     "
            ioi = f"  ioi={inst.min_ioi_s:.2f}" if inst.min_ioi_s > 0 else ""
            print(f"  {tag} {inst.name:32s} {inst.display}{ioi}")
    print(f"\nTOTAL: {len(INSTRUMENTS)} instruments across {len(list_categories())} categories")

"""Instrument catalog for the DIRECT pipeline.

Each instrument is a target the pipeline can render into: a soundfont path,
a program number to select inside that soundfont, and which per-target
arrangement stage to apply (piano_sustain / strings / none).

All FluidR3 GM instruments share one SF2 (128 GM programs). Sonatina files
are per-instrument SF2s; program 0 selects the file's sole preset. Add new
entries by dropping SF2 files on disk and appending here — no code changes.
"""
from dataclasses import dataclass
from typing import Literal

# ─────────────── soundfont paths (edit for your environment) ───────────────

FLUIDR3       = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SONATINA_DIR  = "/workspace/sf2"                                # holds Sonatina Symphonic Orchestra split SF2s
GENERAL_USER  = "/usr/share/sounds/sf2/GeneralUser_GS.sf2"       # optional, install via package or download

Arranger = Literal["piano_sustain", "strings", "none"]


@dataclass(frozen=True)
class Instrument:
    name: str            # short slug used at the CLI (e.g. "piano_grand")
    display: str         # human-readable name shown in UIs
    category: str        # "piano" | "strings" | "brass" | ...
    sf2: str             # soundfont path
    program: int         # program number inside sf2. Sonatina single-preset files: 0
    arranger: Arranger   # which arrangement stage to apply before render

    @property
    def id(self):
        return self.name


# ────────────────────────────── catalog ────────────────────────────────────

_ENTRIES = [
    # ── pianos ───────────────────────────────────────────────────────────
    ("piano_grand",             "Acoustic Grand Piano",       "piano",  FLUIDR3, 0,  "piano_sustain"),
    ("piano_bright",            "Bright Acoustic Piano",      "piano",  FLUIDR3, 1,  "piano_sustain"),
    ("piano_electric_grand",    "Electric Grand Piano",       "piano",  FLUIDR3, 2,  "piano_sustain"),
    ("piano_honky_tonk",        "Honky-Tonk Piano",           "piano",  FLUIDR3, 3,  "piano_sustain"),
    ("piano_ep1_rhodes",        "Electric Piano 1 (Rhodes)",  "piano",  FLUIDR3, 4,  "piano_sustain"),
    ("piano_ep2_chorus",        "Electric Piano 2 (Chorus)",  "piano",  FLUIDR3, 5,  "piano_sustain"),
    ("piano_sonatina_grand",    "Sonatina Grand Piano",       "piano",  f"{SONATINA_DIR}/Keys - Grand Piano.sf2", 0, "piano_sustain"),

    # ── keys / chromatic percussion ─────────────────────────────────────
    ("harpsichord",             "Harpsichord",                "keys",   FLUIDR3, 6,  "piano_sustain"),
    ("clavinet",                "Clavinet",                   "keys",   FLUIDR3, 7,  "piano_sustain"),
    ("celesta",                 "Celesta",                    "keys",   FLUIDR3, 8,  "piano_sustain"),
    ("vibraphone",              "Vibraphone",                 "keys",   FLUIDR3, 11, "piano_sustain"),
    ("marimba",                 "Marimba",                    "keys",   FLUIDR3, 12, "piano_sustain"),

    # ── organ ───────────────────────────────────────────────────────────
    ("organ_drawbar",           "Drawbar Organ",              "organ",  FLUIDR3, 16, "strings"),
    ("organ_church",            "Church Organ",               "organ",  FLUIDR3, 19, "strings"),
    ("accordion",               "Accordion",                  "organ",  FLUIDR3, 21, "strings"),

    # ── guitars ─────────────────────────────────────────────────────────
    ("guitar_nylon",            "Acoustic Guitar (Nylon)",    "guitar", FLUIDR3, 24, "piano_sustain"),
    ("guitar_steel",            "Acoustic Guitar (Steel)",    "guitar", FLUIDR3, 25, "piano_sustain"),
    ("guitar_jazz",             "Electric Guitar (Jazz)",     "guitar", FLUIDR3, 26, "piano_sustain"),
    ("guitar_clean",            "Electric Guitar (Clean)",    "guitar", FLUIDR3, 27, "piano_sustain"),
    ("guitar_overdrive",        "Electric Guitar (Overdrive)","guitar", FLUIDR3, 29, "piano_sustain"),
    ("guitar_distortion",       "Electric Guitar (Distortion)","guitar",FLUIDR3, 30, "piano_sustain"),

    # ── bass ─────────────────────────────────────────────────────────────
    ("bass_acoustic",           "Acoustic Bass",              "bass",   FLUIDR3, 32, "piano_sustain"),
    ("bass_electric_fingered",  "Fingered Electric Bass",     "bass",   FLUIDR3, 33, "piano_sustain"),
    ("bass_electric_picked",    "Picked Electric Bass",       "bass",   FLUIDR3, 34, "piano_sustain"),
    ("bass_fretless",           "Fretless Bass",              "bass",   FLUIDR3, 35, "piano_sustain"),

    # ── strings (FluidR3) ───────────────────────────────────────────────
    ("violin_fluidr3",          "Violin (FluidR3 GM)",        "strings", FLUIDR3, 40, "strings"),
    ("viola_fluidr3",           "Viola (FluidR3 GM)",         "strings", FLUIDR3, 41, "strings"),
    ("cello_fluidr3",           "Cello (FluidR3 GM)",         "strings", FLUIDR3, 42, "strings"),
    ("contrabass_fluidr3",      "Contrabass (FluidR3 GM)",    "strings", FLUIDR3, 43, "strings"),
    ("strings_tremolo",         "Tremolo Strings",            "strings", FLUIDR3, 44, "strings"),
    ("strings_pizzicato",       "Pizzicato Strings",          "strings", FLUIDR3, 45, "none"),
    ("strings_ensemble_1",      "String Ensemble 1",          "strings", FLUIDR3, 48, "strings"),
    ("strings_ensemble_2",      "String Ensemble 2",          "strings", FLUIDR3, 49, "strings"),
    ("synth_strings_1",         "SynthStrings 1",             "strings", FLUIDR3, 50, "strings"),
    ("synth_strings_2",         "SynthStrings 2",             "strings", FLUIDR3, 51, "strings"),

    # ── strings (Sonatina — real sampled orchestral) ────────────────────
    ("violin1_sustain",         "1st Violins Sustain (Sonatina)",   "strings", f"{SONATINA_DIR}/Strings - 1st Violins Sustain.sf2",   0, "strings"),
    ("violin1_pizzicato",       "1st Violins Pizzicato (Sonatina)", "strings", f"{SONATINA_DIR}/Strings - 1st Violins Pizzicato.sf2", 0, "none"),
    ("violin1_staccato",        "1st Violins Staccato (Sonatina)",  "strings", f"{SONATINA_DIR}/Strings - 1st Violins Staccato.sf2",  0, "none"),
    ("violin2_sustain",         "2nd Violins Sustain (Sonatina)",   "strings", f"{SONATINA_DIR}/Strings - 2nd Violins Sustain.sf2",   0, "strings"),
    ("violin_solo",             "Violin Solo (Sonatina)",           "strings", f"{SONATINA_DIR}/Strings - Violin Solo.sf2",           0, "strings"),
    ("viola_sustain",           "Violas Sustain (Sonatina)",        "strings", f"{SONATINA_DIR}/Strings - Violas Sustain.sf2",        0, "strings"),
    ("viola_pizzicato",         "Violas Pizzicato (Sonatina)",      "strings", f"{SONATINA_DIR}/Strings - Violas Pizzicato.sf2",      0, "none"),
    ("cello_sustain",           "Celli Sustain (Sonatina)",         "strings", f"{SONATINA_DIR}/Strings - Celli Sustain.sf2",         0, "strings"),
    ("cello_pizzicato",         "Celli Pizzicato (Sonatina)",       "strings", f"{SONATINA_DIR}/Strings - Celli Pizzicato.sf2",       0, "none"),
    ("cello_solo",              "Cello Solo (Sonatina)",            "strings", f"{SONATINA_DIR}/Strings - Cello Solo.sf2",            0, "strings"),
    ("bass_sustain",            "Basses Sustain (Sonatina)",        "strings", f"{SONATINA_DIR}/Strings - Basses Sustain.sf2",        0, "strings"),
    ("bass_pizzicato",          "Basses Pizzicato (Sonatina)",      "strings", f"{SONATINA_DIR}/Strings - Basses Pizzicato.sf2",      0, "none"),

    # ── brass (FluidR3) ─────────────────────────────────────────────────
    ("trumpet_fluidr3",         "Trumpet (FluidR3 GM)",       "brass",  FLUIDR3, 56, "strings"),
    ("trombone_fluidr3",        "Trombone (FluidR3 GM)",      "brass",  FLUIDR3, 57, "strings"),
    ("tuba_fluidr3",            "Tuba (FluidR3 GM)",          "brass",  FLUIDR3, 58, "strings"),
    ("french_horn_fluidr3",     "French Horn (FluidR3 GM)",   "brass",  FLUIDR3, 60, "strings"),
    ("brass_ensemble",          "Brass Ensemble",             "brass",  FLUIDR3, 61, "strings"),

    # ── brass (Sonatina) ────────────────────────────────────────────────
    ("trumpet_solo",            "Trumpet Solo (Sonatina)",    "brass",  f"{SONATINA_DIR}/Brass - Trumpet Solo.sf2",   0, "strings"),
    ("horns_sustain",           "Horns Sustain (Sonatina)",   "brass",  f"{SONATINA_DIR}/Brass - Horns Sustain.sf2",  0, "strings"),
    ("trombones_sustain",       "Trombones Sustain (Sonatina)","brass", f"{SONATINA_DIR}/Brass - Trombones Sustain.sf2", 0, "strings"),
    ("tuba_sonatina",           "Tuba Sustain (Sonatina)",    "brass",  f"{SONATINA_DIR}/Brass - Tuba Sustain.sf2",   0, "strings"),

    # ── saxophones ──────────────────────────────────────────────────────
    ("soprano_sax",             "Soprano Sax",                "wind",   FLUIDR3, 64, "strings"),
    ("alto_sax",                "Alto Sax",                   "wind",   FLUIDR3, 65, "strings"),
    ("tenor_sax",               "Tenor Sax",                  "wind",   FLUIDR3, 66, "strings"),
    ("baritone_sax",            "Baritone Sax",               "wind",   FLUIDR3, 67, "strings"),

    # ── woodwinds ──────────────────────────────────────────────────────
    ("oboe_fluidr3",            "Oboe (FluidR3 GM)",          "wind",   FLUIDR3, 68, "strings"),
    ("clarinet_fluidr3",        "Clarinet (FluidR3 GM)",      "wind",   FLUIDR3, 71, "strings"),
    ("flute_fluidr3",           "Flute (FluidR3 GM)",         "wind",   FLUIDR3, 73, "strings"),
    ("recorder",                "Recorder",                   "wind",   FLUIDR3, 74, "strings"),
    ("pan_flute",               "Pan Flute",                  "wind",   FLUIDR3, 75, "strings"),
    ("flute_solo",              "Flute Solo (Sonatina)",      "wind",   f"{SONATINA_DIR}/Woodwinds - Flute Solo.sf2",    0, "strings"),
    ("oboe_solo",               "Oboe Solo (Sonatina)",       "wind",   f"{SONATINA_DIR}/Woodwinds - Oboe Solo.sf2",     0, "strings"),
    ("clarinet_solo",           "Clarinet Solo (Sonatina)",   "wind",   f"{SONATINA_DIR}/Woodwinds - Clarinet Solo.sf2", 0, "strings"),
    ("bassoon_solo",            "Bassoon Solo (Sonatina)",    "wind",   f"{SONATINA_DIR}/Woodwinds - Bassoon Solo.sf2",  0, "strings"),

    # ── choir / voice ───────────────────────────────────────────────────
    ("choir_aahs",              "Choir Aahs",                 "voice",  FLUIDR3, 52, "strings"),
    ("voice_oohs",              "Voice Oohs",                 "voice",  FLUIDR3, 53, "strings"),
    ("synth_voice",             "Synth Voice",                "voice",  FLUIDR3, 54, "strings"),

    # ── harp ────────────────────────────────────────────────────────────
    ("harp_fluidr3",            "Orchestral Harp (FluidR3)",  "harp",   FLUIDR3, 46, "piano_sustain"),
    ("harp_sonatina",            "Concert Harp (Sonatina)",    "harp",   f"{SONATINA_DIR}/Concert Harp.sf2", 0, "piano_sustain"),

    # ── synths + pads (FluidR3 80-95) ──────────────────────────────────
    ("synth_square",            "Lead: Square",               "synth",  FLUIDR3, 80, "strings"),
    ("synth_saw",               "Lead: Saw",                  "synth",  FLUIDR3, 81, "strings"),
    ("pad_new_age",             "Pad: New Age",               "synth",  FLUIDR3, 88, "strings"),
    ("pad_warm",                "Pad: Warm",                  "synth",  FLUIDR3, 89, "strings"),
    ("pad_bowed",               "Pad: Bowed",                 "synth",  FLUIDR3, 92, "strings"),
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
            print(f"  {inst.name:32s} {inst.display}")
    print(f"\nTOTAL: {len(INSTRUMENTS)} instruments across {len(list_categories())} categories")

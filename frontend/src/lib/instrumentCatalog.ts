// General-MIDI instruments (gleitz names, matching smplr's Soundfont catalog) that we
// expose in the "change instrument" dropdown, ordered closest-first per source stem.

export interface InstrumentDef {
  id: string; // GM name for smplr Soundfont
  label: string;
  family: string;
}

export const INSTRUMENTS: InstrumentDef[] = [
  // Piano / keys
  { id: "acoustic_grand_piano", label: "Grand Piano", family: "Keys" },
  { id: "bright_acoustic_piano", label: "Bright Piano", family: "Keys" },
  { id: "electric_piano_1", label: "Electric Piano (Rhodes)", family: "Keys" },
  { id: "electric_piano_2", label: "Electric Piano (FM)", family: "Keys" },
  { id: "harpsichord", label: "Harpsichord", family: "Keys" },
  { id: "clavinet", label: "Clavinet", family: "Keys" },
  // Mallets / chromatic
  { id: "vibraphone", label: "Vibraphone", family: "Mallets" },
  { id: "marimba", label: "Marimba", family: "Mallets" },
  { id: "xylophone", label: "Xylophone", family: "Mallets" },
  { id: "celesta", label: "Celesta", family: "Mallets" },
  { id: "music_box", label: "Music Box", family: "Mallets" },
  { id: "kalimba", label: "Kalimba", family: "Mallets" },
  { id: "tubular_bells", label: "Tubular Bells", family: "Mallets" },
  // Organ
  { id: "drawbar_organ", label: "Drawbar Organ", family: "Organ" },
  { id: "church_organ", label: "Church Organ", family: "Organ" },
  { id: "accordion", label: "Accordion", family: "Organ" },
  { id: "harmonica", label: "Harmonica", family: "Organ" },
  // Guitar
  { id: "acoustic_guitar_nylon", label: "Nylon Guitar", family: "Guitar" },
  { id: "acoustic_guitar_steel", label: "Steel Guitar", family: "Guitar" },
  { id: "electric_guitar_clean", label: "Electric Guitar (Clean)", family: "Guitar" },
  { id: "overdriven_guitar", label: "Overdriven Guitar", family: "Guitar" },
  { id: "distortion_guitar", label: "Distortion Guitar", family: "Guitar" },
  { id: "banjo", label: "Banjo", family: "Guitar" },
  { id: "sitar", label: "Sitar", family: "Guitar" },
  // Bass
  { id: "acoustic_bass", label: "Acoustic Bass", family: "Bass" },
  { id: "electric_bass_finger", label: "Electric Bass (Finger)", family: "Bass" },
  { id: "electric_bass_pick", label: "Electric Bass (Pick)", family: "Bass" },
  { id: "fretless_bass", label: "Fretless Bass", family: "Bass" },
  { id: "synth_bass_1", label: "Synth Bass", family: "Bass" },
  // Strings
  { id: "violin", label: "Violin", family: "Strings" },
  { id: "viola", label: "Viola", family: "Strings" },
  { id: "cello", label: "Cello", family: "Strings" },
  { id: "contrabass", label: "Contrabass", family: "Strings" },
  { id: "pizzicato_strings", label: "Pizzicato Strings", family: "Strings" },
  { id: "orchestral_harp", label: "Harp", family: "Strings" },
  { id: "string_ensemble_1", label: "String Ensemble", family: "Ensemble" },
  { id: "synth_strings_1", label: "Synth Strings", family: "Ensemble" },
  // Voice
  { id: "choir_aahs", label: "Choir (Aahs)", family: "Voice" },
  { id: "voice_oohs", label: "Voice (Oohs)", family: "Voice" },
  // Brass
  { id: "trumpet", label: "Trumpet", family: "Brass" },
  { id: "trombone", label: "Trombone", family: "Brass" },
  { id: "french_horn", label: "French Horn", family: "Brass" },
  { id: "brass_section", label: "Brass Section", family: "Brass" },
  { id: "tuba", label: "Tuba", family: "Brass" },
  // Reed / wind
  { id: "alto_sax", label: "Alto Sax", family: "Wind" },
  { id: "tenor_sax", label: "Tenor Sax", family: "Wind" },
  { id: "clarinet", label: "Clarinet", family: "Wind" },
  { id: "oboe", label: "Oboe", family: "Wind" },
  { id: "flute", label: "Flute", family: "Wind" },
  { id: "pan_flute", label: "Pan Flute", family: "Wind" },
  // Synth
  { id: "lead_2_sawtooth", label: "Synth Lead (Saw)", family: "Synth" },
  { id: "lead_1_square", label: "Synth Lead (Square)", family: "Synth" },
  { id: "pad_2_warm", label: "Warm Pad", family: "Synth" },
  { id: "pad_5_bowed", label: "Bowed Pad", family: "Synth" },
];

// Closest-first priority per source stem (the rest follow in catalog order).
const STEM_PRIORITY: Record<string, string[]> = {
  piano: ["acoustic_grand_piano", "electric_piano_1", "electric_piano_2", "harpsichord", "clavinet", "vibraphone", "marimba", "celesta", "music_box"],
  vocals: ["choir_aahs", "voice_oohs", "string_ensemble_1", "violin", "cello", "flute", "clarinet", "alto_sax", "trumpet"],
  bass: ["electric_bass_finger", "acoustic_bass", "fretless_bass", "electric_bass_pick", "synth_bass_1", "cello", "contrabass", "tuba"],
  guitar: ["acoustic_guitar_nylon", "acoustic_guitar_steel", "electric_guitar_clean", "overdriven_guitar", "distortion_guitar", "banjo", "sitar", "orchestral_harp"],
  piano_alt: [],
  drums: ["marimba", "xylophone", "vibraphone", "kalimba", "tubular_bells"],
  other: ["acoustic_grand_piano", "electric_piano_1", "violin", "string_ensemble_1", "flute", "trumpet"],
  instrumental: ["acoustic_grand_piano", "string_ensemble_1", "violin", "electric_guitar_clean", "flute"],
};

export function instrumentsForStem(stem: string): InstrumentDef[] {
  const prio = STEM_PRIORITY[stem] ?? STEM_PRIORITY.other;
  const byId = new Map(INSTRUMENTS.map((i) => [i.id, i]));
  const ordered: InstrumentDef[] = [];
  const seen = new Set<string>();
  for (const id of prio) {
    const d = byId.get(id);
    if (d) {
      ordered.push(d);
      seen.add(id);
    }
  }
  for (const d of INSTRUMENTS) if (!seen.has(d.id)) ordered.push(d);
  return ordered;
}

export function instrumentLabel(id: string | null): string {
  if (!id) return "Original";
  return INSTRUMENTS.find((i) => i.id === id)?.label ?? id;
}

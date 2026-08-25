"""Copy of training/poly/arrange.py — kept in-tree so the direct pipeline
has no cross-directory Python imports. Update both if you change either.

Two idiomatic MIDI transforms applied between transcription and SF2 render:

  arrange_for_piano_sustain(pm)
    Light note-end extension. Basic Pitch (and most transcribers) strip
    real sustain-pedal information, leaving MIDI that reads as staccato
    attacks. This extends each substantive note by up to ~300 ms to
    partially recover the pedaled feel without adding new notes.

  arrange_for_strings(pm)
    Sostenuto: extend each held note to the next onset in its voice
    (nearest neighbor within ±3 semitones), capped at +0.8 s. Skips
    already-staccato notes (<200 ms) so those stay short.
"""
import pretty_midi
import numpy as np


def _all_notes(pm):
    out = []
    for inst in pm.instruments:
        out.extend((n, inst) for n in inst.notes)
    return out


def arrange_for_piano_sustain(pm, extend_s=0.30, min_note_len_for_ext_s=0.10):
    all_notes = [n for (n, _) in _all_notes(pm)]
    for n in all_notes:
        if (n.end - n.start) < min_note_len_for_ext_s:
            continue
        n.end = n.end + extend_s
    return pm


def arrange_for_strings(pm,
                        max_hold_s=0.8,
                        min_note_len_for_hold_s=0.20,
                        legato_overlap_s=0.04,
                        voice_semitone_window=3):
    all_notes = [n for (n, _) in _all_notes(pm)]
    if not all_notes:
        return pm
    all_notes.sort(key=lambda n: n.start)
    for i, n in enumerate(all_notes):
        if (n.end - n.start) < min_note_len_for_hold_s:
            continue
        target_end = n.start + max_hold_s
        for j in range(i + 1, len(all_notes)):
            m = all_notes[j]
            if m.start >= target_end:
                break
            if m.start <= n.start + 0.01:
                continue
            if abs(m.pitch - n.pitch) <= voice_semitone_window:
                target_end = m.start + legato_overlap_s
                break
        n.end = max(n.end, min(target_end, n.start + max_hold_s))
    return pm


ARRANGERS = {
    "piano_sustain": arrange_for_piano_sustain,
    "strings":       arrange_for_strings,
    "none":          lambda pm: pm,
}

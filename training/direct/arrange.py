"""Per-target MIDI transforms applied between transcription and SF2 render.

Two idiomatic post-transcription passes (kept from the original arranger):

  arrange_for_piano_sustain(pm)
    Light note-end extension. Basic Pitch (and most transcribers) strip
    real sustain-pedal information, leaving MIDI that reads as staccato
    attacks. This extends each substantive note by up to ~300 ms to
    partially recover the pedaled feel without adding new notes.

  arrange_for_strings(pm)
    Sostenuto: extend each held note to the next onset in its voice
    (nearest neighbor within ±3 semitones), capped at +0.8 s. Skips
    already-staccato notes (<200 ms) so those stay short.

And four polyphony-aware primitives (Aug 2026 addition) invoked from
`render_direct.render_one` on top of the per-target arranger:

  clamp_to_range(pm, lo, hi, mode="octave")
    Fold or drop notes outside a target's playable MIDI range.

  split_lead_accompaniment(pm)
    Skyline (top-pitch) lead + everything-else accompaniment. Used when
    the target is monophonic and the source is polyphonic.

  enforce_min_ioi(pm, min_ioi_s)
    Per-pitch minimum inter-onset + global 1s rate cap. Stops
    plucked/decaying targets (harp, guitar, harpsichord, marimba, pizz)
    from being retriggered faster than the instrument breathes.

  _reduce_polyphony(notes, max_voices, tol=0.035)
    Cluster-wise voice thinning: keep outer voices (bass + skyline) then
    add from the top. Used by split_lead_accompaniment.
"""
from __future__ import annotations
import copy
import pretty_midi


# ────────────────────────── existing helpers ───────────────────────────────

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


# ────────────────────────── polyphony-aware ────────────────────────────────
#
# CLUSTER_TOL: onsets within this many seconds are treated as one chord for
# the purposes of split_lead_accompaniment / _reduce_polyphony. 35 ms is the
# score-following consensus (see ACCompanion, arXiv:2304.12939) — small enough
# to keep true polyrhythms separate, large enough to catch a piano's
# not-quite-simultaneous chord voicings.
CLUSTER_TOL = 0.035


def clamp_to_range(pm, lo: int, hi: int, mode: str = "octave"):
    """Bring every note into [lo, hi].

    mode="octave"  — fold by whole octaves (preserves pitch class). Musically
                     far better than clamping, which collapses distinct notes
                     onto one pitch. Default.
    mode="drop"    — silently drop out-of-range notes.
    """
    for inst in pm.instruments:
        kept = []
        for n in inst.notes:
            p = n.pitch
            if mode == "octave":
                while p < lo:
                    p += 12
                while p > hi:
                    p -= 12
                if lo <= p <= hi:
                    n.pitch = p
                    kept.append(n)
            else:
                if lo <= p <= hi:
                    kept.append(n)
        inst.notes = kept
    return pm


def _cluster_onsets(notes, tol: float = CLUSTER_TOL):
    """Group notes into onset clusters. Returns list-of-lists sorted by time."""
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: n.start)
    groups, cur = [], [ordered[0]]
    for n in ordered[1:]:
        if n.start - cur[0].start <= tol:
            cur.append(n)
        else:
            groups.append(cur)
            cur = [n]
    groups.append(cur)
    return groups


def _reduce_polyphony(notes, max_voices: int, tol: float = CLUSTER_TOL):
    """Thin simultaneous notes to `max_voices`, keeping the outer voices.

    Musical rationale: in a chord the BASS defines the harmony and the TOP
    defines the melody. Inner voices are the most expendable — standard
    orchestral reduction practice.
    """
    out = []
    for g in _cluster_onsets(notes, tol=tol):
        if len(g) <= max_voices:
            out.extend(g)
            continue
        g_sorted = sorted(g, key=lambda n: n.pitch)
        picked = [g_sorted[0], g_sorted[-1]]      # bass + skyline first
        for n in reversed(g_sorted[1:-1]):        # then interior from top down
            if len(picked) >= max_voices:
                break
            picked.append(n)
        out.extend(picked)
    return sorted(out, key=lambda n: n.start)


def split_lead_accompaniment(pm, lead_voices: int = 1):
    """Skyline split. Returns (lead_pm, accomp_pm) — both are deep copies.

    Per-cluster: the top `lead_voices` pitches go to the lead track; every
    other note in the cluster goes to accompaniment. Notes that are alone in
    their onset cluster still go to the lead — a single-voice line stays
    unchanged.

    Rationale: the skyline (top pitch) is the melody almost by convention in
    Western tonal music (Uitdenbogerd & Zobel 1998). A monophonic target
    (violin_solo, trumpet_solo, flute_solo) should play that line; a
    polyphonic accompaniment carries the harmony underneath.
    """
    lead_pm  = copy.deepcopy(pm)
    accomp_pm = copy.deepcopy(pm)

    for lead_inst, accomp_inst in zip(lead_pm.instruments, accomp_pm.instruments):
        lead_kept, accomp_kept = [], []
        for g in _cluster_onsets(lead_inst.notes):
            if len(g) <= lead_voices:
                lead_kept.extend(g)
                continue
            g_sorted = sorted(g, key=lambda n: n.pitch)
            lead_kept.extend(g_sorted[-lead_voices:])   # top-N pitches
            accomp_kept.extend(g_sorted[:-lead_voices]) # everything below
        lead_inst.notes  = sorted(lead_kept,  key=lambda n: n.start)
        accomp_inst.notes = sorted(accomp_kept, key=lambda n: n.start)

    return lead_pm, accomp_pm


def enforce_min_ioi(pm,
                    min_ioi_s: float,
                    per_pitch: bool = True,
                    global_window_s: float = 1.0,
                    global_max_onsets: int = 14):
    """Density limiter for plucked / short-decay targets.

    Two passes:
      A. Per-pitch minimum inter-onset — for each MIDI pitch, iterate notes
         chronologically and drop any whose start falls within `min_ioi_s`
         of the previously kept note. Ties broken by keeping higher velocity.
      B. Global 1s sliding-window rate cap at `global_max_onsets`. When the
         window exceeds the cap, drop the lowest-scored note in the window,
         where score = velocity × (1.0 if outer voice in its cluster else 0.6).
         14 onsets/s is the perceptual continuous-tone threshold (Fletcher &
         Rossing) — beyond it the ear stops hearing individual attacks.

    FluidSynth handles note-off / release naturally: dropping a note lets the
    previous pluck's SF2 release envelope continue undisturbed.
    """
    for inst in pm.instruments:
        # Pass A — per-pitch min IOI
        if per_pitch and min_ioi_s > 0:
            by_pitch: dict[int, list] = {}
            for n in sorted(inst.notes, key=lambda n: (n.pitch, n.start)):
                by_pitch.setdefault(n.pitch, []).append(n)
            kept = []
            for pitch, notes in by_pitch.items():
                last = None
                for n in notes:
                    if last is None or (n.start - last.start) >= min_ioi_s:
                        kept.append(n)
                        last = n
                    elif n.velocity > last.velocity:
                        kept.remove(last)
                        kept.append(n)
                        last = n
            inst.notes = sorted(kept, key=lambda n: n.start)

        # Pass B — global sliding-window cap
        if global_max_onsets and global_max_onsets > 0:
            notes = sorted(inst.notes, key=lambda n: n.start)
            # score each note against its onset cluster
            score = {}
            for g in _cluster_onsets(notes):
                if len(g) == 1:
                    score[id(g[0])] = float(g[0].velocity)
                else:
                    pitches = sorted(g, key=lambda n: n.pitch)
                    outer = {id(pitches[0]), id(pitches[-1])}
                    for n in g:
                        w = 1.0 if id(n) in outer else 0.6
                        score[id(n)] = w * float(n.velocity)

            kept = []
            for i, n in enumerate(notes):
                kept.append(n)
                # trim from tail of `kept` any notes older than the window
                cutoff = n.start - global_window_s
                # count onsets inside the window that are still in `kept`
                window = [m for m in kept if m.start >= cutoff]
                if len(window) > global_max_onsets:
                    # drop the weakest one currently in the window
                    weakest = min(window, key=lambda m: score.get(id(m), 0.0))
                    kept.remove(weakest)
            inst.notes = sorted(kept, key=lambda n: n.start)

    return pm


# ────────────────────────── registry ───────────────────────────────────────

ARRANGERS = {
    "piano_sustain": arrange_for_piano_sustain,
    "strings":       arrange_for_strings,
    "none":          lambda pm: pm,
}

"""Idiomatic MIDI arrangement per target instrument.

v4 — frame-level sustained-region detection + pattern selection by chord shape.

Why frame-level: onset-cluster detection (v2/v3) breaks up bowed string chords
whose onsets jitter beyond a 60 ms window. Iterating frames instead asks the
right question — "is the SAME set of pitches held for long enough?" — and
catches sustained polyphony regardless of onset alignment.

Patterns picked by shape (research: Alberti bass, murky bass, tremolando,
broken octaves, rocking thirds — see MODELS.md):
  2 notes         → rocking thirds/sixths (alternating pair)
  3 notes         → Alberti [low, high, mid, high]
  4+ notes        → broken-octave sweep
  hold >   1.0 s  → tremolando (split-halves, dense)

v3's REAL-PIANO guards are preserved: after a candidate region is found we
still check that (a) the notes making up that region end within a tight
stdev window and (b) their common hold exceeds the minimum. Piano chords
decay independently and fail these; strings ensembles release together and
pass.
"""
import numpy as np
import pretty_midi


# ─────────────────────────────── shared helpers ───────────────────────────────

def _all_notes(pm):
    out = []
    for inst in pm.instruments:
        out.extend((n, inst) for n in inst.notes)
    return out


def _alberti_order(n):
    if n <= 2:
        return list(range(n))
    if n == 3:
        return [0, 2, 1, 2]
    order = []
    for i in range(n // 2):
        order.extend([i, n - 1 - i])
    if n % 2:
        order.append(n // 2)
    return order


# ─────────────────────────────── strings ─────────────────────────────────────

def arrange_for_piano_sustain(pm, extend_s=0.30, min_note_len_for_ext_s=0.10):
    """Light note-end extension for piano TARGET when input was already piano.

    Basic Pitch strips real sustain-pedal information — every note ends when
    it fell below the frame threshold. The transcribed MIDI then reads as a
    series of short percussive attacks. Rendering that through the piano
    model produces the "chords hit hard, no sustain" complaint.

    This function extends each already-substantive note by up to extend_s to
    partially recover the pedaled feel. Not arpeggiation — no new notes are
    added and pitch structure is untouched. Just longer envelopes.
    """
    all_notes = [n for (n, _) in _all_notes(pm)]
    if not all_notes:
        return pm
    for n in all_notes:
        if (n.end - n.start) < min_note_len_for_ext_s:
            continue                                    # leave staccato/graces alone
        n.end = n.end + extend_s
    return pm


def arrange_for_strings(pm,
                        max_hold_s=0.8,
                        min_note_len_for_hold_s=0.20,
                        legato_overlap_s=0.04,
                        voice_semitone_window=3):
    """Bowed sustain — extend held notes to next onset in the voice.

    Skips already-short notes (< min_note_len_for_hold_s): extending them
    over-sustains what were meant to be short attacks."""
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


# ───────────────────── piano: frame scan + pattern picker ─────────────────────

def _sustained_regions(notes, hop_s=0.025, min_pitches=3, min_hold_s=0.35):
    """Find (start, end, pitches, contributing_notes) tuples where the SAME
    set of pitches (>= min_pitches) are all sounding for >= min_hold_s.

    Runs a frame-level scan (25 ms hop by default). At each candidate frame we
    look up which pitches are currently active and greedily extend the region
    forward as long as that same pitch set remains a subset of active pitches.
    Non-overlapping — once a region closes, we resume the scan after its end.
    """
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: n.start)
    t_end = max(n.end for n in notes)
    n_frames = int(t_end / hop_s) + 1

    # note-index intervals for fast lookup
    def active_at(t):
        return [n for n in notes if n.start <= t < n.end]

    regions = []
    i = 0
    while i < n_frames:
        t = i * hop_s
        active_pitches = {n.pitch for n in active_at(t)}
        if len(active_pitches) < min_pitches:
            i += 1
            continue
        # Take the CURRENT set of pitches as the region's "held" pitches. We do
        # NOT try to expand to a superset later — we only shrink (via `break`
        # below) if any pitch drops out. This is deliberate: adding a new voice
        # mid-region is a fresh musical event and shouldn't retroactively
        # rewrite the pattern we're building.
        region_pitches = frozenset(active_pitches)
        j = i + 1
        while j < n_frames:
            t2 = j * hop_s
            act2 = {n.pitch for n in active_at(t2)}
            if not region_pitches.issubset(act2):
                break
            j += 1
        region_end_t = (j - 1) * hop_s
        if region_end_t - t >= min_hold_s:
            contrib = [n for n in notes
                       if n.pitch in region_pitches
                       and n.start <= t + 0.05
                       and n.end   >= region_end_t - 0.02]
            regions.append((t, region_end_t, sorted(region_pitches), contrib))
            i = j                                      # skip past this region
        else:
            i += 1
    return regions


def _pick_pattern(pitches, duration_s):
    """Return (name, index_sequence, note_len_s). Indices point into `pitches`
    (or into an extended list — broken-octaves adds pi+12)."""
    n = len(pitches)
    if n == 2:
        return "rocking", [0, 1], 0.15                 # sixteenth-ish alternation
    if n == 3:
        return "alberti", [0, 2, 1, 2], 0.12
    if duration_s > 1.0 and n >= 4:
        # tremolando: split into halves, alternate — pattern is [L-block, H-block]
        return "tremolando", "_split_halves_", 0.07
    if n >= 4:
        # broken-octave sweep: for each chord tone, play tone then tone+12
        return "broken_oct", "_octave_sweep_", 0.10
    return "alberti", _alberti_order(n), 0.12          # fallback


def _emit_arpeggio(pattern_name, spec, note_len_s,
                   pitches, start_t, end_t, base_vel):
    """Return a list of pretty_midi.Note objects covering [start_t, end_t]."""
    notes = []

    if pattern_name == "tremolando":
        # split into low / high halves and alternate them as small chord-blocks
        L = pitches[:max(1, len(pitches) // 2)]
        H = pitches[len(pitches) // 2:] or L
        blocks = [L, H]
        k = 0; t = start_t
        while t + note_len_s * 0.5 < end_t:
            for p in blocks[k % 2]:
                notes.append(pretty_midi.Note(
                    velocity=base_vel,
                    pitch=p, start=t,
                    end=min(t + note_len_s * 0.9, end_t - 0.003)))
            t += note_len_s
            k += 1
        return notes

    if pattern_name == "broken_oct":
        # for each chord tone (low → high), play tone, tone+12 as consecutive 16ths
        seq = []
        for p in pitches:
            seq.extend([p, min(127, p + 12)])
        k = 0; t = start_t
        while t + note_len_s * 0.5 < end_t:
            p = seq[k % len(seq)]
            notes.append(pretty_midi.Note(
                velocity=base_vel, pitch=p, start=t,
                end=min(t + note_len_s * 0.9, end_t - 0.003)))
            t += note_len_s
            k += 1
        return notes

    # alberti / rocking — spec is a list of indices into pitches
    order = spec
    k = 0; t = start_t
    while t + note_len_s * 0.5 < end_t:
        idx = order[k % len(order)] if order else 0
        p = pitches[idx]
        notes.append(pretty_midi.Note(
            velocity=base_vel, pitch=p, start=t,
            end=min(t + note_len_s * 0.9, end_t - 0.003)))
        t += note_len_s
        k += 1
    return notes


def arrange_for_piano(pm,
                      hop_s=0.025,
                      min_pitches=3,
                      min_hold_s=0.35,
                      max_end_stdev_s=0.15,
                      preserve_attack_s=0.08,
                      arp_vel_frac=0.75):
    """v4 — frame-level sustained-region detection + pattern selection.

    For each region where >= min_pitches are held for >= min_hold_s AND the
    contributing notes end within a tight stdev window (rejects real piano
    chords whose voices decay independently), the sustained portion is
    replaced with an idiomatic pattern (Alberti / rocking / broken octaves /
    tremolando) chosen by chord shape and duration. Initial attack of the
    original chord is preserved for preserve_attack_s.
    """
    all_notes = [n for (n, _) in _all_notes(pm)]
    if not all_notes:
        return pm

    regions = _sustained_regions(all_notes, hop_s=hop_s,
                                 min_pitches=min_pitches,
                                 min_hold_s=min_hold_s)
    new_notes = []
    for start_t, end_t, pitches, contrib in regions:
        if len(contrib) < min_pitches:
            continue                                    # gate — spurious detection

        # v3 guard: contributing notes must release together (bowed) not
        # decay independently (piano). std of `.end` cheap to compute.
        end_stdev = float(np.std([n.end for n in contrib])) if len(contrib) > 1 else 0.0
        if end_stdev > max_end_stdev_s:
            continue

        base_vel = int(np.mean([n.velocity for n in contrib]) * arp_vel_frac)
        base_vel = max(30, min(110, base_vel))
        # truncate contributing notes to preserve just the initial attack
        arp_start = start_t + preserve_attack_s
        for n in contrib:
            n.end = min(n.end, start_t + preserve_attack_s + 0.02)

        name, spec, note_len_s = _pick_pattern(pitches, end_t - arp_start)
        new_notes.extend(_emit_arpeggio(name, spec, note_len_s,
                                        pitches, arp_start, end_t, base_vel))

    if new_notes:
        target = pm.instruments[0] if pm.instruments else None
        if target is None:
            target = pretty_midi.Instrument(program=0)
            pm.instruments.append(target)
        target.notes.extend(new_notes)

    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if n.end > n.start + 0.005]
    return pm


# ─────────────────────────────── self-test ───────────────────────────────────

if __name__ == "__main__":
    import copy

    def build(chord_pitches, hold_s, decay_stagger_s=0.0, program=0):
        """Helper: build a chord where each successive note ends `decay_stagger_s`
        earlier than the previous (piano-like when stagger > 0)."""
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=program)
        for k, p in enumerate(chord_pitches):
            inst.notes.append(pretty_midi.Note(
                velocity=80, pitch=p, start=0.0,
                end=hold_s - k * decay_stagger_s))
        pm.instruments.append(inst)
        return pm

    print("=== 3-note strings-like chord (uniform release) ===")
    pm = build([60, 64, 67], hold_s=1.0)
    n0 = len(pm.instruments[0].notes)
    pm2 = arrange_for_piano(copy.deepcopy(pm))
    print(f"  {n0} → {len(pm2.instruments[0].notes)} notes")
    print(f"  onsets: {sorted(round(n.start,3) for n in pm2.instruments[0].notes)}")

    print("\n=== 4-note strings-like chord, held 1.5s → should pick tremolando ===")
    pm = build([60, 64, 67, 72], hold_s=1.5)
    pm2 = arrange_for_piano(copy.deepcopy(pm))
    print(f"  {len(pm.instruments[0].notes)} → {len(pm2.instruments[0].notes)} notes")

    print("\n=== 4-note piano-like chord (varied decay) — should NOT arpeggiate ===")
    pm = build([60, 64, 67, 72], hold_s=1.0, decay_stagger_s=0.20)
    n0 = len(pm.instruments[0].notes)
    pm2 = arrange_for_piano(copy.deepcopy(pm))
    print(f"  {n0} → {len(pm2.instruments[0].notes)} notes  (expect same)")

    print("\n=== 2-note interval (dyad) held 0.6s → rocking thirds ===")
    pm = build([60, 64], hold_s=0.6)
    pm2 = arrange_for_piano(copy.deepcopy(pm), min_pitches=2)
    print(f"  {len(pm.instruments[0].notes)} → {len(pm2.instruments[0].notes)} notes")

    print("\n=== strings arranger on brief attack notes ===")
    pm = build([60, 64, 67], hold_s=0.15)                # too short to extend
    pm2 = arrange_for_strings(copy.deepcopy(pm))
    ends_before = sorted(round(n.end, 3) for n in pm.instruments[0].notes)
    ends_after = sorted(round(n.end, 3) for n in pm2.instruments[0].notes)
    print(f"  ends before: {ends_before}\n  ends after : {ends_after}   (unchanged is correct)")

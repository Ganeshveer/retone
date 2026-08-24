"""Idiomatic MIDI arrangement per target instrument.

The piano-roll -> mel model has no musical concept of "how would a real pianist
play a sustained string chord" — it renders whatever notes it's given, exactly.
So the arrangement burden falls on Stage-1.5, between transcription and render.

Two transforms, keyed by target instrument:

  arrange_for_piano(pm)
    A piano can't sustain like a bow. To imitate a sustained string bed on
    piano, real players do three things: (a) roll the chord (stagger onsets so
    the ear reads it as sustained motion, not a block strike); (b) restrike
    interior voices in an Alberti / broken-chord pattern to keep the sound
    moving; (c) leave the outer voices held. This function does all three.

    * Chord = notes with onsets within 25 ms of each other.
    * Rolled: >= 3-note chords get onsets spaced by 25 ms in low-to-high order.
    * Sustained (all cluster notes overlap for > 400 ms): re-strike inner
      voices every 300 ms in an alternating pattern; original notes still
      sustain underneath at their full length. Restrikes are 60% velocity and
      ~150 ms long — grace-note-scale.

  arrange_for_strings(pm)
    Piano transcription gives notes that decay. Strings sustain until the
    player lifts the bow. This function extends each note's END to the next
    onset in its voice (nearest-neighbor within ±3 semitones), capped at
    +2 s, with a 40 ms legato overlap between consecutive notes in a voice.
"""
import pretty_midi


def _all_notes(pm):
    out = []
    for inst in pm.instruments:
        out.extend((n, inst) for n in inst.notes)
    return out


def _alberti_order(n):
    """Broken-chord traversal order. Returns indices into the pitch-sorted chord.
    For n=3: outer-outer-inner-inner: low, hi, mid, hi
    For n>=4: alternate outer/inner pairs: 0, n-1, 1, n-2, 2, n-3, ..."""
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


def arrange_for_piano(pm,
                      chord_window_s=0.025,
                      roll_step_s=0.025,
                      restrike_period_s=0.30,
                      restrike_min_hold_s=0.40,
                      restrike_len_s=0.15,
                      restrike_vel_frac=0.60):
    all_notes = [n for (n, _) in _all_notes(pm)]
    if not all_notes:
        return pm
    all_notes.sort(key=lambda n: n.start)

    # 1) build clusters (onsets within chord_window_s)
    clusters = []
    cur = [all_notes[0]]
    for n in all_notes[1:]:
        if n.start - cur[0].start < chord_window_s:
            cur.append(n)
        else:
            clusters.append(cur); cur = [n]
    clusters.append(cur)

    # 2) roll clusters of 3+ notes
    for cl in clusters:
        if len(cl) < 3:
            continue
        cl.sort(key=lambda n: n.pitch)
        base = cl[0].start
        for i, n in enumerate(cl):
            # preserve total duration by shifting BOTH start and end
            dur = n.end - n.start
            n.start = base + i * roll_step_s
            n.end = n.start + dur

    # 3) restrike inner voices during held clusters
    new_notes = []
    for cl in clusters:
        if len(cl) < 3:
            continue
        cl_start = min(n.start for n in cl)
        cl_end = min(n.end for n in cl)              # only during the SHORTEST-held voice
        hold = cl_end - cl_start
        if hold < restrike_min_hold_s:
            continue
        cl_sorted = sorted(cl, key=lambda n: n.pitch)
        order = _alberti_order(len(cl_sorted))
        n_hits = max(0, int(hold / restrike_period_s) - 1)
        for k in range(n_hits):
            t = cl_start + (k + 1) * restrike_period_s
            if t >= cl_end - 0.05:
                break
            src = cl_sorted[order[k % len(order)]]
            new_notes.append(pretty_midi.Note(
                velocity=max(20, int(src.velocity * restrike_vel_frac)),
                pitch=src.pitch,
                start=t,
                end=min(t + restrike_len_s, cl_end - 0.02),
            ))

    if new_notes:
        # dump into first instrument (transcriber usually emits just one)
        (pm.instruments[0] if pm.instruments
         else pm.instruments.append(pretty_midi.Instrument(program=0)) or pm.instruments[0]).notes.extend(new_notes)
    return pm


def arrange_for_strings(pm,
                        max_hold_s=2.0,
                        legato_overlap_s=0.04,
                        voice_semitone_window=3):
    """Extend each note to the next onset in its voice (nearest pitch neighbor)."""
    all_notes = [n for (n, _) in _all_notes(pm)]
    if not all_notes:
        return pm
    all_notes.sort(key=lambda n: n.start)

    # for each note, find the next note within ±window semitones and after this note's start
    for i, n in enumerate(all_notes):
        target_end = n.start + max_hold_s
        for j in range(i + 1, len(all_notes)):
            m = all_notes[j]
            if m.start >= target_end:
                break
            if m.start <= n.start + 0.01:            # simultaneous / earlier — not "next"
                continue
            if abs(m.pitch - n.pitch) <= voice_semitone_window:
                target_end = m.start + legato_overlap_s
                break
        # only ever EXTEND, never shorten
        n.end = max(n.end, min(target_end, n.start + max_hold_s))
    return pm


if __name__ == "__main__":
    # sanity: build a 4-note block chord + a melody, run through both arrangers, print
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    # 4-note chord at t=0, held 1s
    for p in [60, 64, 67, 72]:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=0.0, end=1.0))
    # melody note at t=1.2
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=74, start=1.2, end=1.4))
    pm.instruments.append(inst)

    import copy
    piano_pm = arrange_for_piano(copy.deepcopy(pm))
    strings_pm = arrange_for_strings(copy.deepcopy(pm))

    print(f"original notes: {len(pm.instruments[0].notes)}")
    print(f"after arrange_for_piano: {len(piano_pm.instruments[0].notes)}")
    print("  onsets:", sorted(round(n.start, 3) for n in piano_pm.instruments[0].notes))
    print(f"after arrange_for_strings: {len(strings_pm.instruments[0].notes)}")
    print("  (start, end) pairs:", sorted((round(n.start, 3), round(n.end, 3))
                                          for n in strings_pm.instruments[0].notes))

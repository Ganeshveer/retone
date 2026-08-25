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
import numpy as np
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
                      chord_window_s=0.06,
                      roll_step_s=0.025,
                      arp_min_hold_s=0.35,
                      arp_min_notes=4,
                      arp_note_len_s=0.12,
                      preserve_attack_s=0.08,
                      arp_vel_frac=0.75,
                      max_end_stdev_s=0.15):
    """Sustained N-note chord clusters get REPLACED, in place, with a rolling
    Alberti-style arpeggio for their sustained portion. The initial attack of
    the original chord is preserved (~preserve_attack_s), then original notes
    are truncated and the remainder is filled with broken-chord notes.

    Why replace, not decorate: an earlier version added decorative restrikes on
    top of the held chord, but the block chord underneath still dominated — the
    render still sounded "hit and hold." Real pianists don't hold a big chord to
    imitate strings; they play a rolling pattern from the outset (Alberti bass,
    broken thirds, murky bass). This function does that.
    """
    all_notes = [n for (n, _) in _all_notes(pm)]
    if not all_notes:
        return pm
    all_notes.sort(key=lambda n: n.start)

    # 1) build clusters using wider window (Basic Pitch onset jitter ~20-40 ms)
    clusters = []
    cur = [all_notes[0]]
    for n in all_notes[1:]:
        if n.start - cur[0].start < chord_window_s:
            cur.append(n)
        else:
            clusters.append(cur); cur = [n]
    clusters.append(cur)

    # 2) roll onsets of 3+ note clusters (staggered ascending)
    for cl in clusters:
        if len(cl) < arp_min_notes:
            continue
        cl.sort(key=lambda n: n.pitch)
        base = cl[0].start
        for i, n in enumerate(cl):
            dur = n.end - n.start
            n.start = base + i * roll_step_s
            n.end = n.start + dur

    # 3) REPLACE the sustained portion of long-held clusters with an arpeggio.
    #    Two guards to avoid over-arpeggiating REAL piano performances:
    #      (a) require the COMMON hold (min end - start) to be > arp_min_hold_s
    #          so we only fire on uniformly-sustained chords (bowed strings) and
    #          NOT on piano-style block chords where notes decay independently.
    #      (b) require the stdev of note ends to be small — a real piano attack
    #          has notes ending at wildly different times (release+decay+damper);
    #          a bowed string chord has all voices ending near-simultaneously.
    arp_notes = []
    for cl in clusters:
        if len(cl) < arp_min_notes:
            continue
        cl_start = min(n.start for n in cl)
        cl_end   = max(n.end   for n in cl)
        common_end = min(n.end for n in cl)
        common_hold = common_end - cl_start
        end_stdev = float(np.std([n.end for n in cl])) if len(cl) > 1 else 0.0

        # gate (a): common hold long enough
        if common_hold < arp_min_hold_s:
            continue
        # gate (b): note ends bunched — chord released as one, not piano decay
        if end_stdev > max_end_stdev_s:
            continue

        arp_start = cl_start + preserve_attack_s
        # truncate original notes so the arpeggio replaces the sustain
        for n in cl:
            n.end = min(n.end, cl_start + preserve_attack_s + 0.02)

        cl_sorted = sorted(cl, key=lambda n: n.pitch)
        pitches = [n.pitch for n in cl_sorted]
        base_vel = int(np.mean([n.velocity for n in cl_sorted]) * arp_vel_frac) if pitches else 60
        # Alberti/interleaved order — creates the "rock back and forth" motion
        order = _alberti_order(len(pitches))

        t = arp_start
        k = 0
        while t + 0.5 * arp_note_len_s < cl_end:
            p = pitches[order[k % len(order)]]
            arp_notes.append(pretty_midi.Note(
                velocity=max(30, min(110, base_vel)),
                pitch=p,
                start=t,
                end=min(t + arp_note_len_s * 0.95, cl_end - 0.005),  # tiny gap between arp notes
            ))
            t += arp_note_len_s
            k += 1

    if arp_notes:
        target = pm.instruments[0] if pm.instruments else None
        if target is None:
            target = pretty_midi.Instrument(program=0)
            pm.instruments.append(target)
        target.notes.extend(arp_notes)

    # Purge any notes whose end fell before start after truncation
    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if n.end > n.start + 0.005]
    return pm


def arrange_for_strings(pm,
                        max_hold_s=0.8,
                        min_note_len_for_hold_s=0.20,
                        legato_overlap_s=0.04,
                        voice_semitone_window=3):
    """Bowed sustain — but only for notes that were ALREADY held long enough to be
    genuine sustained tones. Short attack notes (<200 ms) stay short; extending them
    over-sustains the render and sounded worse than PLAIN on some test material."""
    """Extend each note to the next onset in its voice (nearest pitch neighbor)."""
    all_notes = [n for (n, _) in _all_notes(pm)]
    if not all_notes:
        return pm
    all_notes.sort(key=lambda n: n.start)

    # for each note, find the next note within ±window semitones and after this note's start
    for i, n in enumerate(all_notes):
        # skip staccato/attack-only notes — extending them over-sustains
        if (n.end - n.start) < min_note_len_for_hold_s:
            continue
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

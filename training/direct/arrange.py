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


def clamp_to_range(pm, lo: int, hi: int, mode: str = "drop"):
    """Bring every note into [lo, hi].

    mode="drop"    — silently drop out-of-range notes. Default. Safe choice
                     when the accompaniment already carries the harmony —
                     out-of-range notes just disappear from the lead.
    mode="octave"  — fold by whole octaves (preserves pitch class). Musically
                     preserves pitch class but on a narrow-range target
                     (viola 48-88) an A0 folds to A#3, materially changing
                     the harmony. Only use when the target owns the full
                     playable range and there is no accompaniment to catch
                     the dropped notes.
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


def _skyline_split(pm, lead_voices: int = 1):
    """Original zero-parameter skyline: top-N pitches per cluster → lead.

    Kept as a fallback / A-B baseline. In practice `split_lead_accompaniment`
    (weighted-skyline with hysteresis) is the shipping algorithm.
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
            lead_kept.extend(g_sorted[-lead_voices:])
            accomp_kept.extend(g_sorted[:-lead_voices])
        lead_inst.notes   = sorted(lead_kept,   key=lambda n: n.start)
        accomp_inst.notes = sorted(accomp_kept, key=lambda n: n.start)

    return lead_pm, accomp_pm


# ────────────────── weighted-skyline melody extraction ─────────────────────
#
# Pure skyline (Uitdenbogerd & Zobel 1998) picks the top pitch of every onset
# cluster. Famously brittle: a grace note above the melody hijacks the line
# for one beat; every note of an arpeggio becomes "lead" because clusters
# are single-note; a mid-voice melody under piano chord tones gets lost.
#
# The revised algorithm (post-Chai 2000, Rizo 2006, Jiang SMC 2019, jSymbolic
# feature research) scores every candidate note against the last-picked lead
# note using several complementary signals, then applies register hysteresis
# so a plausible melodic voice is preserved across brief absences.
#
# Weights below are hand-tuned starting points; they are exposed as arguments
# so they can be sweep-tuned on real content if a subjective A/B pushes them.

def _note_scores(cluster, prev_lead_pitch,
                 pitch_mean, pitch_std,
                 w_pitch=1.5, w_dur=0.8, w_vel=0.6, w_lock=2.0,
                 register_lock_semitones=7):
    """Return the score of every note in `cluster` against the running lead.

    Higher = more melody-like. Components:
      + `w_pitch * pitch_z`         — piece-wide register normalization
      + `w_dur   * log(duration+ε)` — sustained notes are more melodic
      + `w_vel   * velocity / 127`  — louder notes carry the tune
      + `w_lock  * register_lock`   — reward staying near prev_lead within
                                      ±register_lock_semitones, penalize
                                      octave jumps
    """
    import math
    out = []
    for n in cluster:
        pitch_z = (n.pitch - pitch_mean) / pitch_std if pitch_std > 0 else 0.0
        dur     = math.log(1e-3 + (n.end - n.start))
        vel     = n.velocity / 127.0
        if prev_lead_pitch is None:
            lock = 0.0
        else:
            d = abs(n.pitch - prev_lead_pitch)
            if d <= register_lock_semitones:
                lock = 1.0 - (d / register_lock_semitones) * 0.3    # ∈ [0.7, 1.0]
            else:
                # Penalize octave-plus jumps hard
                lock = -min(1.0, (d - register_lock_semitones) / 12.0)
        out.append(w_pitch * pitch_z + w_dur * dur + w_vel * vel + w_lock * lock)
    return out


def split_lead_accompaniment(pm,
                             tol: float = CLUSTER_TOL,
                             register_lock_semitones: int = 7,
                             hysteresis_s: float = 0.5,
                             lead_overlap_s: float = 0.0,
                             score_threshold: float | None = None,
                             bootstrap_top_pitch: bool = True,
                             min_lead_hold_s: float = 0.08):
    """Weighted-skyline melody extraction with register hysteresis.

    Returns (lead_pm, accomp_pm) — both are deep copies.

    Improvements over pure skyline (see `_skyline_split` for the baseline):

    1. **Weighted scoring** — for each onset cluster, every candidate note is
       scored on pitch_z, log(duration), velocity, and register continuity
       with the previously chosen lead. Argmax → lead; the rest go to accomp.
       Fixes: grace-note hijacks (short high notes don't score), mid-voice
       melodies (a G4 sustained under a C5 chord tone can still win because
       duration + register-lock outweigh pitch_z).

    2. **Register lock (±`register_lock_semitones`)** — once a lead is
       established, notes within that window get a strong bonus, notes
       further away get penalized. This keeps a coherent melodic line
       through arpeggios: only the arpeggio note nearest the melody register
       stays on lead; the rest fall to accompaniment.

    3. **Hysteresis (`hysteresis_s`)** — after that many seconds with no
       candidate scoring above `score_threshold`, unlock the register so a
       genuine new melodic line can be bootstrapped. Prevents the algorithm
       from locking onto a wandering register.

    4. **Lead overlap (`lead_overlap_s`)** — each lead note is stretched
       slightly so it overlaps its successor. FluidSynth (and any GM/legato
       renderer) sees this as legato instead of hard note-off; when combined
       with `apply_mono_legato`, it produces smooth transitions.

    5. **Guaranteed monophonic output** — after selection, any overlapping
       lead notes are truncated so the lead line is strictly mono at MIDI
       level, regardless of downstream mono-mode support.

    Fails on: fugues / contrapuntal music (multiple equally-important
    voices), piano concertos with mid-voice countermelodies, jazz solos in
    the same register as loud comping. These will always be judgment calls;
    document at the pipeline level.
    """
    import math

    lead_pm   = copy.deepcopy(pm)
    accomp_pm = copy.deepcopy(pm)

    for lead_inst, accomp_inst in zip(lead_pm.instruments, accomp_pm.instruments):
        all_notes = lead_inst.notes
        if not all_notes:
            lead_inst.notes   = []
            accomp_inst.notes = []
            continue

        # Piece-wide pitch stats for z-scoring
        pitches = [n.pitch for n in all_notes]
        pitch_mean = float(sum(pitches)) / len(pitches)
        pitch_std  = (
            math.sqrt(sum((p - pitch_mean) ** 2 for p in pitches) / len(pitches))
            if len(pitches) > 1 else 1.0
        )

        clusters = _cluster_onsets(all_notes, tol=tol)
        lead_kept, accomp_kept = [], []
        prev_lead_pitch = None
        prev_lead_time  = None

        for g in clusters:
            # Bootstrap the first cluster: if there is no prev_lead OR
            # hysteresis expired, fall back to the highest note in the
            # cluster to seed the melodic line.
            hysteresis_expired = (
                prev_lead_time is not None
                and (g[0].start - prev_lead_time) > hysteresis_s
            )
            if prev_lead_pitch is None or hysteresis_expired:
                if bootstrap_top_pitch:
                    lead_note = max(g, key=lambda n: n.pitch)
                else:
                    scores = _note_scores(g, None, pitch_mean, pitch_std,
                                          register_lock_semitones=register_lock_semitones)
                    lead_note = g[scores.index(max(scores))]
                lead_kept.append(lead_note)
                for n in g:
                    if n is not lead_note:
                        accomp_kept.append(n)
                prev_lead_pitch = lead_note.pitch
                prev_lead_time  = lead_note.start
                continue

            # Regular cluster: score, pick argmax. If score_threshold is set
            # and the winner is still below it, route the whole cluster to
            # accompaniment (creates a gap on lead) — off by default so
            # every cluster contributes something to the lead line.
            scores = _note_scores(g, prev_lead_pitch, pitch_mean, pitch_std,
                                  register_lock_semitones=register_lock_semitones)
            best_i    = scores.index(max(scores))
            best_note = g[best_i]
            best_score = scores[best_i]

            if score_threshold is None or best_score >= score_threshold:
                lead_kept.append(best_note)
                for n in g:
                    if n is not best_note:
                        accomp_kept.append(n)
                prev_lead_pitch = best_note.pitch
                prev_lead_time  = best_note.start
            else:
                accomp_kept.extend(g)

        # Enforce monophonic on the lead line without over-truncating: never
        # cut a note below `min_lead_hold_s`, and never chop more than half
        # of its original duration for an ornament-sized successor.
        lead_sorted = sorted(lead_kept, key=lambda n: n.start)
        for i in range(len(lead_sorted) - 1):
            cur, nxt = lead_sorted[i], lead_sorted[i + 1]
            orig_dur = cur.end - cur.start
            target_end = nxt.start + lead_overlap_s
            floor = cur.start + max(min_lead_hold_s, 0.5 * orig_dur)
            if cur.end > target_end:
                cur.end = max(target_end, floor)

        lead_inst.notes   = lead_sorted
        accomp_inst.notes = sorted(accomp_kept, key=lambda n: n.start)

    return lead_pm, accomp_pm


def apply_mono_legato(pm, mono=True, legato=False):
    """Insert MIDI CC 126 (Mono Mode On, value=1) at t=0 for every instrument.

    CC 126 value=1 is the unambiguous "one-voice mono" — the value=0 special
    case has implementation-defined behavior across MIDI hosts. FluidSynth
    understands the value=1 form.

    CC 68 (Legato Footswitch) is OFF by default because most SoundFont-based
    synths — FluidSynth's default engine included — ignore it, and asserting
    it can produce inconsistent results across builds.

    Safe to call unconditionally: applying to a poly instrument that ignores
    the CC is a no-op.
    """
    for inst in pm.instruments:
        if mono:
            inst.control_changes.append(
                pretty_midi.ControlChange(number=126, value=1, time=0.0)
            )
        if legato:
            inst.control_changes.append(
                pretty_midi.ControlChange(number=68, value=127, time=0.0)
            )
    return pm


def enforce_min_ioi(pm,
                    min_ioi_s: float,
                    per_pitch: bool = True,
                    global_window_s: float = 1.0,
                    global_max_onsets: int | None = 14):
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

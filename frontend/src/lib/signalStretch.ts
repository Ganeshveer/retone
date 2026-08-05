import SignalsmithStretch from "signalsmith-stretch";

/**
 * Offline, formant-preserving pitch shift of a mono/stereo buffer, returning audio of the
 * SAME length (pitch shift, not time-stretch), using Signalsmith Stretch.
 *
 * Runs inside an OfflineAudioContext in buffer-fed mode. Because an AudioWorklet only
 * initializes once the offline render thread is running, we use the `suspend(0)` pattern:
 * start rendering (which spins up the worklet thread) but immediately pause at t=0, wait
 * for the node's async "ready" handshake, feed buffers + schedule the shift, then resume.
 * Without this, `await SignalsmithStretch(offlineCtx)` deadlocks.
 */
export async function offlinePitchShift(
  channels: Float32Array[],
  sampleRate: number,
  semitones: number,
  opts: { formantCompensation?: boolean } = {}
): Promise<Float32Array[]> {
  const numCh = channels.length;
  const length = channels[0]?.length ?? 0;
  if (length === 0 || Math.abs(semitones) < 0.001) {
    return channels.map((c) => c.slice());
  }

  // Extra tail covers the node's processing latency; we slice the aligned region back out.
  const extra = Math.ceil(0.6 * sampleRate);
  const ctx = new OfflineAudioContext(numCh, length + extra, sampleRate);

  // Buffer-fed mode: no live inputs.
  const nodePromise = SignalsmithStretch(ctx, {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [numCh],
  });

  const suspendPromise = ctx.suspend(0); // pause at t=0 so the worklet can initialize
  const renderPromise = ctx.startRendering();
  await suspendPromise;

  const node = await nodePromise; // now resolves — worklet thread is alive
  node.connect(ctx.destination);
  await node.addBuffers(channels);
  await node.schedule({
    output: 0,
    active: true,
    input: 0,
    rate: 1,
    semitones,
    tonalityHz: 8000,
    formantSemitones: 0,
    formantCompensation: opts.formantCompensation ?? true,
    formantBaseHz: 0,
  });

  await ctx.resume();
  const rendered = await renderPromise;

  const out: Float32Array[] = [];
  for (let c = 0; c < numCh; c++) {
    out.push(rendered.getChannelData(c).slice(0, length));
  }
  return out;
}

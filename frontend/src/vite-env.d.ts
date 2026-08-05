/// <reference types="vite/client" />

declare module "signalsmith-stretch" {
  /** Async factory: returns an AudioWorkletNode with Signalsmith Stretch methods attached. */
  interface StretchNode extends AudioWorkletNode {
    addBuffers(channels: Float32Array[]): Promise<number>;
    schedule(obj: Record<string, number | boolean>): Promise<unknown>;
    start(when?: number, offset?: number, duration?: number, rate?: number, semitones?: number): Promise<unknown>;
    stop(when?: number): Promise<unknown>;
    configure(cfg: Record<string, number | boolean>): Promise<unknown>;
    latency(): Promise<number>;
    inputTime: number;
  }
  const SignalsmithStretch: (
    audioContext: BaseAudioContext,
    options?: AudioWorkletNodeOptions
  ) => Promise<StretchNode>;
  export default SignalsmithStretch;
}

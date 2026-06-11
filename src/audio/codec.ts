const MODEL_CODEBOOKS = 8;
const BITS_PER_CODE = 10;
const TOKEN_STEP_SAMPLES = 1_920;
const TOKEN_STEP_MS = 80;
const TOKEN_STEPS = 1;
const FRAME_MS = TOKEN_STEP_MS;
const ACTIVE_CODEBOOKS = MODEL_CODEBOOKS;
const CODE_BYTES = Math.ceil((TOKEN_STEPS * ACTIVE_CODEBOOKS * BITS_PER_CODE) / 8);
const PACKET_BYTES = CODE_BYTES;

export const CIAO_ACTIVE_CODEC = {
  id: 'mimi-streaming',
  label: 'Mimi',
  bitrate: Math.round((PACKET_BYTES * 8 * 1000) / FRAME_MS),
  sampleRate: 24_000,
  channels: 1,
  frameMs: FRAME_MS,
  tokenStepMs: TOKEN_STEP_MS,
  tokenStepSamples: TOKEN_STEP_SAMPLES,
  tokenSteps: TOKEN_STEPS,
  inputSamples: TOKEN_STEP_SAMPLES * TOKEN_STEPS,
  playoutSamples: TOKEN_STEP_SAMPLES * TOKEN_STEPS,
  rawDecoderSamples: TOKEN_STEP_SAMPLES * TOKEN_STEPS,
  modelCodebooks: MODEL_CODEBOOKS,
  codebooks: ACTIVE_CODEBOOKS,
  bitsPerCode: BITS_PER_CODE,
} as const;

export const CODEC_FRAME_MS = CIAO_ACTIVE_CODEC.frameMs;
export const CODEC_CODE_BYTES = CODE_BYTES;
export const CODEC_PACKET_BYTES = PACKET_BYTES;

export type CiaoCodec = {
  readonly id: typeof CIAO_ACTIVE_CODEC.id;
  readonly label: typeof CIAO_ACTIVE_CODEC.label;
  readonly bitrate: typeof CIAO_ACTIVE_CODEC.bitrate;
  readonly frameMs: typeof CIAO_ACTIVE_CODEC.frameMs;
  encode(pcm: Float32Array, sampleRate: number): Promise<Uint8Array>;
  decode(packet: Uint8Array, sampleRate: number): Promise<Float32Array>;
  reset(): void;
};

export type CodecModelCacheProgress = {
  status: 'checking' | 'cached' | 'downloading' | 'ready';
  loadedBytes: number;
  totalBytes: number;
  percent: number;
};

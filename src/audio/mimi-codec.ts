import { CIAO_ACTIVE_CODEC, CODEC_PACKET_BYTES, type CiaoCodec } from './codec';
import {
  createInitialMimiState,
  disposeState,
  disposeTensor,
  loadMimiRuntime,
  updateMimiState,
  warmMimiModelCache,
  type MimiOrtFeeds,
  type MimiOrtOutputs,
  type MimiOrtTensor,
  type MimiRuntime,
} from './mimi-runtime';

type InferencePriority = 'normal' | 'high';
type InferenceTask<T> = {
  priority: InferencePriority;
  work: () => Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
};

const MIMI_SAMPLE_RATE = CIAO_ACTIVE_CODEC.sampleRate;
const MIMI_FRAME_SAMPLES = CIAO_ACTIVE_CODEC.inputSamples;
const MIMI_PLAYOUT_SAMPLES = CIAO_ACTIVE_CODEC.playoutSamples;
const MIMI_TOKEN_STEPS = CIAO_ACTIVE_CODEC.tokenSteps;
const highPriorityInferenceQueue: InferenceTask<unknown>[] = [];
const normalInferenceQueue: InferenceTask<unknown>[] = [];
let inferenceQueueRunning = false;
let pooledCodec: MimiOnnxStreamingCodec | null = null;
let pooledCodecPromise: Promise<MimiOnnxStreamingCodec> | null = null;

export async function createCodec(): Promise<CiaoCodec> {
  const codec = await preparePooledCodec();
  if (pooledCodec === codec) {
    pooledCodec = null;
  }

  return codec;
}

export async function prepareCodecRuntime() {
  await loadMimiRuntime();
}

export async function prewarmCodec() {
  await preparePooledCodec();
}

export function isCodecPrewarmed() {
  return Boolean(pooledCodec);
}

export { warmMimiModelCache };

function preparePooledCodec() {
  if (pooledCodec) {
    return Promise.resolve(pooledCodec);
  }

  if (!pooledCodecPromise) {
    pooledCodecPromise = createPrimedCodec()
      .then((codec) => {
        pooledCodec = codec;
        return codec;
      })
      .finally(() => {
        pooledCodecPromise = null;
      });
  }

  return pooledCodecPromise;
}

async function createPrimedCodec() {
  const startedAt = performance.now();
  console.info('ciao codec prewarm start');
  const runtime = await loadMimiRuntime();
  const codec = new MimiOnnxStreamingCodec(runtime);
  await codec.warmup();
  console.info(`ciao codec prewarm ready in ${Math.round(performance.now() - startedAt)}ms`);
  return codec;
}

class MimiOnnxStreamingCodec implements CiaoCodec {
  readonly id = CIAO_ACTIVE_CODEC.id;
  readonly label = CIAO_ACTIVE_CODEC.label;
  readonly bitrate = CIAO_ACTIVE_CODEC.bitrate;
  readonly frameMs = CIAO_ACTIVE_CODEC.frameMs;
  private readonly runtime: MimiRuntime;
  private encoderState: MimiOrtFeeds;
  private decoderState: MimiOrtFeeds;
  private encodeBuffer = new Float32Array(0);
  private generation = 0;

  constructor(runtime: MimiRuntime) {
    this.runtime = runtime;
    this.assertModelShape();
    this.encoderState = createInitialMimiState(runtime.ort, runtime.stateSpec.encoder);
    this.decoderState = createInitialMimiState(runtime.ort, runtime.stateSpec.decoder);
  }

  encode(pcm: Float32Array, sampleRate: number) {
    const generation = this.generation;
    return enqueueInference(() => this.encodeUnsafe(pcm, sampleRate, generation), 'normal');
  }

  decode(packet: Uint8Array, sampleRate: number) {
    const generation = this.generation;
    return enqueueInference(() => this.decodeUnsafe(packet, sampleRate, generation), 'high');
  }

  async warmup() {
    const generation = this.generation;
    await enqueueInference(async () => {
      const packet = await this.encodeFrameUnsafe(new Float32Array(MIMI_FRAME_SAMPLES), MIMI_SAMPLE_RATE, generation);
      if (packet.byteLength > 0 && generation === this.generation) {
        await this.decodeUnsafe(packet, MIMI_SAMPLE_RATE, generation);
      }
      if (generation === this.generation) {
        this.reset();
      }
    }, 'high');
  }

  reset() {
    this.generation += 1;
    disposeState(this.encoderState);
    disposeState(this.decoderState);
    this.encoderState = createInitialMimiState(this.runtime.ort, this.runtime.stateSpec.encoder);
    this.decoderState = createInitialMimiState(this.runtime.ort, this.runtime.stateSpec.decoder);
    this.encodeBuffer = new Float32Array(0);
  }

  private async encodeUnsafe(pcm: Float32Array, sampleRate: number, generation: number) {
    if (generation !== this.generation) {
      return new Uint8Array();
    }

    const pcm24 = sampleRate === MIMI_SAMPLE_RATE ? pcm : resampleLinear(pcm, sampleRate, MIMI_SAMPLE_RATE);
    this.encodeBuffer = concatFloat32(this.encodeBuffer, pcm24);

    if (this.encodeBuffer.length < MIMI_FRAME_SAMPLES) {
      return new Uint8Array();
    }

    const frame = this.encodeBuffer.slice(0, MIMI_FRAME_SAMPLES);
    this.encodeBuffer = this.encodeBuffer.slice(MIMI_FRAME_SAMPLES);
    return this.encodeFrameUnsafe(frame, MIMI_SAMPLE_RATE, generation);
  }

  private async encodeFrameUnsafe(frame: Float32Array, sampleRate: number, generation: number) {
    if (generation !== this.generation) {
      return new Uint8Array();
    }

    const frame24 = sampleRate === MIMI_SAMPLE_RATE ? frame : resampleLinear(frame, sampleRate, MIMI_SAMPLE_RATE);
    const currentFrame = trimOrPadFrame(frame24, MIMI_FRAME_SAMPLES);
    const previousState = this.encoderState;
    const feeds: MimiOrtFeeds = {
      input_values: new this.runtime.ort.Tensor('float32', currentFrame, [1, 1, MIMI_FRAME_SAMPLES]),
      ...previousState,
    };

    const outputs = await this.runtime.encoder.run(feeds);
    disposeTensor(feeds.input_values);

    if (generation !== this.generation) {
      disposeMimiOutputs(outputs, false, 'audio_codes');
      return new Uint8Array();
    }

    this.encoderState = updateMimiState(outputs);
    disposeState(previousState);

    const packet = packCodes(outputs.audio_codes);
    disposeTensor(outputs.audio_codes);
    return packet;
  }

  private async decodeUnsafe(packet: Uint8Array, sampleRate: number, generation: number) {
    if (generation !== this.generation) {
      return new Float32Array();
    }

    const codes = unpackCodes(packet);
    const previousState = this.decoderState;
    const feeds: MimiOrtFeeds = {
      audio_codes: new this.runtime.ort.Tensor('int64', codes, [1, CIAO_ACTIVE_CODEC.modelCodebooks, MIMI_TOKEN_STEPS]),
      ...previousState,
    };

    const outputs = await this.runtime.decoder.run(feeds);
    disposeTensor(feeds.audio_codes);

    if (generation !== this.generation) {
      disposeMimiOutputs(outputs, false, 'audio_values');
      return new Float32Array();
    }

    this.decoderState = updateMimiState(outputs);
    disposeState(previousState);

    const audio = outputs.audio_values;
    if (!audio) {
      throw new Error('Mimi decoder output incompleto');
    }

    const mono = trimOrPadFrame((audio.data as Float32Array).slice(0, MIMI_PLAYOUT_SAMPLES), MIMI_PLAYOUT_SAMPLES);
    disposeTensor(audio);
    return sampleRate === MIMI_SAMPLE_RATE ? mono : resampleLinear(mono, MIMI_SAMPLE_RATE, sampleRate);
  }

  private assertModelShape() {
    if (
      CIAO_ACTIVE_CODEC.modelCodebooks !== 8 ||
      CIAO_ACTIVE_CODEC.codebooks !== 8 ||
      CIAO_ACTIVE_CODEC.bitsPerCode !== 11 ||
      CIAO_ACTIVE_CODEC.inputSamples !== 1_920
    ) {
      throw new Error('Mimi codec config non compatibile');
    }
  }
}

function enqueueInference<T>(work: () => Promise<T>, priority: InferencePriority) {
  return new Promise<T>((resolve, reject) => {
    const task: InferenceTask<T> = { priority, work, resolve, reject };
    if (priority === 'high') {
      highPriorityInferenceQueue.push(task as InferenceTask<unknown>);
    } else {
      normalInferenceQueue.push(task as InferenceTask<unknown>);
    }

    drainInferenceQueue();
  });
}

function drainInferenceQueue() {
  if (inferenceQueueRunning) {
    return;
  }

  inferenceQueueRunning = true;
  void (async () => {
    for (;;) {
      const task = highPriorityInferenceQueue.shift() ?? normalInferenceQueue.shift();
      if (!task) {
        inferenceQueueRunning = false;
        return;
      }

      try {
        task.resolve(await task.work());
      } catch (error) {
        task.reject(error);
      }
    }
  })();
}

export function packCodes(tensor: MimiOrtTensor | undefined) {
  if (!tensor) {
    throw new Error('Mimi encoder output incompleto');
  }

  const layout = mimiCodeLayout(tensor.dims);
  if (layout.frames !== MIMI_TOKEN_STEPS) {
    throw new Error(`Mimi token steps inattesi: ${layout.frames}`);
  }

  const data = tensor.data as BigInt64Array | Int32Array | number[];
  const packet = new Uint8Array(CODEC_PACKET_BYTES);
  let bitOffset = 0;
  for (let frame = 0; frame < MIMI_TOKEN_STEPS; frame += 1) {
    for (let codebook = 0; codebook < CIAO_ACTIVE_CODEC.codebooks; codebook += 1) {
      const value = Number(data[layout.index(codebook, frame)]) & 0x7ff;
      writeBits(packet, bitOffset, value, CIAO_ACTIVE_CODEC.bitsPerCode);
      bitOffset += CIAO_ACTIVE_CODEC.bitsPerCode;
    }
  }

  return packet;
}

export function unpackCodes(packet: Uint8Array) {
  if (packet.byteLength !== CODEC_PACKET_BYTES) {
    throw new Error('payload Mimi non valido');
  }

  const data = new BigInt64Array(CIAO_ACTIVE_CODEC.modelCodebooks * MIMI_TOKEN_STEPS);
  let bitOffset = 0;

  for (let frame = 0; frame < MIMI_TOKEN_STEPS; frame += 1) {
    for (let codebook = 0; codebook < CIAO_ACTIVE_CODEC.codebooks; codebook += 1) {
      const value = readBits(packet, bitOffset, CIAO_ACTIVE_CODEC.bitsPerCode);
      data[codebook * MIMI_TOKEN_STEPS + frame] = BigInt(value);
      bitOffset += CIAO_ACTIVE_CODEC.bitsPerCode;
    }
  }

  return data;
}

function mimiCodeLayout(dims: readonly number[]) {
  if (dims.length !== 3) {
    throw new Error(`Mimi audio_codes shape inattesa: [${dims.join(', ')}]`);
  }

  if (dims[0] === 1 && dims[1] === CIAO_ACTIVE_CODEC.modelCodebooks) {
    return {
      frames: Number(dims[2] ?? 0),
      index: (codebook: number, frame: number) => codebook * Number(dims[2] ?? 0) + frame,
    };
  }

  if (dims[0] === 1 && dims[2] === CIAO_ACTIVE_CODEC.modelCodebooks) {
    return {
      frames: Number(dims[1] ?? 0),
      index: (codebook: number, frame: number) => frame * CIAO_ACTIVE_CODEC.modelCodebooks + codebook,
    };
  }

  throw new Error(`Mimi audio_codes shape non compatibile: [${dims.join(', ')}]`);
}

function disposeMimiOutputs(outputs: MimiOrtOutputs, keepState: boolean, primaryOutput: string) {
  disposeTensor(outputs[primaryOutput]);
  if (keepState) {
    return;
  }

  for (const [name, tensor] of Object.entries(outputs)) {
    if (name !== primaryOutput) {
      disposeTensor(tensor);
    }
  }
}

function trimOrPadFrame(frame: Float32Array, samples: number) {
  if (frame.length === samples) {
    return frame;
  }

  if (frame.length > samples) {
    return frame.slice(0, samples);
  }

  const output = new Float32Array(samples);
  output.set(frame);
  return output;
}

function writeBits(target: Uint8Array, bitOffset: number, value: number, bits: number) {
  for (let bit = 0; bit < bits; bit += 1) {
    if (value & (1 << bit)) {
      const offset = bitOffset + bit;
      target[offset >> 3] |= 1 << (offset & 7);
    }
  }
}

function readBits(source: Uint8Array, bitOffset: number, bits: number) {
  let value = 0;

  for (let bit = 0; bit < bits; bit += 1) {
    const offset = bitOffset + bit;
    if (source[offset >> 3] & (1 << (offset & 7))) {
      value |= 1 << bit;
    }
  }

  return value;
}

function concatFloat32(left: Float32Array, right: Float32Array) {
  const output = new Float32Array(left.length + right.length);
  output.set(left);
  output.set(right, left.length);
  return output;
}

function resampleLinear(input: Float32Array, fromRate: number, toRate: number) {
  const ratio = fromRate / toRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outputLength);

  for (let i = 0; i < output.length; i += 1) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(input.length - 1, left + 1);
    const mix = position - left;
    output[i] = input[left] * (1 - mix) + input[right] * mix;
  }

  return output;
}

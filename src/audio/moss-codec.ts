import { CIAO_ACTIVE_CODEC, CODEC_CODE_BYTES, type CiaoCodec } from './codec';
import {
  createInitialMossDecoderState,
  disposeState,
  disposeTensor,
  loadMossRuntime,
  updateMossDecoderState,
  warmMossModelCache,
  type MossRuntime,
  type OrtFeeds,
  type OrtOutputs,
  type OrtTensor,
} from './moss-runtime';

type InferencePriority = 'normal' | 'high';
type InferenceTask<T> = {
  priority: InferencePriority;
  work: () => Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
};

const MOSS_SAMPLE_RATE = CIAO_ACTIVE_CODEC.sampleRate;
const MOSS_CHANNELS = CIAO_ACTIVE_CODEC.channels;
const MOSS_FRAME_SAMPLES = CIAO_ACTIVE_CODEC.inputSamples;
const MOSS_PLAYOUT_SAMPLES = CIAO_ACTIVE_CODEC.playoutSamples;
const MOSS_TOKEN_STEPS = CIAO_ACTIVE_CODEC.tokenSteps;
const highPriorityInferenceQueue: InferenceTask<unknown>[] = [];
const normalInferenceQueue: InferenceTask<unknown>[] = [];
let inferenceQueueRunning = false;

export async function createCodec(): Promise<CiaoCodec> {
  const runtime = await loadMossRuntime();
  return new MossOnnxStreamingCodec(runtime);
}

export async function prepareCodecRuntime() {
  await loadMossRuntime();
}

export { warmMossModelCache };

class MossOnnxStreamingCodec implements CiaoCodec {
  readonly id = CIAO_ACTIVE_CODEC.id;
  readonly label = CIAO_ACTIVE_CODEC.label;
  readonly bitrate = CIAO_ACTIVE_CODEC.bitrate;
  readonly frameMs = CIAO_ACTIVE_CODEC.frameMs;
  private readonly runtime: MossRuntime;
  private decoderState: OrtFeeds;
  private encodeBuffer = new Float32Array(0);
  private generation = 0;

  constructor(runtime: MossRuntime) {
    this.runtime = runtime;
    this.assertModelShape();
    this.decoderState = createInitialMossDecoderState(runtime.ort, runtime.meta);
  }

  encode(pcm: Float32Array, sampleRate: number) {
    const generation = this.generation;
    return enqueueInference(() => this.encodeUnsafe(pcm, sampleRate, generation), 'normal');
  }

  decode(packet: Uint8Array, sampleRate: number) {
    const generation = this.generation;
    return enqueueInference(() => this.decodeUnsafe(packet, sampleRate, generation), 'high');
  }

  reset() {
    this.generation += 1;
    disposeState(this.decoderState);
    this.decoderState = createInitialMossDecoderState(this.runtime.ort, this.runtime.meta);
    this.encodeBuffer = new Float32Array(0);
  }

  private async encodeUnsafe(pcm: Float32Array, sampleRate: number, generation: number) {
    if (generation !== this.generation) {
      return new Uint8Array();
    }

    const pcm48 = sampleRate === MOSS_SAMPLE_RATE ? pcm : resampleLinear(pcm, sampleRate, MOSS_SAMPLE_RATE);
    this.encodeBuffer = concatFloat32(this.encodeBuffer, pcm48);

    if (this.encodeBuffer.length < MOSS_FRAME_SAMPLES) {
      return new Uint8Array();
    }

    const frame = this.encodeBuffer.slice(0, MOSS_FRAME_SAMPLES);
    this.encodeBuffer = this.encodeBuffer.slice(MOSS_FRAME_SAMPLES);

    const waveform = monoToChannelMajor(frame, MOSS_CHANNELS);
    const feeds = {
      waveform: new this.runtime.ort.Tensor('float32', waveform, [1, MOSS_CHANNELS, MOSS_FRAME_SAMPLES]),
      input_lengths: new this.runtime.ort.Tensor('int32', new Int32Array([MOSS_FRAME_SAMPLES]), [1]),
    };

    const outputs = await this.runtime.encoder.run(feeds);
    disposeTensor(feeds.waveform);
    disposeTensor(feeds.input_lengths);

    if (generation !== this.generation) {
      disposeMossEncodeOutputs(outputs);
      return new Uint8Array();
    }

    const packet = packCodes(outputs.audio_codes);
    disposeMossEncodeOutputs(outputs);
    return packet;
  }

  private async decodeUnsafe(packet: Uint8Array, sampleRate: number, generation: number) {
    if (generation !== this.generation) {
      return new Float32Array();
    }

    const codes = unpackCodes(packet);
    const previousState = this.decoderState;
    const feeds: OrtFeeds = {
      audio_codes: new this.runtime.ort.Tensor('int32', codes.data, [
        1,
        codes.frames,
        CIAO_ACTIVE_CODEC.modelCodebooks,
      ]),
      audio_code_lengths: new this.runtime.ort.Tensor('int32', new Int32Array([codes.decodeLength]), [1]),
      ...previousState,
    };

    const outputs = await this.runtime.decoder.run(feeds);
    disposeTensor(feeds.audio_codes);
    disposeTensor(feeds.audio_code_lengths);

    if (generation !== this.generation) {
      disposeMossDecodeOutputs(outputs);
      return new Float32Array();
    }

    this.decoderState = updateMossDecoderState(outputs, this.runtime.meta);
    disposeState(previousState);

    const audio = outputs.audio;
    if (!audio) {
      throw new Error('MOSS decoder output incompleto');
    }

    const mono = trimOrPadFrame(channelMajorToMono(audio.data as Float32Array, audio.dims), MOSS_PLAYOUT_SAMPLES);
    disposeTensor(outputs.audio);
    disposeTensor(outputs.audio_lengths);
    return sampleRate === MOSS_SAMPLE_RATE ? mono : resampleLinear(mono, MOSS_SAMPLE_RATE, sampleRate);
  }

  private assertModelShape() {
    const config = this.runtime.meta.codec_config;
    if (
      config.sample_rate !== MOSS_SAMPLE_RATE ||
      config.channels !== MOSS_CHANNELS ||
      config.num_quantizers !== CIAO_ACTIVE_CODEC.modelCodebooks
    ) {
      throw new Error('MOSS model config non compatibile');
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

export function packCodes(tensor: OrtTensor | undefined) {
  if (!tensor) {
    throw new Error('MOSS encoder output incompleto');
  }

  const layout = mossCodeLayout(tensor.dims);
  const data = tensor.data as BigInt64Array | Int32Array | number[];
  if (layout.frames !== MOSS_TOKEN_STEPS) {
    throw new Error(`MOSS token steps inattesi: ${layout.frames}`);
  }

  const payloadBytes = Math.ceil(
    (layout.frames * CIAO_ACTIVE_CODEC.codebooks * CIAO_ACTIVE_CODEC.bitsPerCode) / 8,
  );
  const packet = new Uint8Array(payloadBytes);
  let bitOffset = 0;
  for (let frame = 0; frame < layout.frames; frame += 1) {
    for (let codebook = 0; codebook < CIAO_ACTIVE_CODEC.codebooks; codebook += 1) {
      const value = Number(data[layout.index(codebook, frame)]) & 0x3ff;
      writeBits(packet, bitOffset, value, CIAO_ACTIVE_CODEC.bitsPerCode);
      bitOffset += CIAO_ACTIVE_CODEC.bitsPerCode;
    }
  }

  return packet;
}

export function unpackCodes(packet: Uint8Array) {
  if (packet.byteLength !== CODEC_CODE_BYTES) {
    throw new Error('payload MOSS non valido');
  }

  const frames = MOSS_TOKEN_STEPS;
  const data = new Int32Array(CIAO_ACTIVE_CODEC.modelCodebooks * frames);
  let bitOffset = 0;

  for (let frame = 0; frame < frames; frame += 1) {
    for (let codebook = 0; codebook < CIAO_ACTIVE_CODEC.codebooks; codebook += 1) {
      const value = readBits(packet, bitOffset, CIAO_ACTIVE_CODEC.bitsPerCode);
      data[frame * CIAO_ACTIVE_CODEC.modelCodebooks + codebook] = value;
      bitOffset += CIAO_ACTIVE_CODEC.bitsPerCode;
    }
  }

  return { data, frames, decodeLength: Math.max(1, frames - 1) };
}

function mossCodeLayout(dims: readonly number[]) {
  if (dims.length !== 3) {
    throw new Error(`MOSS audio_codes shape inattesa: [${dims.join(', ')}]`);
  }

  if (dims[0] === CIAO_ACTIVE_CODEC.modelCodebooks && dims[1] === 1) {
    return {
      frames: Number(dims[2] ?? 0),
      index: (codebook: number, frame: number) => codebook * Number(dims[2] ?? 0) + frame,
    };
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

  throw new Error(`MOSS audio_codes shape non compatibile: [${dims.join(', ')}]`);
}

function disposeMossEncodeOutputs(outputs: OrtOutputs) {
  disposeTensor(outputs.audio_codes);
  disposeTensor(outputs.audio_code_lengths);
}

function disposeMossDecodeOutputs(outputs: OrtOutputs) {
  disposeTensor(outputs.audio);
  disposeTensor(outputs.audio_lengths);
  for (const [name, tensor] of Object.entries(outputs)) {
    if (name !== 'audio' && name !== 'audio_lengths') {
      disposeTensor(tensor);
    }
  }
}

function monoToChannelMajor(frame: Float32Array, channels: number) {
  const output = new Float32Array(channels * frame.length);

  for (let channel = 0; channel < channels; channel += 1) {
    output.set(frame, channel * frame.length);
  }

  return output;
}

function channelMajorToMono(data: Float32Array, dims: readonly number[]) {
  const channels = Number(dims[1] ?? 1);
  const samples = Number(dims[2] ?? data.length);

  if (channels <= 1) {
    return data.slice(0, samples);
  }

  const mono = new Float32Array(samples);
  for (let sample = 0; sample < samples; sample += 1) {
    let value = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      value += data[channel * samples + sample] ?? 0;
    }
    mono[sample] = value / channels;
  }

  return mono;
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

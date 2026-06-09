import { CIAO_ACTIVE_CODEC, CODEC_CODE_BYTES } from '../audio/codec';

export const MIN_AUDIO_AGGREGATION = 1;
export const MAX_AUDIO_AGGREGATION = 2;

export type RemoteAudioFrame = {
  sequence: number;
  sentAt: number;
  receivedAt: number;
  payload: Uint8Array;
};

export type DecodedAudioPayload = {
  dtx: boolean;
  frames: Uint8Array[];
};

export type StreamFeedback = {
  highestSequence: number;
  receivedFrames: number;
  lostFrames: number;
  jitterMs: number;
  bufferFrames: number;
  targetAggregation: number;
};

const AUDIO_PAYLOAD_MAGIC = 0x4d;
const AUDIO_PAYLOAD_VERSION = 6;
const AUDIO_PAYLOAD_HEADER_BYTES = 16;
const AUDIO_FLAG_DTX = 1;

const FEEDBACK_MAGIC = 0x53;
const FEEDBACK_VERSION = 1;
const FEEDBACK_BYTES = 16;

export function encodeAudioPayload(frames: Uint8Array[], options: { dtx?: boolean } = {}) {
  const frameCount = frames.length;

  if (frameCount > MAX_AUDIO_AGGREGATION) {
    throw new Error(`troppi frame audio aggregati: ${frameCount}`);
  }

  for (const frame of frames) {
    if (frame.byteLength !== CODEC_CODE_BYTES) {
      throw new Error(`payload MOSS non valido: ${frame.byteLength}/${CODEC_CODE_BYTES} bytes`);
    }
  }

  const output = new Uint8Array(AUDIO_PAYLOAD_HEADER_BYTES + frameCount * CODEC_CODE_BYTES);
  const view = new DataView(output.buffer);
  output[0] = AUDIO_PAYLOAD_MAGIC;
  output[1] = AUDIO_PAYLOAD_VERSION;
  output[2] = options.dtx ? AUDIO_FLAG_DTX : 0;
  output[3] = frameCount;
  output[4] = CIAO_ACTIVE_CODEC.tokenSteps;
  output[5] = CIAO_ACTIVE_CODEC.codebooks;
  output[6] = CIAO_ACTIVE_CODEC.bitsPerCode;
  output[7] = CIAO_ACTIVE_CODEC.modelCodebooks;
  view.setUint16(8, CIAO_ACTIVE_CODEC.inputSamples, true);
  view.setUint16(10, CIAO_ACTIVE_CODEC.playoutSamples, true);
  view.setUint16(12, CIAO_ACTIVE_CODEC.rawDecoderSamples, true);
  view.setUint16(14, CIAO_ACTIVE_CODEC.sampleRate, true);

  for (let index = 0; index < frameCount; index += 1) {
    output.set(frames[index], AUDIO_PAYLOAD_HEADER_BYTES + index * CODEC_CODE_BYTES);
  }

  return output;
}

export function decodeAudioPayload(payload: Uint8Array): DecodedAudioPayload | null {
  if (payload.byteLength < AUDIO_PAYLOAD_HEADER_BYTES) {
    return null;
  }

  if (payload[0] !== AUDIO_PAYLOAD_MAGIC || payload[1] !== AUDIO_PAYLOAD_VERSION) {
    return null;
  }

  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const frameCount = payload[3];
  const expectedBytes = AUDIO_PAYLOAD_HEADER_BYTES + frameCount * CODEC_CODE_BYTES;

  if (
    frameCount > MAX_AUDIO_AGGREGATION ||
    payload.byteLength !== expectedBytes ||
    payload[4] !== CIAO_ACTIVE_CODEC.tokenSteps ||
    payload[5] !== CIAO_ACTIVE_CODEC.codebooks ||
    payload[6] !== CIAO_ACTIVE_CODEC.bitsPerCode ||
    payload[7] !== CIAO_ACTIVE_CODEC.modelCodebooks ||
    view.getUint16(8, true) !== CIAO_ACTIVE_CODEC.inputSamples ||
    view.getUint16(10, true) !== CIAO_ACTIVE_CODEC.playoutSamples ||
    view.getUint16(12, true) !== CIAO_ACTIVE_CODEC.rawDecoderSamples ||
    view.getUint16(14, true) !== CIAO_ACTIVE_CODEC.sampleRate
  ) {
    return null;
  }

  const frames: Uint8Array[] = [];
  for (let index = 0; index < frameCount; index += 1) {
    const offset = AUDIO_PAYLOAD_HEADER_BYTES + index * CODEC_CODE_BYTES;
    frames.push(payload.slice(offset, offset + CODEC_CODE_BYTES));
  }

  return {
    dtx: Boolean(payload[2] & AUDIO_FLAG_DTX),
    frames,
  };
}

export function encodeStreamFeedback(feedback: StreamFeedback) {
  const output = new Uint8Array(FEEDBACK_BYTES);
  const view = new DataView(output.buffer);
  output[0] = FEEDBACK_MAGIC;
  output[1] = FEEDBACK_VERSION;
  output[2] = clampByte(feedback.bufferFrames);
  output[3] = clampByte(feedback.targetAggregation);
  view.setUint32(4, feedback.highestSequence >>> 0, true);
  view.setUint16(8, clampUint16(feedback.receivedFrames), true);
  view.setUint16(10, clampUint16(feedback.lostFrames), true);
  view.setUint16(12, clampUint16(Math.round(feedback.jitterMs)), true);
  view.setUint16(14, 0, true);
  return output;
}

export function decodeStreamFeedback(payload: Uint8Array): StreamFeedback | null {
  if (payload.byteLength !== FEEDBACK_BYTES || payload[0] !== FEEDBACK_MAGIC || payload[1] !== FEEDBACK_VERSION) {
    return null;
  }

  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  return {
    bufferFrames: payload[2],
    targetAggregation: payload[3],
    highestSequence: view.getUint32(4, true),
    receivedFrames: view.getUint16(8, true),
    lostFrames: view.getUint16(10, true),
    jitterMs: view.getUint16(12, true),
  };
}

export function clampAggregation(value: number) {
  return Math.max(MIN_AUDIO_AGGREGATION, Math.min(MAX_AUDIO_AGGREGATION, Math.round(value)));
}

function clampByte(value: number) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function clampUint16(value: number) {
  return Math.max(0, Math.min(0xffff, Math.round(value)));
}

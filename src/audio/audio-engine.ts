import { CODEC_FRAME_MS, type CiaoCodec } from './codec';
import { type RemoteAudioFrame } from '../webrtc/streaming';

const AUDIO_WORKLET_URL = '/worklets/ciao-audio-worklet.js?v=6';
const MAX_PENDING_CAPTURE_FRAMES = 3;
const MIN_REMOTE_BUFFER_TARGET_FRAMES = 1;
const MAX_REMOTE_BUFFER_TARGET_FRAMES = 3;
const MAX_REMOTE_BUFFER_FRAMES = 12;
const REMOTE_BUFFER_TARGET_UPDATE_MS = 1_000;
const REMOTE_BUFFER_TARGET_RELAX_MS = 8_000;
const REMOTE_JITTER_GROW_MS = CODEC_FRAME_MS * 0.7;
const REMOTE_JITTER_SHRINK_MS = CODEC_FRAME_MS * 0.3;
const REMOTE_STATS_IDLE_RESET_MS = 2_000;
const REMOTE_DECODE_LEAD_MIN_MS = Math.min(80, CODEC_FRAME_MS * 0.25);
const REMOTE_DECODE_LEAD_MAX_MS = CODEC_FRAME_MS * 0.75;
const REMOTE_DECODE_LEAD_SAFETY_MS = 32;
const REMOTE_MISSING_RETRY_MS = Math.min(24, CODEC_FRAME_MS / 16);
const REMOTE_MISSING_RETRY_LIMIT = 8;

export type AudioPerformanceSample =
  | {
      kind: 'encode';
      durationMs: number;
      captureBacklogMs: number;
      pendingCaptureFrames: number;
      droppedCaptureFrames: number;
    }
  | {
      kind: 'decode';
      durationMs: number;
      remoteBufferFrames: number;
    }
  | {
      kind: 'capture-drop';
      captureBacklogMs: number;
      pendingCaptureFrames: number;
      droppedCaptureFrames: number;
    };

type AudioEngineOptions = {
  codec: CiaoCodec;
  onEncodedFrame: (frame: Uint8Array) => void;
  onDtx: () => void;
  onError: (error: unknown) => void;
  onPerformance: (sample: AudioPerformanceSample) => void;
  vadEnabled?: boolean;
  voiceProcessing?: boolean;
};

export class CiaoAudioEngine {
  private readonly codec: CiaoCodec;
  private readonly onEncodedFrame: (frame: Uint8Array) => void;
  private readonly onDtx: () => void;
  private readonly onError: (error: unknown) => void;
  private readonly onPerformance: (sample: AudioPerformanceSample) => void;
  private readonly vadEnabled: boolean;
  private readonly voiceProcessing: boolean;
  private context: AudioContext | null = null;
  private captureNode: AudioWorkletNode | null = null;
  private playbackNode: AudioWorkletNode | null = null;
  private contextInit: Promise<AudioContext> | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private micStream: MediaStream | null = null;
  private vadHangoverChunks = 0;
  private wasSpeaking = false;
  private lastDtxAt = 0;
  private remoteBuffer = new Map<number, RemoteAudioFrame>();
  private expectedRemoteSequence: number | null = null;
  private remoteDtxSequence: number | null = null;
  private remotePlayoutTimer: number | undefined;
  private lastDecodedFrame: Float32Array | null = null;
  private consecutiveLosses = 0;
  private targetRemoteBufferFrames = MIN_REMOTE_BUFFER_TARGET_FRAMES;
  private remoteUnderrunsWindow = 0;
  private remoteJitterMs = 0;
  private lastRemoteArrivalAt = 0;
  private lastRemoteSequence: number | null = null;
  private lastRemoteTargetUpdateAt = 0;
  private remoteStableSince = 0;
  private remoteLateWaits = 0;
  private remoteDecodeLeadMs = REMOTE_DECODE_LEAD_MIN_MS;
  private failed = false;
  private pendingCaptureFrames = 0;
  private droppedCaptureFrames = 0;

  constructor(options: AudioEngineOptions) {
    this.codec = options.codec;
    this.onEncodedFrame = options.onEncodedFrame;
    this.onDtx = options.onDtx;
    this.onError = options.onError;
    this.onPerformance = options.onPerformance;
    this.vadEnabled = options.vadEnabled ?? true;
    this.voiceProcessing = options.voiceProcessing ?? true;
  }

  async startCapture(micStream?: MediaStream) {
    const context = await this.ensureContext();
    if (this.failed) {
      micStream?.getTracks().forEach((track) => track.stop());
      return;
    }

    await context.resume();

    if (this.captureNode) {
      micStream?.getTracks().forEach((track) => track.stop());
      return;
    }

    this.micStream = micStream ?? (await requestMicrophoneStream(this.voiceProcessing));

    this.sourceNode = context.createMediaStreamSource(this.micStream);
    this.captureNode = new AudioWorkletNode(context, 'ciao-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
    });

    this.captureNode.port.onmessage = async (event: MessageEvent<Float32Array>) => {
      if (this.failed) {
        return;
      }

      const pcm = event.data;
      const level = levelFromFrame(pcm);
      const captureFrameMs = (pcm.length / context.sampleRate) * 1_000;

      if (this.vadEnabled && !this.shouldEncodeCaptureFrame(level)) {
        return;
      }

      if (this.pendingCaptureFrames >= MAX_PENDING_CAPTURE_FRAMES) {
        this.droppedCaptureFrames += 1;
        this.onPerformance({
          kind: 'capture-drop',
          captureBacklogMs: this.pendingCaptureFrames * captureFrameMs,
          pendingCaptureFrames: this.pendingCaptureFrames,
          droppedCaptureFrames: this.droppedCaptureFrames,
        });
        return;
      }

      this.pendingCaptureFrames += 1;
      try {
        const startedAt = performance.now();
        const encoded = await this.codec.encode(pcm, context.sampleRate);
        if (this.failed) {
          return;
        }
        const durationMs = performance.now() - startedAt;
        if (encoded.byteLength > 0 || durationMs > captureFrameMs) {
          this.onPerformance({
            kind: 'encode',
            durationMs,
            captureBacklogMs: this.pendingCaptureFrames * captureFrameMs,
            pendingCaptureFrames: this.pendingCaptureFrames,
            droppedCaptureFrames: this.droppedCaptureFrames,
          });
        }
        if (encoded.byteLength > 0) {
          this.onEncodedFrame(encoded);
        }
      } catch (error) {
        this.fail(error);
      } finally {
        this.pendingCaptureFrames = Math.max(0, this.pendingCaptureFrames - 1);
      }
    };

    this.sourceNode.connect(this.captureNode);
  }

  stopCapture() {
    this.captureNode?.disconnect();
    this.sourceNode?.disconnect();
    this.captureNode = null;
    this.sourceNode = null;
    this.micStream?.getTracks().forEach((track) => track.stop());
    this.micStream = null;
  }

  async receiveRemoteFrames(frames: RemoteAudioFrame[]) {
    if (this.failed || frames.length === 0) {
      return;
    }

    try {
      const context = await this.ensureContext();
      await context.resume();

      const acceptedFrames: RemoteAudioFrame[] = [];

      for (const frame of frames) {
        if (this.expectedRemoteSequence !== null && frame.sequence < this.expectedRemoteSequence) {
          continue;
        }

        this.remoteBuffer.set(frame.sequence, frame);
        acceptedFrames.push(frame);
      }

      if (acceptedFrames.length === 0) {
        return;
      }

      this.updateRemoteArrivalStats(acceptedFrames);
      this.trimRemoteBuffer();
      this.remoteDtxSequence = null;

      if (this.expectedRemoteSequence === null) {
        this.expectedRemoteSequence = Math.min(...acceptedFrames.map((frame) => frame.sequence));
      }

      this.updateRemoteBufferTarget();

      if (this.remoteBuffer.size >= this.targetRemoteBufferFrames) {
        this.scheduleRemotePlayout(0);
      }
    } catch (error) {
      this.fail(error);
    }
  }

  receiveRemoteDtx(sequence: number) {
    this.remoteDtxSequence = sequence;
  }

  getReceiveBufferDepth() {
    return this.remoteBuffer.size;
  }

  dispose() {
    this.failed = true;
    this.stopCapture();
    this.stopRemotePlayout();
    this.playbackNode?.disconnect();
    this.playbackNode = null;
    this.contextInit = null;
    this.context?.close();
    this.context = null;
  }

  private async ensureContext() {
    if (this.context && this.playbackNode) {
      return this.context;
    }

    if (this.contextInit) {
      return this.contextInit;
    }

    const context = this.context ?? new AudioContext({ latencyHint: 'interactive' });
    this.context = context;

    const init = (async () => {
      await context.audioWorklet.addModule(AUDIO_WORKLET_URL);

      if (this.context !== context) {
        throw new Error('audio context replaced during worklet init');
      }

      this.playbackNode = new AudioWorkletNode(context, 'ciao-playback', {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      this.playbackNode.connect(context.destination);
      return context;
    })();

    this.contextInit = init;

    try {
      return await init;
    } finally {
      if (this.contextInit === init) {
        this.contextInit = null;
      }
    }
  }

  private shouldEncodeCaptureFrame(level: number) {
    const isSpeaking = level > 0.035;

    if (isSpeaking) {
      this.vadHangoverChunks = 12;
      this.wasSpeaking = true;
      return true;
    }

    if (this.vadHangoverChunks > 0) {
      this.vadHangoverChunks -= 1;
      return true;
    }

    if (this.wasSpeaking) {
      this.wasSpeaking = false;
      this.sendDtxMarker(true);
    } else {
      this.sendDtxMarker(false);
    }

    return false;
  }

  private sendDtxMarker(force: boolean) {
    const now = performance.now();

    if (!force && now - this.lastDtxAt < 1_000) {
      return;
    }

    this.lastDtxAt = now;
    this.onDtx();
  }

  private scheduleRemotePlayout(delayMs: number = CODEC_FRAME_MS) {
    if (this.remotePlayoutTimer !== undefined || this.expectedRemoteSequence === null) {
      return;
    }

    this.remotePlayoutTimer = window.setTimeout(() => {
      this.remotePlayoutTimer = undefined;
      void this.playRemoteTick();
    }, delayMs);
  }

  private async playRemoteTick() {
    if (this.failed) {
      return;
    }

    try {
      if (this.expectedRemoteSequence === null) {
        return;
      }

      const context = await this.ensureContext();
      if (this.failed) {
        return;
      }

      const sequence = this.expectedRemoteSequence;
      const buffered = this.remoteBuffer.get(sequence);
      let decoded: Float32Array;
      let decodedRealFrame = false;

      if (buffered) {
        this.remoteBuffer.delete(sequence);
        decoded = await this.decodeTimed(buffered.payload, context.sampleRate);
        this.lastDecodedFrame = decoded;
        this.consecutiveLosses = 0;
        this.remoteLateWaits = 0;
        decodedRealFrame = true;
      } else if (
        this.remoteDtxSequence !== null &&
        sequence >= this.remoteDtxSequence &&
        this.remoteBuffer.size === 0
      ) {
        this.stopRemotePlayout();
        return;
      } else {
        if (this.remoteLateWaits < REMOTE_MISSING_RETRY_LIMIT && this.remoteBuffer.size < this.targetRemoteBufferFrames) {
          this.remoteLateWaits += 1;
          this.scheduleRemotePlayout(REMOTE_MISSING_RETRY_MS);
          return;
        }

        const alreadyCountedAsLate = this.remoteLateWaits > 0;
        this.remoteLateWaits = 0;
        if (!alreadyCountedAsLate) {
          this.remoteUnderrunsWindow += 1;
        }
        this.consecutiveLosses += 1;
        this.updateRemoteBufferTarget(performance.now(), true);
        decoded = this.concealRemoteFrame(context.sampleRate);
      }

      if (this.failed) {
        return;
      }

      this.postPlaybackFrame(decoded);
      this.expectedRemoteSequence = (sequence + 1) >>> 0;

      if (this.consecutiveLosses >= 8 && this.remoteBuffer.size === 0) {
        this.stopRemotePlayout();
        return;
      }

      this.scheduleRemotePlayout(decodedRealFrame ? this.nextRemoteDecodeDelay() : CODEC_FRAME_MS);
    } catch (error) {
      this.fail(error);
    }
  }

  private fail(error: unknown) {
    if (this.failed) {
      return;
    }

    this.failed = true;
    this.onError(error);
  }

  private concealRemoteFrame(sampleRate: number) {
    return this.copyLastFrame(sampleRate);
  }

  private async decodeTimed(packet: Uint8Array, sampleRate: number) {
    const startedAt = performance.now();
    const decoded = await this.codec.decode(packet, sampleRate);
    const durationMs = performance.now() - startedAt;
    this.updateRemoteDecodeLead(durationMs);
    this.onPerformance({
      kind: 'decode',
      durationMs,
      remoteBufferFrames: this.remoteBuffer.size,
    });
    return decoded;
  }

  private updateRemoteDecodeLead(durationMs: number) {
    const targetLead = Math.max(
      REMOTE_DECODE_LEAD_MIN_MS,
      Math.min(REMOTE_DECODE_LEAD_MAX_MS, durationMs + REMOTE_DECODE_LEAD_SAFETY_MS),
    );
    this.remoteDecodeLeadMs = this.remoteDecodeLeadMs * 0.7 + targetLead * 0.3;
  }

  private nextRemoteDecodeDelay() {
    return Math.max(0, CODEC_FRAME_MS - this.remoteDecodeLeadMs);
  }

  private copyLastFrame(sampleRate: number) {
    if (!this.lastDecodedFrame) {
      return new Float32Array(samplesPerCodecFrame(sampleRate));
    }

    const copy = new Float32Array(this.lastDecodedFrame);
    this.lastDecodedFrame = fadeFrame(copy, this.consecutiveLosses);
    return this.lastDecodedFrame;
  }

  private postPlaybackFrame(frame: Float32Array) {
    const playbackFrame = new Float32Array(frame);
    this.playbackNode?.port.postMessage(playbackFrame, [playbackFrame.buffer]);
  }

  private stopRemotePlayout() {
    if (this.remotePlayoutTimer !== undefined) {
      window.clearTimeout(this.remotePlayoutTimer);
      this.remotePlayoutTimer = undefined;
    }

    this.expectedRemoteSequence = null;
    this.remoteDtxSequence = null;
    this.consecutiveLosses = 0;
    this.resetRemoteBufferAdaptation();
  }

  private trimRemoteBuffer() {
    if (this.remoteBuffer.size <= MAX_REMOTE_BUFFER_FRAMES) {
      return;
    }

    const sequences = [...this.remoteBuffer.keys()].sort((left, right) => left - right);
    for (const sequence of sequences.slice(0, this.remoteBuffer.size - MAX_REMOTE_BUFFER_FRAMES)) {
      this.remoteBuffer.delete(sequence);
    }
  }

  private updateRemoteArrivalStats(frames: RemoteAudioFrame[]) {
    const orderedFrames = [...frames].sort((left, right) => left.sequence - right.sequence);
    const firstFrame = orderedFrames[0];
    const lastFrame = orderedFrames[orderedFrames.length - 1];

    if (!firstFrame || !lastFrame) {
      return;
    }

    if (
      this.lastRemoteSequence !== null &&
      this.lastRemoteArrivalAt > 0 &&
      firstFrame.sequence > this.lastRemoteSequence
    ) {
      const sequenceDelta = firstFrame.sequence - this.lastRemoteSequence;
      const expectedDelta = sequenceDelta * CODEC_FRAME_MS;
      const arrivalDelta = Math.max(0, firstFrame.receivedAt - this.lastRemoteArrivalAt);

      if (expectedDelta < REMOTE_STATS_IDLE_RESET_MS && arrivalDelta < REMOTE_STATS_IDLE_RESET_MS) {
        const variation = Math.abs(arrivalDelta - expectedDelta);
        this.remoteJitterMs += (variation - this.remoteJitterMs) / 16;
      } else {
        this.remoteJitterMs = 0;
      }
    }

    if (this.lastRemoteSequence === null || lastFrame.sequence >= this.lastRemoteSequence) {
      this.lastRemoteSequence = lastFrame.sequence;
      this.lastRemoteArrivalAt = lastFrame.receivedAt;
    }
  }

  private updateRemoteBufferTarget(now = performance.now(), force = false) {
    const shouldGrow =
      this.remoteUnderrunsWindow >= 2 ||
      this.consecutiveLosses >= 2 ||
      this.remoteJitterMs > REMOTE_JITTER_GROW_MS;

    if (shouldGrow) {
      if (this.targetRemoteBufferFrames < MAX_REMOTE_BUFFER_TARGET_FRAMES) {
        this.targetRemoteBufferFrames += 1;
      }

      this.lastRemoteTargetUpdateAt = now;
      this.remoteUnderrunsWindow = 0;
      this.remoteStableSince = 0;
      return;
    }

    if (force) {
      return;
    }

    if (!force && now - this.lastRemoteTargetUpdateAt < REMOTE_BUFFER_TARGET_UPDATE_MS) {
      return;
    }

    this.lastRemoteTargetUpdateAt = now;

    const isStable =
      this.remoteUnderrunsWindow === 0 &&
      this.consecutiveLosses === 0 &&
      this.remoteJitterMs < REMOTE_JITTER_SHRINK_MS;

    this.remoteUnderrunsWindow = 0;

    if (!isStable) {
      this.remoteStableSince = 0;
      return;
    }

    if (this.remoteStableSince === 0) {
      this.remoteStableSince = now;
      return;
    }

    if (
      this.targetRemoteBufferFrames > MIN_REMOTE_BUFFER_TARGET_FRAMES &&
      now - this.remoteStableSince >= REMOTE_BUFFER_TARGET_RELAX_MS
    ) {
      this.targetRemoteBufferFrames -= 1;
      this.remoteStableSince = now;
    }
  }

  private resetRemoteBufferAdaptation() {
    this.targetRemoteBufferFrames = MIN_REMOTE_BUFFER_TARGET_FRAMES;
    this.remoteUnderrunsWindow = 0;
    this.remoteJitterMs = 0;
    this.lastRemoteArrivalAt = 0;
    this.lastRemoteSequence = null;
    this.lastRemoteTargetUpdateAt = 0;
    this.remoteStableSince = 0;
    this.remoteLateWaits = 0;
    this.remoteDecodeLeadMs = REMOTE_DECODE_LEAD_MIN_MS;
  }
}

export function requestMicrophoneStream(voiceProcessing: boolean) {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      autoGainControl: false,
      channelCount: 1,
      echoCancellation: voiceProcessing,
      noiseSuppression: voiceProcessing,
    },
    video: false,
  });
}

function levelFromFrame(frame: Float32Array) {
  let sum = 0;
  for (let i = 0; i < frame.length; i += 1) {
    sum += Math.abs(frame[i]);
  }
  return Math.min(1, sum / frame.length / 0.18);
}

function samplesPerCodecFrame(sampleRate: number) {
  return Math.max(1, Math.round((sampleRate * CODEC_FRAME_MS) / 1_000));
}

function fadeFrame(frame: Float32Array, lossCount: number) {
  const gain = Math.max(0, 1 - lossCount * 0.28);

  for (let i = 0; i < frame.length; i += 1) {
    const tail = 1 - i / frame.length;
    frame[i] *= gain * Math.max(0.55, tail);
  }

  return frame;
}

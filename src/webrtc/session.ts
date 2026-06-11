import type { PeerSignal } from './signaling';
import { CIAO_ACTIVE_CODEC, CODEC_CODE_BYTES, CODEC_FRAME_MS, CODEC_PACKET_BYTES } from '../audio/codec';
import { CIAO_PROTOCOL, CiaoFrameType, decodeFrame, encodeFrame } from './protocol';
import {
  MAX_AUDIO_AGGREGATION,
  clampAggregation,
  decodeAudioPacket,
  decodeStreamFeedback,
  encodeAudioPacket,
  encodeStreamFeedback,
  type RemoteAudioFrame,
} from './streaming';

type SessionOptions = {
  onStateChange: (state: RTCPeerConnectionState) => void;
  onLocalSignal: (signal: PeerSignal) => void;
  onChannelOpen: () => void;
  onProtocolError: (message: string) => void;
  onLocalAudioDrop: (frames: number) => void;
  onRemoteAudioFrames: (frames: RemoteAudioFrame[]) => void;
  onRemoteDtx: (sequence: number) => void;
  getReceiveBufferDepth: () => number;
  rtcConfig?: RTCConfiguration;
};

const rtcConfig: RTCConfiguration = {
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
};
const HELLO_TIMEOUT_MS = 3_000;
const AUDIO_SEND_DROP_BUFFER_BYTES = 128_000;
const AUDIO_SEND_AGGREGATE_BUFFER_BYTES = 48_000;
const AUDIO_SEND_DEAGGREGATE_BUFFER_BYTES = 12_000;
const AUDIO_FEEDBACK_INTERVAL_MS = 1_000;

export class CiaoPeerSession {
  private readonly peer: RTCPeerConnection;
  private readonly options: SessionOptions;
  private channel: RTCDataChannel | null = null;
  private pendingIceCandidates: RTCIceCandidateInit[] = [];
  private controlSequence = 0;
  private audioFrameSequence = 0;
  private audioQueue: Uint8Array[] = [];
  private audioFlushTimer: number | undefined;
  private audioAggregation = 1;
  private highestReceivedAudioSequence: number | null = null;
  private lastReceivedAudioSequence: number | null = null;
  private lastAudioArrivalAt = 0;
  private receivedFramesWindow = 0;
  private lostFramesWindow = 0;
  private jitterMs = 0;
  private lastFeedbackAt = 0;
  private helloTimer: number | undefined;
  private channelReady = false;

  constructor(options: SessionOptions) {
    this.options = options;
    this.peer = new RTCPeerConnection(options.rtcConfig ?? rtcConfig);
    this.peer.onconnectionstatechange = () => {
      this.options.onStateChange(this.peer.connectionState);
    };
    this.peer.ondatachannel = (event) => {
      this.attachChannel(event.channel);
    };
    this.peer.onicecandidate = (event) => {
      if (event.candidate) {
        this.options.onLocalSignal({
          type: 'ice',
          candidate: event.candidate.toJSON(),
        });
      }
    };
  }

  async createOfferSignal() {
    this.ensureChannel();
    const offer = await this.peer.createOffer();
    await this.peer.setLocalDescription(offer);
    return {
      type: 'offer',
      description: this.peer.localDescription?.toJSON() ?? offer,
    } satisfies PeerSignal;
  }

  async acceptOfferSignal(description: RTCSessionDescriptionInit) {
    await this.peer.setRemoteDescription(description);
    await this.flushIceCandidates();
    const answer = await this.peer.createAnswer();
    await this.peer.setLocalDescription(answer);
    return {
      type: 'answer',
      description: this.peer.localDescription?.toJSON() ?? answer,
    } satisfies PeerSignal;
  }

  async acceptAnswerSignal(description: RTCSessionDescriptionInit) {
    await this.peer.setRemoteDescription(description);
    await this.flushIceCandidates();
  }

  async addIceCandidate(candidate: RTCIceCandidateInit) {
    if (!this.peer.remoteDescription) {
      this.pendingIceCandidates.push(candidate);
      return;
    }

    await this.peer.addIceCandidate(candidate);
  }

  sendAudioFrame(payload: Uint8Array) {
    if (!this.channel || this.channel.readyState !== 'open') {
      return;
    }

    this.audioQueue.push(payload);

    if (this.audioQueue.length >= this.audioAggregation) {
      this.flushAudioQueue();
      return;
    }

    this.scheduleAudioFlush();
  }

  sendDtx() {
    this.flushAudioQueue();
    this.sendAudioPacket([], { dtx: true });
  }

  close() {
    this.clearAudioFlushTimer();
    this.clearHelloTimer();
    this.channel?.close();
    this.peer.close();
  }

  private ensureChannel() {
    if (this.channel) {
      return;
    }

    this.attachChannel(
      this.peer.createDataChannel('voice', {
        ordered: false,
        maxRetransmits: 0,
        protocol: CIAO_PROTOCOL,
      }),
    );
  }

  private async flushIceCandidates() {
    const candidates = this.pendingIceCandidates.splice(0);

    for (const candidate of candidates) {
      await this.peer.addIceCandidate(candidate);
    }
  }

  private attachChannel(channel: RTCDataChannel) {
    if (channel.protocol && channel.protocol !== CIAO_PROTOCOL) {
      channel.close();
      this.options.onProtocolError('protocollo voce non compatibile');
      return;
    }

    this.channel = channel;
    this.channel.binaryType = 'arraybuffer';
    this.channel.onopen = () => {
      this.sendHello();
      this.scheduleHelloTimeout();
    };
    this.channel.onclose = () => {
      this.channelReady = false;
      this.clearHelloTimer();
    };
    this.channel.onmessage = (event) => {
      if (!(event.data instanceof ArrayBuffer)) {
        return;
      }

      const packet = decodeAudioPacket(new Uint8Array(event.data));
      if (packet) {
        if (this.channelReady) {
          this.handleAudioPacket(packet);
        }
        return;
      }

      const frame = decodeFrame(event.data);
      if (!frame) {
        return;
      }

      if (frame.type === CiaoFrameType.State) {
        if (!this.channelReady) {
          return;
        }
        this.handleStreamFeedback(frame.payload);
      } else if (frame.type === CiaoFrameType.Hello) {
        this.handleHello(frame.payload);
      } else if (frame.type === CiaoFrameType.Bye) {
        this.close();
      }
    };
  }

  private flushAudioQueue() {
    if (this.audioQueue.length === 0) {
      return;
    }

    this.clearAudioFlushTimer();
    const frames = this.audioQueue.splice(0, this.audioAggregation);
    this.sendAudioPacket(frames);

    if (this.audioQueue.length > 0) {
      this.scheduleAudioFlush();
    }
  }

  private scheduleAudioFlush() {
    if (this.audioFlushTimer !== undefined) {
      return;
    }

    this.audioFlushTimer = window.setTimeout(() => {
      this.audioFlushTimer = undefined;
      this.flushAudioQueue();
    }, CODEC_FRAME_MS);
  }

  private clearAudioFlushTimer() {
    if (this.audioFlushTimer === undefined) {
      return;
    }

    window.clearTimeout(this.audioFlushTimer);
    this.audioFlushTimer = undefined;
  }

  private sendAudioPacket(frames: Uint8Array[], options: { dtx?: boolean } = {}) {
    if (!this.channel || this.channel.readyState !== 'open') {
      return;
    }

    const baseSequence = this.audioFrameSequence;
    const packet = encodeAudioPacket(baseSequence, frames, options);

    if (frames.length > 0) {
      this.audioFrameSequence = (this.audioFrameSequence + frames.length) >>> 0;
    }

    if (this.channel.bufferedAmount < AUDIO_SEND_DROP_BUFFER_BYTES) {
      this.channel.send(packet);
    } else if (frames.length > 0) {
      this.options.onLocalAudioDrop(frames.length);
    }
  }

  private handleAudioPacket(packet: NonNullable<ReturnType<typeof decodeAudioPacket>>) {
    if (packet.dtx) {
      this.options.onRemoteDtx(packet.sequence);
    }

    if (packet.frames.length === 0) {
      return;
    }

    const now = performance.now();
    const frames = packet.frames.map((frame, index) => ({
      sequence: (packet.sequence + index) >>> 0,
      sentAt: 0,
      receivedAt: now,
      payload: frame,
    }));

    this.updateReceiveStats(frames);
    this.options.onRemoteAudioFrames(frames);
    this.maybeSendStreamFeedback(now);
  }

  private updateReceiveStats(frames: RemoteAudioFrame[]) {
    const firstFrame = frames[0];
    const lastFrame = frames[frames.length - 1];

    if (!firstFrame || !lastFrame) {
      return;
    }

    if (
      this.lastReceivedAudioSequence !== null &&
      firstFrame.sequence > this.lastReceivedAudioSequence
    ) {
      const expectedDelta = Math.max(1, firstFrame.sequence - this.lastReceivedAudioSequence) * CODEC_FRAME_MS;
      const arrivalDelta = Math.max(0, firstFrame.receivedAt - this.lastAudioArrivalAt);
      const variation = Math.abs(arrivalDelta - expectedDelta);
      this.jitterMs += (variation - this.jitterMs) / 16;
    }

    for (const frame of frames) {
      if (
        this.highestReceivedAudioSequence !== null &&
        frame.sequence > this.highestReceivedAudioSequence + 1
      ) {
        this.lostFramesWindow += frame.sequence - this.highestReceivedAudioSequence - 1;
      }

      if (this.highestReceivedAudioSequence === null || frame.sequence > this.highestReceivedAudioSequence) {
        this.highestReceivedAudioSequence = frame.sequence;
      }

      this.receivedFramesWindow += 1;
    }

    if (this.lastReceivedAudioSequence === null || lastFrame.sequence > this.lastReceivedAudioSequence) {
      this.lastReceivedAudioSequence = lastFrame.sequence;
      this.lastAudioArrivalAt = firstFrame.receivedAt;
    }
  }

  private maybeSendStreamFeedback(now: number) {
    if (now - this.lastFeedbackAt < AUDIO_FEEDBACK_INTERVAL_MS || this.highestReceivedAudioSequence === null) {
      return;
    }

    this.lastFeedbackAt = now;
    this.sendStateFeedback();
  }

  private sendStateFeedback() {
    if (!this.channel || this.channel.readyState !== 'open' || this.highestReceivedAudioSequence === null) {
      return;
    }

    const payload = encodeStreamFeedback({
      highestSequence: this.highestReceivedAudioSequence,
      receivedFrames: this.receivedFramesWindow,
      lostFrames: this.lostFramesWindow,
      jitterMs: this.jitterMs,
      bufferFrames: this.options.getReceiveBufferDepth(),
      targetAggregation: this.recommendedRemoteAggregation(),
    });

    this.receivedFramesWindow = 0;
    this.lostFramesWindow = 0;

    this.channel.send(
      encodeFrame({
        type: CiaoFrameType.State,
        sequence: this.controlSequence,
        timestamp: audioTimestamp(),
        payload,
      }),
    );
    this.controlSequence = (this.controlSequence + 1) >>> 0;
  }

  private handleStreamFeedback(payload: Uint8Array) {
    const feedback = decodeStreamFeedback(payload);

    if (!feedback) {
      return;
    }

    const totalFrames = feedback.receivedFrames + feedback.lostFrames;
    const lossRate = totalFrames > 0 ? feedback.lostFrames / totalFrames : 0;
    const buffered = this.channel?.bufferedAmount ?? 0;
    const requestedAggregation = clampAggregation(feedback.targetAggregation);

    if (lossRate > 0.15) {
      this.audioAggregation = 1;
      return;
    }

    const shouldGrow =
      buffered > AUDIO_SEND_AGGREGATE_BUFFER_BYTES ||
      (requestedAggregation > this.audioAggregation && lossRate < 0.05);
    const shouldShrink =
      buffered < AUDIO_SEND_DEAGGREGATE_BUFFER_BYTES &&
      requestedAggregation < this.audioAggregation &&
      lossRate === 0 &&
      feedback.jitterMs < CODEC_FRAME_MS * 0.45;

    if (shouldGrow) {
      this.audioAggregation += 1;
    } else if (shouldShrink) {
      this.audioAggregation -= 1;
    }

    this.audioAggregation = clampAggregation(this.audioAggregation);
  }

  private recommendedRemoteAggregation() {
    const totalFrames = this.receivedFramesWindow + this.lostFramesWindow;
    const lossRate = totalFrames > 0 ? this.lostFramesWindow / totalFrames : 0;

    if (lossRate > 0.08) {
      return 1;
    }

    if (this.jitterMs > CODEC_FRAME_MS * 1.25 || this.options.getReceiveBufferDepth() >= 3) {
      return 3;
    }

    if (this.jitterMs > CODEC_FRAME_MS * 0.75 || this.options.getReceiveBufferDepth() >= 2) {
      return 2;
    }

    return 1;
  }

  private sendHello() {
    if (!this.channel || this.channel.readyState !== 'open') {
      return;
    }

    const payload = new TextEncoder().encode(
      JSON.stringify({
        app: 'ciao',
        protocol: CIAO_PROTOCOL,
        codec: CIAO_ACTIVE_CODEC.id,
        bitrate: CIAO_ACTIVE_CODEC.bitrate,
        codecConfig: {
          sampleRate: CIAO_ACTIVE_CODEC.sampleRate,
          tokenStepMs: CIAO_ACTIVE_CODEC.tokenStepMs,
          tokenStepSamples: CIAO_ACTIVE_CODEC.tokenStepSamples,
          tokenSteps: CIAO_ACTIVE_CODEC.tokenSteps,
          inputSamples: CIAO_ACTIVE_CODEC.inputSamples,
          playoutSamples: CIAO_ACTIVE_CODEC.playoutSamples,
          rawDecoderSamples: CIAO_ACTIVE_CODEC.rawDecoderSamples,
          modelCodebooks: CIAO_ACTIVE_CODEC.modelCodebooks,
          codebooks: CIAO_ACTIVE_CODEC.codebooks,
          bitsPerCode: CIAO_ACTIVE_CODEC.bitsPerCode,
          codeBytes: CODEC_CODE_BYTES,
          packetBytes: CODEC_PACKET_BYTES,
        },
        aggregation: [1, MAX_AUDIO_AGGREGATION],
        dtx: true,
        jitterBufferMs: CODEC_FRAME_MS,
      }),
    );

    this.channel.send(
      encodeFrame({
        type: CiaoFrameType.Hello,
        sequence: this.controlSequence,
        timestamp: audioTimestamp(),
        payload,
      }),
    );
    this.controlSequence = (this.controlSequence + 1) >>> 0;
  }

  private handleHello(payload: Uint8Array) {
    let hello: unknown;

    try {
      hello = JSON.parse(new TextDecoder().decode(payload));
    } catch {
      this.rejectChannel('hello voce non valido');
      return;
    }

    if (!isCompatibleHello(hello)) {
      this.rejectChannel('protocollo voce non compatibile');
      return;
    }

    this.clearHelloTimer();
    if (!this.channelReady) {
      this.channelReady = true;
      this.options.onChannelOpen();
    }
  }

  private scheduleHelloTimeout() {
    this.clearHelloTimer();
    this.helloTimer = window.setTimeout(() => {
      if (!this.channelReady) {
        this.rejectChannel('hello voce non ricevuto');
      }
    }, HELLO_TIMEOUT_MS);
  }

  private clearHelloTimer() {
    if (this.helloTimer === undefined) {
      return;
    }

    window.clearTimeout(this.helloTimer);
    this.helloTimer = undefined;
  }

  private rejectChannel(message: string) {
    this.clearHelloTimer();
    this.channelReady = false;
    this.options.onProtocolError(message);
    this.channel?.close();
    this.peer.close();
  }
}

function audioTimestamp() {
  return Math.floor(performance.now()) >>> 0;
}

function isCompatibleHello(value: unknown) {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const hello = value as {
    app?: unknown;
    protocol?: unknown;
    codec?: unknown;
    bitrate?: unknown;
    codecConfig?: Partial<Record<
      | 'sampleRate'
      | 'tokenStepMs'
      | 'tokenStepSamples'
      | 'tokenSteps'
      | 'inputSamples'
      | 'playoutSamples'
      | 'rawDecoderSamples'
      | 'modelCodebooks'
      | 'codebooks'
      | 'bitsPerCode'
      | 'codeBytes'
      | 'packetBytes',
      unknown
    >>;
  };
  return (
    hello.app === 'ciao' &&
    hello.protocol === CIAO_PROTOCOL &&
    hello.codec === CIAO_ACTIVE_CODEC.id &&
    hello.bitrate === CIAO_ACTIVE_CODEC.bitrate &&
    hello.codecConfig?.sampleRate === CIAO_ACTIVE_CODEC.sampleRate &&
    hello.codecConfig?.tokenStepMs === CIAO_ACTIVE_CODEC.tokenStepMs &&
    hello.codecConfig?.tokenStepSamples === CIAO_ACTIVE_CODEC.tokenStepSamples &&
    hello.codecConfig?.tokenSteps === CIAO_ACTIVE_CODEC.tokenSteps &&
    hello.codecConfig?.inputSamples === CIAO_ACTIVE_CODEC.inputSamples &&
    hello.codecConfig?.playoutSamples === CIAO_ACTIVE_CODEC.playoutSamples &&
    hello.codecConfig?.rawDecoderSamples === CIAO_ACTIVE_CODEC.rawDecoderSamples &&
    hello.codecConfig?.modelCodebooks === CIAO_ACTIVE_CODEC.modelCodebooks &&
    hello.codecConfig?.codebooks === CIAO_ACTIVE_CODEC.codebooks &&
    hello.codecConfig?.bitsPerCode === CIAO_ACTIVE_CODEC.bitsPerCode &&
    hello.codecConfig?.codeBytes === CODEC_CODE_BYTES &&
    hello.codecConfig?.packetBytes === CODEC_PACKET_BYTES
  );
}

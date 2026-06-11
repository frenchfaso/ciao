import { decodeAudioPacket, encodeAudioPacket, type RemoteAudioFrame } from './streaming';

type ServerEchoOptions = {
  onClose: () => void;
  onAudioFrames: (frames: RemoteAudioFrame[]) => void;
};

export class CiaoServerEcho {
  private socket: WebSocket | null = null;
  private audioFrameSequence = 0;
  private closing = false;

  constructor(private readonly options: ServerEchoOptions) {}

  connect(timeoutMs = 5_000) {
    if (this.socket) {
      return Promise.resolve();
    }

    const socket = new WebSocket(echoSocketUrl());
    socket.binaryType = 'arraybuffer';
    this.socket = socket;

    return new Promise<void>((resolve, reject) => {
      let settled = false;
      const timeout = window.setTimeout(() => {
        if (settled) {
          return;
        }

        settled = true;
        this.closing = true;
        socket.close();
        reject(new Error('echo server non disponibile'));
      }, timeoutMs);

      socket.onopen = () => {
        if (this.socket !== socket || this.closing) {
          return;
        }

        settled = true;
        window.clearTimeout(timeout);
        resolve();
      };

      socket.onmessage = (event) => {
        void this.handleMessage(event.data);
      };

      socket.onerror = () => {
        if (!settled) {
          settled = true;
          window.clearTimeout(timeout);
          reject(new Error('echo server non disponibile'));
        }
      };

      socket.onclose = () => {
        if (this.socket === socket) {
          this.socket = null;
        }

        window.clearTimeout(timeout);
        if (!settled) {
          settled = true;
          reject(new Error('echo server chiuso'));
          return;
        }

        if (!this.closing) {
          this.options.onClose();
        }
      };
    });
  }

  sendAudioFrame(frame: Uint8Array) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN || this.socket.bufferedAmount > 128_000) {
      return;
    }

    const baseSequence = this.audioFrameSequence;
    const packet = encodeAudioPacket(baseSequence, [frame]);

    this.audioFrameSequence = (this.audioFrameSequence + 1) >>> 0;
    this.socket.send(packet);
  }

  close() {
    this.closing = true;
    this.socket?.close();
    this.socket = null;
  }

  private async handleMessage(data: unknown) {
    if (this.closing) {
      return;
    }

    const buffer = await messageBuffer(data);
    if (!buffer) {
      return;
    }

    const packet = decodeAudioPacket(new Uint8Array(buffer));
    if (!packet || packet.frames.length === 0) {
      return;
    }

    const now = performance.now();
    this.options.onAudioFrames(
      packet.frames.map((payload, index) => ({
        sequence: (packet.sequence + index) >>> 0,
        sentAt: 0,
        receivedAt: now,
        payload,
      })),
    );
  }
}

async function messageBuffer(data: unknown) {
  if (data instanceof ArrayBuffer) {
    return data;
  }

  if (data instanceof Blob) {
    return data.arrayBuffer();
  }

  return null;
}

function echoSocketUrl() {
  const url = new URL('/echo', location.href);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url;
}

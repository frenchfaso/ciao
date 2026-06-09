export type SignalRole = 'host' | 'guest';

export type PeerSignal =
  | { type: 'offer'; description: RTCSessionDescriptionInit }
  | { type: 'answer'; description: RTCSessionDescriptionInit }
  | { type: 'ice'; candidate: RTCIceCandidateInit };

type ServerMessage =
  | { type: 'joined'; role: SignalRole; room: string }
  | { type: 'peer-joined'; role: SignalRole }
  | { type: 'peer-left'; role: SignalRole }
  | { type: 'error'; message: string }
  | PeerSignal;

type SignalingOptions = {
  room: string;
  role: SignalRole;
  onOpen: () => void;
  onClose: () => void;
  onStatus: (status: string) => void;
  onPeerJoined: (role: SignalRole) => void;
  onPeerLeft: (role: SignalRole) => void;
  onSignal: (signal: PeerSignal) => void;
};

const ROOM_PATH_PATTERN = /^\/r\/([a-zA-Z0-9_-]{6,48})\/?$/;
const ROOM_ID_ALPHABET = 'abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789';

export class CiaoSignalingClient {
  private readonly options: SignalingOptions;
  private socket: WebSocket | null = null;

  constructor(options: SignalingOptions) {
    this.options = options;
  }

  connect() {
    if (this.socket) {
      return Promise.resolve();
    }

    return new Promise<void>((resolve, reject) => {
      const socket = new WebSocket(signalingUrl(this.options.room, this.options.role));
      this.socket = socket;

      socket.addEventListener(
        'open',
        () => {
          this.options.onOpen();
          resolve();
        },
        { once: true },
      );

      socket.addEventListener(
        'error',
        () => {
          reject(new Error('signaling non disponibile'));
        },
        { once: true },
      );

      socket.addEventListener('close', () => {
        this.options.onClose();
      });

      socket.addEventListener('message', (event) => {
        if (typeof event.data !== 'string') {
          return;
        }

        let message: ServerMessage;

        try {
          message = JSON.parse(event.data) as ServerMessage;
        } catch {
          this.options.onStatus('segnale non valido');
          return;
        }

        if (message.type === 'joined') {
          this.options.onStatus('stanza pronta');
        } else if (message.type === 'peer-joined') {
          this.options.onPeerJoined(message.role);
        } else if (message.type === 'peer-left') {
          this.options.onPeerLeft(message.role);
        } else if (message.type === 'error') {
          this.options.onStatus(message.message);
        } else {
          this.options.onSignal(message);
        }
      });
    });
  }

  send(signal: PeerSignal) {
    if (this.socket?.readyState !== WebSocket.OPEN) {
      return;
    }

    this.socket.send(JSON.stringify(signal));
  }

  close() {
    this.socket?.close(1000, 'reset');
    this.socket = null;
  }
}

export function createRoomId(bytes = 10) {
  const values = new Uint8Array(bytes);
  crypto.getRandomValues(values);

  return Array.from(values, (value) => ROOM_ID_ALPHABET[value % ROOM_ID_ALPHABET.length]).join('');
}

export function buildRoomUrl(room: string) {
  const url = new URL(location.href);
  url.pathname = `/r/${room}`;
  url.search = '';
  url.hash = '';
  return url.toString();
}

export function getRoomFromUrl(value: string) {
  const url = new URL(value, location.href);
  return url.pathname.match(ROOM_PATH_PATTERN)?.[1] ?? null;
}

function signalingUrl(room: string, role: SignalRole) {
  const url = new URL('/ws', location.href);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.searchParams.set('room', room);
  url.searchParams.set('role', role);
  return url.toString();
}

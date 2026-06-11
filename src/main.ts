import { Link, Mic, type IconNode } from 'lucide';
import { registerSW } from 'virtual:pwa-register';
import './styles.css';
import { CIAO_ACTIVE_CODEC, CODEC_FRAME_MS, type CodecModelCacheProgress } from './audio/codec';
import { CiaoAudioEngine, requestMicrophoneStream, type AudioPerformanceSample } from './audio/audio-engine';
import {
  createCodec,
  isCodecPrewarmed,
  prepareCodecRuntime,
  prewarmCodec,
  warmMimiModelCache as warmCodecModelCache,
} from './audio/mimi-codec';
import { CiaoServerEcho } from './webrtc/server-echo';
import { CiaoPeerSession } from './webrtc/session';
import {
  buildRoomUrl,
  CiaoSignalingClient,
  createRoomId,
  getRoomFromUrl,
  type PeerSignal,
  type SignalRole,
} from './webrtc/signaling';

type Role = 'idle' | 'echo' | SignalRole;
type ModelStatus = 'checking' | 'downloading' | 'initializing' | 'warming' | 'ready' | 'unsupported' | 'error';
type NavigatorWithStandalone = Navigator & {
  standalone?: boolean;
};

type AppState = {
  role: Role;
  status: string;
  modelStatus: ModelStatus;
  modelProgress: number;
  modelError: string;
  codecPrewarmed: boolean;
  pwaInstalled: boolean;
  connection: RTCPeerConnectionState | 'idle';
  channelOpen: boolean;
  micEnabled: boolean;
  roomId: string;
  inviteUrl: string;
  hasRoomLink: boolean;
  peerPresent: boolean;
  performanceSlow: boolean;
  encodeMs: number | null;
  decodeMs: number | null;
  fullDuplexMs: number | null;
  captureBacklogMs: number;
  droppedCaptureFrames: number;
  droppedNetworkFrames: number;
  remoteFramesReceived: number;
  decodedRemoteFrames: number;
};

const icons = { Link, Mic };
const initialRoom = getRoomFromUrl(location.href);
const HOST_ROOM_STORAGE_KEY = 'ciao.hostRoom';
const LAST_ROOM_STORAGE_KEY = 'ciao.lastRoom';
const LAST_ROLE_STORAGE_KEY = 'ciao.lastRole';
const PERFORMANCE_SLOW_CONFIRM_MS = 3_500;
const PERFORMANCE_RECOVERY_CONFIRM_MS = 1_200;
const PERFORMANCE_DROP_HOLD_MS = 1_000;
const PERFORMANCE_DUPLEX_BUDGET_MS = CODEC_FRAME_MS * 3;
const PERFORMANCE_CODEC_BUDGET_MS = CODEC_FRAME_MS * 2.5;
const PERFORMANCE_CAPTURE_BACKLOG_BUDGET_MS = CODEC_FRAME_MS * 5;

const state: AppState = {
  role: 'idle',
  status: initialRoom ? 'link ricevuto' : '',
  modelStatus: 'checking',
  modelProgress: 0,
  modelError: '',
  codecPrewarmed: false,
  pwaInstalled: isPwaInstalled(),
  connection: 'idle',
  channelOpen: false,
  micEnabled: false,
  roomId: initialRoom ?? '',
  inviteUrl: initialRoom ? buildRoomUrl(initialRoom) : '',
  hasRoomLink: Boolean(initialRoom),
  peerPresent: false,
  performanceSlow: false,
  encodeMs: null,
  decodeMs: null,
  fullDuplexMs: null,
  captureBacklogMs: 0,
  droppedCaptureFrames: 0,
  droppedNetworkFrames: 0,
  remoteFramesReceived: 0,
  decodedRemoteFrames: 0,
};

let session: CiaoPeerSession | null = null;
let serverEcho: CiaoServerEcho | null = null;
let signaling: CiaoSignalingClient | null = null;
let audio: CiaoAudioEngine | null = null;
let refreshApp: ((reloadPage?: boolean) => Promise<void>) | undefined;
let offerSent = false;
let signalingVersion = 0;
let autoJoinStarted = false;
let pendingAppRefresh = false;
let lastPerformanceRenderAt = 0;
let lastModelRenderAt = 0;
let lastRemoteStatsRenderAt = 0;
let lastCaptureDropAt = 0;
let performancePressureStartedAt = 0;
let performanceRecoveryStartedAt = 0;
let codecPromise: Promise<void> | null = null;
let codecWarmPromise: Promise<void> | null = null;
let codecReady = false;

const app = document.querySelector<HTMLDivElement>('#app');

if (!app) {
  throw new Error('Missing #app root');
}

const appRoot = app;

refreshApp = registerSW({
  immediate: true,
  onNeedRefresh() {
    pendingAppRefresh = true;
    applyPendingAppRefresh();
  },
  onOfflineReady() {
    state.status = 'app disponibile offline';
    render();
  },
});

window.addEventListener('popstate', () => {
  syncRoomFromUrl();
  void autoJoinRoomFromUrl();
});

window.addEventListener('appinstalled', () => {
  state.pwaInstalled = true;
  render();
});

const standaloneMedia = window.matchMedia('(display-mode: standalone)');
const handleStandaloneChange = () => {
  state.pwaInstalled = isPwaInstalled();
  render();
};
if (standaloneMedia.addEventListener) {
  standaloneMedia.addEventListener('change', handleStandaloneChange);
} else {
  standaloneMedia.addListener?.(handleStandaloneChange);
}

function render() {
  const connected = isCallConnected();
  const modelReady = isCodecReady();
  const micDisabled = !connected && !modelReady;
  const modelPercent = Math.round(state.modelProgress * 100);
  const modelLabel = modelStatusText();
  const modelCaption = modelCaptionText();
  const micLabel = connected ? 'Chiudi connessione' : modelReady ? 'Microfono' : modelLabel;
  appRoot.dataset.ciaoRole = state.role;
  appRoot.dataset.ciaoStatus = state.status;
  appRoot.dataset.ciaoModelStatus = state.modelStatus;
  appRoot.dataset.ciaoModelProgress = String(modelPercent);
  appRoot.dataset.ciaoModelError = state.modelError;
  appRoot.dataset.ciaoCodecPrewarmed = String(state.codecPrewarmed);
  appRoot.dataset.ciaoPwaInstalled = String(state.pwaInstalled);
  appRoot.dataset.ciaoConnection = state.connection;
  appRoot.dataset.ciaoChannelOpen = String(state.channelOpen);
  appRoot.dataset.ciaoMicEnabled = String(state.micEnabled);
  appRoot.dataset.ciaoPerformanceSlow = String(state.performanceSlow);
  appRoot.dataset.ciaoEncodeMs = formatMetric(state.encodeMs);
  appRoot.dataset.ciaoDecodeMs = formatMetric(state.decodeMs);
  appRoot.dataset.ciaoFullDuplexMs = formatMetric(state.fullDuplexMs);
  appRoot.dataset.ciaoCaptureBacklogMs = formatMetric(state.captureBacklogMs);
  appRoot.dataset.ciaoDroppedCaptureFrames = String(state.droppedCaptureFrames);
  appRoot.dataset.ciaoDroppedNetworkFrames = String(state.droppedNetworkFrames);
  appRoot.dataset.ciaoRemoteFrames = String(state.remoteFramesReceived);
  appRoot.dataset.ciaoDecodedFrames = String(state.decodedRemoteFrames);

  appRoot.innerHTML = `
    <main class="shell">
      <section class="status-band">
        <button class="home-hand" data-action="home" title="Home" aria-label="Home">👋</button>
      </section>

      <section class="stage" aria-live="polite">
        <button
          class="orb ${connected ? 'is-connected is-live' : ''} ${state.performanceSlow ? 'is-slow' : ''} ${!modelReady ? 'is-waiting' : ''}"
          data-action="mic-press"
          title="${escapeHtml(micLabel)}"
          aria-label="${escapeHtml(micLabel)}"
          ${micDisabled ? 'disabled' : ''}
        >
          <span>${icon('Mic')}</span>
        </button>
        ${modelReady ? '' : `
          <div class="model-readiness">
            <div
              class="model-progress"
              role="progressbar"
              aria-label="${escapeHtml(modelCaption)}"
              aria-valuemin="0"
              aria-valuemax="100"
              aria-valuenow="${modelPercent}"
            >
              <span style="width: ${modelPercent}%"></span>
            </div>
            <div class="model-caption ${state.modelStatus === 'error' || state.modelStatus === 'unsupported' ? 'is-error' : ''}">
              ${escapeHtml(modelCaption)}
            </div>
          </div>
        `}
      </section>

      <section class="controls" aria-label="Controlli chiamata">
        <button class="primary icon-action" data-action="copy-link" title="Copia link" aria-label="Copia link">
          ${icon('Link')}
        </button>
      </section>

      <span class="build-version" aria-label="Versione">${escapeHtml(__CIAO_APP_VERSION__)}</span>
    </main>
  `;

  appRoot.querySelectorAll<HTMLButtonElement>('button[data-action]').forEach((button) => {
    button.addEventListener('click', () => dispatch(button.dataset.action ?? ''));
  });
}

function icon(name: keyof typeof icons) {
  return iconNodeToSvg(icons[name] as IconNode, {
    width: '22',
    height: '22',
    'stroke-width': '2.2',
    'aria-hidden': 'true',
  });
}

async function dispatch(action: string) {
  try {
    if (action === 'home') goHome();
    if (action === 'copy-link') await copyCurrentOrCreateLink();
    if (action === 'mic-press') await handleMicPress();
  } catch (error) {
    state.status = error instanceof Error ? error.message : 'errore';
    render();
  }
}

async function copyCurrentOrCreateLink() {
  if (state.inviteUrl) {
    await copyText(state.inviteUrl);
    render();
    return;
  }

  await startHost();
}

function goHome() {
  resetCall({ clearRememberedRoom: true });
}

async function startHost() {
  resetCall({ clearRoute: false });
  const room = createRoomId();
  const roomUrl = buildRoomUrl(room);
  rememberRoom(room, 'host');
  history.pushState({ room, role: 'host' }, '', roomUrl);

  state.role = 'host';
  state.roomId = room;
  state.inviteUrl = roomUrl;
  state.hasRoomLink = true;
  state.status = 'link';
  render();

  await copyText(roomUrl);
  await startSignaledSession('host', room);
}

async function joinCurrentRoom() {
  const room = getRoomFromUrl(location.href);

  if (!room) {
    throw new Error('link stanza mancante');
  }

  const role: SignalRole = isHostedRoom(room) ? 'host' : 'guest';

  resetCall({ clearRoute: false, preserveHostedRoom: role === 'host' ? room : undefined });
  if (role === 'host') {
    rememberHostedRoom(room);
  }
  rememberRoom(room, role);

  state.role = role;
  state.roomId = room;
  state.inviteUrl = buildRoomUrl(room);
  state.hasRoomLink = true;
  state.status = role === 'host' ? 'stanza pronta' : 'entro nella stanza';
  render();

  await startSignaledSession(role, room);
}

async function startSignaledSession(role: SignalRole, room: string) {
  rememberRoom(room, role);
  offerSent = false;
  const version = signalingVersion + 1;
  signalingVersion = version;
  session = buildSession();
  signaling = buildSignaling(role, room, version);

  state.status = 'signaling';
  render();
  await signaling.connect();

  state.status = role === 'host' ? 'condividi il link' : 'aspetto offer';
  render();
}

function buildSession(
  options: { onLocalSignal?: (signal: PeerSignal) => void; rtcConfig?: RTCConfiguration } = {},
) {
  const peer = new CiaoPeerSession({
    rtcConfig: options.rtcConfig,
    onLocalSignal(signal) {
      if (options.onLocalSignal) {
        options.onLocalSignal(signal);
        return;
      }

      signaling?.send(signal);
    },
    onStateChange(connection) {
      state.connection = connection;
      if (connection === 'failed' || connection === 'closed') {
        state.channelOpen = false;
      }
      state.status = isCallConnected() ? 'canale dati attivo' : connection;
      render();
    },
    onProtocolError(message) {
      state.channelOpen = false;
      state.status = message;
      render();
    },
    onChannelOpen: () => {
      state.channelOpen = true;
      state.status = 'canale voce';
      render();
      void ensureMicEnabled().catch((error: unknown) => {
        state.status = error instanceof Error ? error.message : 'microfono non disponibile';
        render();
      });
    },
    onLocalAudioDrop(frames) {
      state.droppedNetworkFrames += frames;
      render();
    },
    onRemoteAudioFrames: (frames) => {
      state.remoteFramesReceived += frames.length;
      renderRemoteStats();
      if (state.role === 'echo') {
        void audio?.receiveRemoteFrames(frames);
        return;
      }

      void audio?.receiveRemoteFrames(frames);
    },
    onRemoteDtx: (sequence) => {
      audio?.receiveRemoteDtx(sequence);
    },
    getReceiveBufferDepth: () => audio?.getReceiveBufferDepth() ?? 0,
  });

  return peer;
}

function buildSignaling(role: SignalRole, room: string, version: number) {
  const isCurrent = () => version === signalingVersion;

  return new CiaoSignalingClient({
    room,
    role,
    onOpen() {
      if (!isCurrent()) return;
      state.status = 'stanza pronta';
      render();
    },
    onClose() {
      if (!isCurrent()) return;
      state.peerPresent = false;
      state.status = isCallConnected() ? state.status : 'signaling chiuso';
      render();
    },
    onStatus(status) {
      if (!isCurrent()) return;
      state.status = status;
      render();
    },
    onPeerJoined(peerRole) {
      if (!isCurrent()) return;
      state.peerPresent = true;
      state.status = peerRole === 'guest' ? 'peer entrato' : 'host online';
      render();

      if (role === 'host') {
        if (offerSent && !isCallConnected()) {
          rebuildPeerSession();
        }

        void sendOffer().catch((error: unknown) => {
          if (!isCurrent()) return;
          state.status = error instanceof Error ? error.message : 'offer fallita';
          offerSent = false;
          render();
        });
      }
    },
    onPeerLeft() {
      if (!isCurrent()) return;
      rebuildPeerSession();
      state.peerPresent = false;
      state.status = 'peer uscito';
      render();
    },
    onSignal(signal) {
      if (!isCurrent()) return;
      void handleSignal(signal).catch((error: unknown) => {
        if (!isCurrent()) return;
        state.status = error instanceof Error ? error.message : 'segnale non valido';
        render();
      });
    },
  });
}

async function sendOffer() {
  if (!session || !signaling || offerSent) {
    return;
  }

  offerSent = true;
  state.status = 'preparo connessione';
  render();
  signaling.send(await session.createOfferSignal());
  state.status = 'offer inviata';
  render();
}

async function handleSignal(signal: PeerSignal) {
  if (!session) {
    return;
  }

  if (signal.type === 'offer') {
    state.status = 'offer ricevuta';
    render();
    signaling?.send(await session.acceptOfferSignal(signal.description));
    state.status = 'answer inviata';
  } else if (signal.type === 'answer') {
    await session.acceptAnswerSignal(signal.description);
    state.status = 'answer ricevuta';
  } else if (signal.type === 'ice') {
    await session.addIceCandidate(signal.candidate);
    state.status = isCallConnected() ? state.status : 'ice';
  }

  render();
}

function rebuildPeerSession() {
  audio?.dispose();
  session?.close();
  audio = null;
  session = buildSession();
  offerSent = false;
  state.connection = 'idle';
  state.channelOpen = false;
  state.micEnabled = false;
  resetPerformanceState();
  warmIdleCodec();
}

async function handleMicPress() {
  const room = getActiveRoom();

  if (state.role === 'echo') {
    resetCall({ clearRoute: false });
    return;
  }

  if (session) {
    const role = room ? getRememberedRole(room) : state.role;
    const signalRole = role === 'host' || role === 'guest' ? role : undefined;

    resetCall({
      clearRoute: false,
      preserveHostedRoom: signalRole === 'host' ? room ?? undefined : undefined,
      preserveRole: signalRole,
      preserveRoom: room ?? undefined,
    });

    state.status = room ? 'disconnesso' : '';
    render();
    return;
  }

  if (!isCodecReady()) {
    void ensureCodecReady().catch(() => undefined);
    return;
  }

  if (room) {
    await reconnectRoom(room);
    return;
  }

  await startEchoTest();
}

async function reconnectRoom(room: string) {
  const role = getRememberedRole(room);

  if (getRoomFromUrl(location.href) !== room) {
    history.pushState({ room, role }, '', buildRoomUrl(room));
  }

  if (role === 'host') {
    rememberHostedRoom(room);
  }
  rememberRoom(room, role);

  state.role = role;
  state.roomId = room;
  state.inviteUrl = buildRoomUrl(room);
  state.hasRoomLink = true;
  state.status = role === 'host' ? 'stanza pronta' : 'entro nella stanza';
  render();

  await startSignaledSession(role, room);
}

async function startEchoTest() {
  resetCall({ clearRoute: false });

  state.role = 'echo';
  state.channelOpen = false;
  state.status = 'echo server';
  render();

  const echo = new CiaoServerEcho({
    onClose() {
      if (serverEcho !== echo || state.role !== 'echo') {
        return;
      }

      audio?.dispose();
      audio = null;
      serverEcho = null;
      state.role = 'idle';
      state.channelOpen = false;
      state.micEnabled = false;
      state.status = 'echo chiuso';
      resetPerformanceState();
      render();
      warmIdleCodec();
    },
    onAudioFrames(frames) {
      if (serverEcho !== echo || state.role !== 'echo') {
        return;
      }

      state.remoteFramesReceived += frames.length;
      renderRemoteStats();
      void audio?.receiveRemoteFrames(frames);
    },
  });

  serverEcho = echo;

  try {
    await echo.connect();
    if (serverEcho !== echo || state.role !== 'echo') {
      echo.close();
      return;
    }

    state.channelOpen = true;
    state.status = 'echo server';
    render();
    await ensureMicEnabled();
  } catch (error) {
    console.error('ciao echo test failed', error);

    if (serverEcho === echo) {
      serverEcho = null;
    }

    echo.close();
    audio?.dispose();
    audio = null;
    state.role = 'idle';
    state.channelOpen = false;
    state.micEnabled = false;
    state.status = error instanceof Error ? error.message : 'microfono non disponibile';
    resetPerformanceState();
    render();
    warmIdleCodec();
  }
}

async function ensureMicEnabled() {
  if ((!session && !serverEcho) || state.micEnabled) {
    return;
  }

  let micStream: MediaStream | undefined;

  try {
    if (!audio) {
      const voiceProcessing = state.role !== 'echo';
      await ensureCodecReady();
      const codec = await createCodec();
      state.codecPrewarmed = false;
      micStream = await requestMicrophoneStream(voiceProcessing);
      let engine: CiaoAudioEngine;
      engine = new CiaoAudioEngine({
        codec,
        onEncodedFrame(frame) {
          if (state.role === 'echo') {
            serverEcho?.sendAudioFrame(frame);
            return;
          }

          session?.sendAudioFrame(frame);
        },
        onDtx() {
          if (state.role === 'echo') {
            return;
          }

          session?.sendDtx();
        },
        onError(error) {
          if (audio === engine) {
            handleAudioError(error);
          }
        },
        onPerformance(sample) {
          if (audio === engine) {
            handleAudioPerformance(sample);
          }
        },
        vadEnabled: state.role !== 'echo',
        voiceProcessing,
      });
      audio = engine;
    }

    await audio.startCapture(micStream);
    micStream = undefined;
  } finally {
    micStream?.getTracks().forEach((track) => track.stop());
  }

  state.micEnabled = true;
  state.status = 'microfono attivo';
  render();
}

function handleAudioPerformance(sample: AudioPerformanceSample) {
  const previousSlow = state.performanceSlow;
  const now = performance.now();

  if (sample.kind === 'encode') {
    state.encodeMs = smoothMetric(state.encodeMs, sample.durationMs);
    state.captureBacklogMs = sample.captureBacklogMs;
    state.droppedCaptureFrames = sample.droppedCaptureFrames;
  } else if (sample.kind === 'decode') {
    state.decodeMs = smoothMetric(state.decodeMs, sample.durationMs);
    state.decodedRemoteFrames += 1;
  } else {
    state.captureBacklogMs = sample.captureBacklogMs;
    state.droppedCaptureFrames = sample.droppedCaptureFrames;
    lastCaptureDropAt = now;
  }

  state.fullDuplexMs =
    state.encodeMs !== null && state.decodeMs !== null ? state.encodeMs + state.decodeMs : null;
  state.performanceSlow = updateClientPerformanceSlow(now);

  if (previousSlow !== state.performanceSlow || now - lastPerformanceRenderAt > 1_000) {
    lastPerformanceRenderAt = now;
    render();
  }
}

function renderRemoteStats() {
  const now = performance.now();

  if (now - lastRemoteStatsRenderAt > 1_000) {
    lastRemoteStatsRenderAt = now;
    render();
  }
}

function updateClientPerformanceSlow(now: number) {
  const underPressure = isClientPerformanceUnderPressure(now);

  if (underPressure) {
    performanceRecoveryStartedAt = 0;
    if (performancePressureStartedAt === 0) {
      performancePressureStartedAt = now;
    }

    return state.performanceSlow || now - performancePressureStartedAt >= PERFORMANCE_SLOW_CONFIRM_MS;
  }

  performancePressureStartedAt = 0;
  if (performanceRecoveryStartedAt === 0) {
    performanceRecoveryStartedAt = now;
  }

  return state.performanceSlow && now - performanceRecoveryStartedAt < PERFORMANCE_RECOVERY_CONFIRM_MS;
}

function isClientPerformanceUnderPressure(now: number) {
  const recentDrop = now - lastCaptureDropAt < PERFORMANCE_DROP_HOLD_MS;
  const overloadedDuplex =
    state.fullDuplexMs !== null && state.fullDuplexMs > PERFORMANCE_DUPLEX_BUDGET_MS;
  const overloadedEncode = state.encodeMs !== null && state.encodeMs > PERFORMANCE_CODEC_BUDGET_MS;
  const overloadedDecode = state.decodeMs !== null && state.decodeMs > PERFORMANCE_CODEC_BUDGET_MS;
  const overloadedCapture = state.captureBacklogMs > PERFORMANCE_CAPTURE_BACKLOG_BUDGET_MS;

  return recentDrop || overloadedDuplex || overloadedEncode || overloadedDecode || overloadedCapture;
}

function smoothMetric(current: number | null, value: number) {
  return current === null ? value : current * 0.82 + value * 0.18;
}

function resetPerformanceState() {
  state.performanceSlow = false;
  state.encodeMs = null;
  state.decodeMs = null;
  state.fullDuplexMs = null;
  state.captureBacklogMs = 0;
  state.droppedCaptureFrames = 0;
  state.droppedNetworkFrames = 0;
  state.remoteFramesReceived = 0;
  state.decodedRemoteFrames = 0;
  lastCaptureDropAt = 0;
  lastPerformanceRenderAt = 0;
  lastRemoteStatsRenderAt = 0;
  performancePressureStartedAt = 0;
  performanceRecoveryStartedAt = 0;
}

function handleAudioError(error: unknown) {
  console.error('ciao audio failed', error);

  const role = state.role;
  const room = state.roomId || getRoomFromUrl(location.href) || undefined;
  const preserveSignalRole = role === 'host' || role === 'guest' ? role : undefined;
  const message = error instanceof Error ? error.message : 'audio non disponibile';

  resetCall({
    clearRoute: false,
    preserveHostedRoom: preserveSignalRole === 'host' ? room : undefined,
    preserveRole: preserveSignalRole,
    preserveRoom: preserveSignalRole ? room : undefined,
  });

  state.status = message;
  render();
}

function resetCall(
  options: {
    clearRoute?: boolean;
    clearRememberedRoom?: boolean;
    preserveHostedRoom?: string;
    preserveRole?: SignalRole;
    preserveRoom?: string;
  } = {},
) {
  const hostedRoom = getHostedRoom();

  signalingVersion += 1;
  audio?.dispose();
  serverEcho?.close();
  signaling?.close();
  session?.close();
  audio = null;
  serverEcho = null;
  signaling = null;
  session = null;
  offerSent = false;
  state.role = options.preserveRole ?? 'idle';
  state.connection = 'idle';
  state.channelOpen = false;
  state.micEnabled = false;
  resetPerformanceState();
  state.roomId = options.preserveRoom ?? '';
  state.inviteUrl = options.preserveRoom ? buildRoomUrl(options.preserveRoom) : '';
  state.hasRoomLink = Boolean(options.preserveRoom);
  state.peerPresent = false;
  state.status = '';

  if (hostedRoom && hostedRoom !== options.preserveHostedRoom) {
    forgetHostedRoom(hostedRoom);
  }

  if (options.clearRememberedRoom) {
    forgetRememberedRoom();
  }

  if (options.clearRoute !== false && !options.preserveRoom && getRoomFromUrl(location.href)) {
    history.pushState({}, '', '/');
  }

  if (!options.preserveRoom) {
    syncRoomFromUrl(false);
  }
  render();
  applyPendingAppRefresh();
  warmIdleCodec();
}

function applyPendingAppRefresh() {
  if (!pendingAppRefresh || !refreshApp || hasActiveRuntime()) {
    return;
  }

  pendingAppRefresh = false;
  void refreshApp(true);
}

function hasActiveRuntime() {
  return state.role !== 'idle' || Boolean(session || serverEcho || signaling || audio);
}

async function ensureCodecReady() {
  if (codecReady && state.codecPrewarmed && isCodecPrewarmed()) {
    return;
  }

  if (!codecReady && !codecPromise) {
    codecPromise = prepareCodec();
  }

  if (!codecReady) {
    return codecPromise;
  }

  return warmIdleCodec(true);
}

async function prepareCodec() {
  try {
    state.codecPrewarmed = false;
    updateModelState('checking', state.modelProgress, '', true);
    await warmCodecModelCache((progress) => {
      updateModelState(modelStatusFromCacheProgress(progress), progress.percent);
    });
    updateModelState('initializing', 1, '', true);
    await prepareCodecRuntime();
    codecReady = true;
    updateModelState('warming', 1, '', true);
    await warmIdleCodec(true);
    updateModelState('ready', 1, '', true);
  } catch (error) {
    codecReady = false;
    codecPromise = null;
    codecWarmPromise = null;
    state.codecPrewarmed = false;
    const message = error instanceof Error ? error.message : `${CIAO_ACTIVE_CODEC.label} non disponibile`;
    const lowerMessage = message.toLowerCase();
    updateModelState(
      lowerMessage.includes('webgpu') || lowerMessage.includes('navigator.gpu') ? 'unsupported' : 'error',
      state.modelProgress,
      message,
      true,
    );
    throw error;
  }
}

async function warmIdleCodec(wait = false) {
  if (!codecReady || audio) {
    return;
  }

  if (state.codecPrewarmed && isCodecPrewarmed()) {
    return;
  }

  if (!codecWarmPromise) {
    state.codecPrewarmed = false;
    if (state.modelStatus === 'ready') {
      updateModelState('warming', 1, '', true);
    } else {
      render();
    }

    codecWarmPromise = prewarmCodec()
      .then(() => {
        state.codecPrewarmed = isCodecPrewarmed();
        if (state.codecPrewarmed) {
          updateModelState('ready', 1, '', true);
        } else {
          render();
        }
      })
      .catch((error: unknown) => {
        console.warn('ciao codec warmup failed', error);
        throw error;
      })
      .finally(() => {
        codecWarmPromise = null;
      });
  }

  if (wait) {
    await codecWarmPromise;
  }
}

function modelStatusFromCacheProgress(progress: CodecModelCacheProgress): ModelStatus {
  if (progress.status === 'downloading') {
    return 'downloading';
  }

  if (progress.status === 'ready') {
    return 'initializing';
  }

  return 'checking';
}

function updateModelState(status: ModelStatus, progress: number, error = '', forceRender = false) {
  const now = performance.now();
  const changed =
    state.modelStatus !== status ||
    Math.round(state.modelProgress * 100) !== Math.round(progress * 100) ||
    state.modelError !== error;

  state.modelStatus = status;
  state.modelProgress = Math.max(0, Math.min(1, progress));
  state.modelError = error;

  if (forceRender || (changed && now - lastModelRenderAt > 160)) {
    lastModelRenderAt = now;
    render();
  }
}

function isCodecReady() {
  return state.modelStatus === 'ready' && codecReady && state.codecPrewarmed && isCodecPrewarmed();
}

function modelStatusText() {
  if (
    state.modelStatus === 'downloading' ||
    (state.modelStatus === 'checking' && state.modelProgress > 0 && state.modelProgress < 1)
  ) {
    return `${CIAO_ACTIVE_CODEC.label} ${Math.round(state.modelProgress * 100)}%`;
  }

  if (state.modelStatus === 'unsupported') {
    return 'WebGPU non disponibile';
  }

  if (state.modelStatus === 'error') {
    return `${CIAO_ACTIVE_CODEC.label} non disponibile`;
  }

  if (state.modelStatus === 'warming') {
    return `${CIAO_ACTIVE_CODEC.label} warmup`;
  }

  return CIAO_ACTIVE_CODEC.label;
}

function modelCaptionText() {
  if ((state.modelStatus === 'error' || state.modelStatus === 'unsupported') && state.modelError) {
    return compactModelError(state.modelError);
  }

  return modelStatusText();
}

function compactModelError(message: string) {
  return message
    .replace(new RegExp(`^${CIAO_ACTIVE_CODEC.label}\\s+`, 'i'), '')
    .replace(/^richiede\s+/i, '')
    .slice(0, 96);
}

function isPwaInstalled() {
  return (
    window.matchMedia('(display-mode: standalone)').matches ||
    Boolean((navigator as NavigatorWithStandalone).standalone)
  );
}

async function copyText(value: string) {
  if (await writeClipboard(value)) {
    state.status = 'link copiato';
    return;
  }

  state.status = 'link nella barra indirizzi';
}

async function writeClipboard(value: string) {
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch {
    return copyTextWithSelection(value);
  }
}

function copyTextWithSelection(value: string) {
  const input = document.createElement('textarea');
  input.value = value;
  input.setAttribute('readonly', '');
  input.style.position = 'fixed';
  input.style.left = '-9999px';
  input.style.top = '0';
  document.body.append(input);
  input.focus();
  input.select();

  try {
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    input.remove();
  }
}

async function autoJoinRoomFromUrl() {
  if (!getRoomFromUrl(location.href) || session || autoJoinStarted) {
    return;
  }

  autoJoinStarted = true;

  try {
    await joinCurrentRoom();
  } finally {
    autoJoinStarted = false;
  }
}

function syncRoomFromUrl(shouldRender = true) {
  const room = getRoomFromUrl(location.href);

  if (!session) {
    state.roomId = room ?? '';
    state.inviteUrl = room ? buildRoomUrl(room) : '';
    state.hasRoomLink = Boolean(room);
    state.status = room ? 'link ricevuto' : '';
  }

  if (shouldRender) {
    render();
  }
}

function isCallConnected() {
  if (state.role === 'echo') {
    return state.channelOpen;
  }

  return state.channelOpen || state.connection === 'connected';
}

function getActiveRoom() {
  return state.roomId || getRoomFromUrl(location.href);
}

function getHostedRoom() {
  try {
    return sessionStorage.getItem(HOST_ROOM_STORAGE_KEY);
  } catch {
    return null;
  }
}

function isHostedRoom(room: string) {
  const historyState = history.state as { room?: unknown; role?: unknown } | null;
  const tabCreatedRoom = historyState?.room === room && (historyState.role === 'host' || !historyState.role);

  return getHostedRoom() === room || tabCreatedRoom;
}

function rememberHostedRoom(room: string) {
  try {
    sessionStorage.setItem(HOST_ROOM_STORAGE_KEY, room);
  } catch {
    return;
  }
}

function getRememberedRole(room: string): SignalRole {
  if (isHostedRoom(room)) {
    return 'host';
  }

  if (state.roomId === room && (state.role === 'host' || state.role === 'guest')) {
    return state.role;
  }

  try {
    const rememberedRoom = sessionStorage.getItem(LAST_ROOM_STORAGE_KEY);
    const rememberedRole = sessionStorage.getItem(LAST_ROLE_STORAGE_KEY);

    if (rememberedRoom === room && (rememberedRole === 'host' || rememberedRole === 'guest')) {
      return rememberedRole;
    }
  } catch {
    return 'guest';
  }

  return 'guest';
}

function rememberRoom(room: string, role: SignalRole) {
  try {
    sessionStorage.setItem(LAST_ROOM_STORAGE_KEY, room);
    sessionStorage.setItem(LAST_ROLE_STORAGE_KEY, role);
  } catch {
    return;
  }

  if (role === 'host') {
    rememberHostedRoom(room);
  }
}

function forgetRememberedRoom() {
  try {
    sessionStorage.removeItem(LAST_ROOM_STORAGE_KEY);
    sessionStorage.removeItem(LAST_ROLE_STORAGE_KEY);
  } catch {
    return;
  }
}

function forgetHostedRoom(room: string) {
  try {
    if (sessionStorage.getItem(HOST_ROOM_STORAGE_KEY) === room) {
      sessionStorage.removeItem(HOST_ROOM_STORAGE_KEY);
    }
  } catch {
    return;
  }
}

function iconNodeToSvg(node: IconNode, overrides: Record<string, string>) {
  const [tag, attrs, children] = node;
  return renderSvgNode(tag, { ...attrs, ...overrides }, children);
}

function renderSvgNode(
  tag: string,
  attrs: Record<string, string | number>,
  children: IconNode[] = [],
): string {
  const serializedAttrs = Object.entries(attrs)
    .map(([key, value]) => `${key}="${escapeHtml(String(value))}"`)
    .join(' ');
  const serializedChildren = children.map(([childTag, childAttrs, childChildren]) =>
    renderSvgNode(childTag, childAttrs, childChildren),
  );
  return `<${tag} ${serializedAttrs}>${serializedChildren.join('')}</${tag}>`;
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function formatMetric(value: number | null) {
  return value === null ? '' : value.toFixed(1);
}

render();
void ensureCodecReady().catch(() => undefined);
void autoJoinRoomFromUrl();

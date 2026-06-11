import type * as Ort from 'onnxruntime-web';
import type { CodecModelCacheProgress } from './codec';
import { importOrtWebGpuModule } from './ort-loader';

type NavigatorWithGpu = Navigator & {
  gpu?: {
    requestAdapter?: (options?: { powerPreference?: 'high-performance' | 'low-power' }) => Promise<unknown>;
  };
};

export type MimiOrtModule = typeof Ort;
export type MimiOrtTensor = Ort.Tensor;
export type MimiOrtFeeds = Record<string, Ort.Tensor>;
export type MimiOrtOutputs = Ort.InferenceSession.ReturnType;

export type MimiRuntime = {
  ort: MimiOrtModule;
  encoder: Ort.InferenceSession;
  decoder: Ort.InferenceSession;
  stateSpec: MimiStateSpec;
};

type MimiStateSpec = {
  encoder: MimiConvStateSpec[];
  decoder: MimiConvStateSpec[];
};

type MimiConvStateSpec = {
  name: string;
  channels: number;
  temporalSize: number;
};

type RuntimeCacheAsset = {
  key: string;
  label: string;
  path: string;
  version: string;
  size: number;
  cacheName: string;
};

type MimiAssetKey = 'encoderModel' | 'decoderModel' | 'stateSpec';

export type MimiProbeResult = {
  frameMs: number;
  measuredFrames: number;
  encodeMs: TimingSummary;
  decodeMs: TimingSummary;
  encodePlusDecodeMs: TimingSummary;
  realtime: {
    avgRtf: number;
    p95Rtf: number;
    maxRtf: number;
    passesP95: boolean;
  };
};

type TimingSummary = {
  avg: number;
  p50: number;
  p95: number;
  max: number;
};

const MIMI_MODEL_BASE = '/models/mimi/streaming-8cb-fp16';
const ORT_ASSET_BASE = '/ort/1.26.0';
const ORT_CACHE_NAME = 'ciao-ort-runtime-v1';
const ORT_ASSET_VERSION = 'ort-1.26.0';
const MIMI_ASSET_VERSION = 'mimi-streaming-8cb-fp16-v1';
const MIMI_MODEL_CACHE_NAME = 'ciao-mimi-streaming-8cb-fp16-v1';
const MIMI_FRAME_MS = 80;
const MIMI_FRAME_SAMPLES = 1_920;
const MIMI_CODEBOOKS = 8;
const MIMI_KV_LAYERS = 8;
const MIMI_KV_HEADS = 8;
const MIMI_KV_HEAD_DIM = 64;

const MIMI_RUNTIME_ASSETS = [
  {
    key: 'encoderModel',
    label: 'Mimi encoder ONNX',
    path: `${MIMI_MODEL_BASE}/encoder_model.onnx`,
    version: MIMI_ASSET_VERSION,
    size: 124_768_461,
    cacheName: MIMI_MODEL_CACHE_NAME,
  },
  {
    key: 'decoderModel',
    label: 'Mimi decoder ONNX',
    path: `${MIMI_MODEL_BASE}/decoder_model.onnx`,
    version: MIMI_ASSET_VERSION,
    size: 97_478_284,
    cacheName: MIMI_MODEL_CACHE_NAME,
  },
  {
    key: 'stateSpec',
    label: 'Mimi state spec',
    path: `${MIMI_MODEL_BASE}/state_spec.txt`,
    version: MIMI_ASSET_VERSION,
    size: 534,
    cacheName: MIMI_MODEL_CACHE_NAME,
  },
] as const satisfies readonly RuntimeCacheAsset[];

const ORT_WASM_ASSET = {
  key: 'ortWasm',
  label: 'ort-wasm',
  path: `${ORT_ASSET_BASE}/ort-wasm-simd-threaded.asyncify.wasm`,
  version: ORT_ASSET_VERSION,
  size: 23_678_474,
  cacheName: ORT_CACHE_NAME,
} as const satisfies RuntimeCacheAsset;

const REQUIRED_RUNTIME_ASSETS = [...MIMI_RUNTIME_ASSETS, ORT_WASM_ASSET] as const;

const runtimeAssetMemoryCache = new Map<string, Uint8Array>();
let ortPromise: Promise<MimiOrtModule> | null = null;
let runtimePromise: Promise<MimiRuntime> | null = null;

export async function loadMimiRuntime(): Promise<MimiRuntime> {
  if (!runtimePromise) {
    runtimePromise = createMimiRuntime().catch((error: unknown) => {
      runtimePromise = null;
      throw error;
    });
  }

  return runtimePromise;
}

export async function warmMimiModelCache(onProgress: (progress: CodecModelCacheProgress) => void) {
  await ensureWebGpuAvailable();

  const totalBytes = totalRequiredBytes();
  let loadedBytes = 0;
  onProgress(progress('checking', loadedBytes, totalBytes));

  const missingAssets: RuntimeCacheAsset[] = [];
  for (const asset of REQUIRED_RUNTIME_ASSETS) {
    if (await assetCached(asset)) {
      loadedBytes += asset.size;
      onProgress(progress('cached', loadedBytes, totalBytes));
    } else {
      missingAssets.push(asset);
    }
  }

  for (const asset of missingAssets) {
    const previousLoadedBytes = loadedBytes;
    await loadAsset(asset, (assetLoadedBytes) => {
      onProgress(progress('downloading', previousLoadedBytes + assetLoadedBytes, totalBytes));
    });
    loadedBytes = previousLoadedBytes + asset.size;
    onProgress(progress('cached', loadedBytes, totalBytes));
  }

  onProgress(progress('ready', totalBytes, totalBytes));
}

export async function probeMimiWebGpu(options: { warmupFrames?: number; measuredFrames?: number } = {}) {
  const warmupFrames = options.warmupFrames ?? 4;
  const measuredFrames = options.measuredFrames ?? 32;
  const runtime = await loadMimiRuntime();
  let encoderState = createInitialMimiState(runtime.ort, runtime.stateSpec.encoder);
  let decoderState = createInitialMimiState(runtime.ort, runtime.stateSpec.decoder);
  const encodeTimings: number[] = [];
  const decodeTimings: number[] = [];
  const totalTimings: number[] = [];

  try {
    for (let frame = 0; frame < warmupFrames + measuredFrames; frame += 1) {
      const shouldMeasure = frame >= warmupFrames;
      const waveform = syntheticWaveform(frame);
      const encodeStartedAt = performance.now();
      const encodeFeeds = {
        input_values: new runtime.ort.Tensor('float32', waveform, [1, 1, MIMI_FRAME_SAMPLES]),
        ...encoderState,
      };
      const encodeOutputs = await runtime.encoder.run(encodeFeeds);
      const encodeMs = performance.now() - encodeStartedAt;
      disposeTensor(encodeFeeds.input_values);
      const previousEncoderState = encoderState;
      encoderState = updateMimiState(encodeOutputs);
      disposeState(previousEncoderState);

      const codes = requiredTensor(encodeOutputs, 'audio_codes');
      assertCodes(codes);

      const decodeStartedAt = performance.now();
      const decodeFeeds = {
        audio_codes: codes,
        ...decoderState,
      };
      const decodeOutputs = await runtime.decoder.run(decodeFeeds);
      const decodeMs = performance.now() - decodeStartedAt;
      const previousDecoderState = decoderState;
      decoderState = updateMimiState(decodeOutputs);
      disposeState(previousDecoderState);
      disposeTensor(codes);
      disposeTensor(decodeOutputs.audio_values);

      if (shouldMeasure) {
        encodeTimings.push(encodeMs);
        decodeTimings.push(decodeMs);
        totalTimings.push(encodeMs + decodeMs);
      }
    }
  } finally {
    disposeState(encoderState);
    disposeState(decoderState);
  }

  const total = summarize(totalTimings);
  return {
    frameMs: MIMI_FRAME_MS,
    measuredFrames,
    encodeMs: summarize(encodeTimings),
    decodeMs: summarize(decodeTimings),
    encodePlusDecodeMs: total,
    realtime: {
      avgRtf: total.avg / MIMI_FRAME_MS,
      p95Rtf: total.p95 / MIMI_FRAME_MS,
      maxRtf: total.max / MIMI_FRAME_MS,
      passesP95: total.p95 <= MIMI_FRAME_MS,
    },
  } satisfies MimiProbeResult;
}

async function createMimiRuntime(): Promise<MimiRuntime> {
  await ensureWebGpuAvailable();
  const assets = await loadMimiAssets();
  const stateSpec = parseStateSpec(textFromBytes(assets.stateSpec));
  const ort = await loadOrt();
  const encoder = await createMimiSession(ort, assets.encoderModel, mimiPreferredOutputLocation('audio_codes'));
  const decoder = await createMimiSession(ort, assets.decoderModel, mimiPreferredOutputLocation('audio_values'));
  return { ort, encoder, decoder, stateSpec };
}

async function loadOrt() {
  if (!ortPromise) {
    ortPromise = importOrtWebGpuModule(`${ORT_ASSET_BASE}/ort.webgpu.bundle.min.mjs`)
      .then((ort) => {
        configureOrt(ort);
        return ort;
      })
      .catch((error: unknown) => {
        throw new Error(`Mimi ORT runtime: ${errorMessage(error)}`);
      });
  }

  const ort = await ortPromise;
  configureOrt(ort);
  return ort;
}

async function createMimiSession(
  ort: MimiOrtModule,
  model: Uint8Array,
  preferredOutputLocation: Ort.InferenceSession.SessionOptions['preferredOutputLocation'],
) {
  const options = {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
    logVerbosityLevel: 0,
    preferredOutputLocation,
  } satisfies Ort.InferenceSession.SessionOptions;

  try {
    return await ort.InferenceSession.create(model, options);
  } catch (error) {
    console.warn('Mimi WebGPU session without GPU state outputs', error);
    return await ort.InferenceSession.create(model, {
      ...options,
      preferredOutputLocation: undefined,
    });
  }
}

function mimiPreferredOutputLocation(primaryOutput: 'audio_codes' | 'audio_values') {
  const outputLocation: Record<string, 'cpu' | 'gpu-buffer'> = {
    [primaryOutput]: 'cpu',
  };

  for (const name of convPresentStateNames()) {
    outputLocation[name] = 'gpu-buffer';
  }

  for (let index = 0; index < MIMI_KV_LAYERS; index += 1) {
    outputLocation[`present_key_${index}`] = 'gpu-buffer';
    outputLocation[`present_value_${index}`] = 'gpu-buffer';
  }

  return outputLocation;
}

function convPresentStateNames() {
  return [
    'present_conv_enc_0_state',
    'present_conv_enc_1_b1_state',
    'present_conv_enc_3_state',
    'present_conv_enc_4_b1_state',
    'present_conv_enc_6_state',
    'present_conv_enc_7_b1_state',
    'present_conv_enc_9_state',
    'present_conv_enc_10_b1_state',
    'present_conv_enc_12_state',
    'present_conv_enc_14_state',
    'present_conv_ds_state',
    'present_conv_us_state',
    'present_conv_dec_0_state',
    'present_conv_dec_2_state',
    'present_conv_dec_3_b1_state',
    'present_conv_dec_5_state',
    'present_conv_dec_6_b1_state',
    'present_conv_dec_8_state',
    'present_conv_dec_9_b1_state',
    'present_conv_dec_11_state',
    'present_conv_dec_12_b1_state',
    'present_conv_dec_14_state',
  ];
}

export function createInitialMimiState(ort: MimiOrtModule, convSpecs: MimiConvStateSpec[]) {
  const state: MimiOrtFeeds = {};

  for (const spec of convSpecs) {
    state[`past_conv_${spec.name}_state`] = new ort.Tensor(
      'float32',
      new Float32Array(spec.channels * spec.temporalSize),
      [1, spec.channels, spec.temporalSize],
    );
  }

  for (let index = 0; index < MIMI_KV_LAYERS; index += 1) {
    state[`past_key_${index}`] = new ort.Tensor('float32', new Float32Array(), [
      1,
      MIMI_KV_HEADS,
      0,
      MIMI_KV_HEAD_DIM,
    ]);
    state[`past_value_${index}`] = new ort.Tensor('float32', new Float32Array(), [
      1,
      MIMI_KV_HEADS,
      0,
      MIMI_KV_HEAD_DIM,
    ]);
  }

  return state;
}

export function updateMimiState(outputs: Ort.InferenceSession.ReturnType) {
  const state: MimiOrtFeeds = {};

  for (const [name, tensor] of Object.entries(outputs)) {
    if (name.startsWith('present_conv_')) {
      state[name.replace('present_conv_', 'past_conv_')] = tensor;
    } else if (name.startsWith('present_key_')) {
      state[name.replace('present_key_', 'past_key_')] = tensor;
    } else if (name.startsWith('present_value_')) {
      state[name.replace('present_value_', 'past_value_')] = tensor;
    }
  }

  return state;
}

function parseStateSpec(text: string): MimiStateSpec {
  const spec: MimiStateSpec = { encoder: [], decoder: [] };
  let section: keyof MimiStateSpec | null = null;

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) {
      continue;
    }

    if (line === '[encoder]' || line === '[decoder]') {
      section = line.slice(1, -1) as keyof MimiStateSpec;
      continue;
    }

    if (!section) {
      continue;
    }

    const [_kind, name, channels, temporalSize] = line.split(/\s+/);
    spec[section].push({
      name,
      channels: Number(channels),
      temporalSize: Number(temporalSize),
    });
  }

  return spec;
}

async function ensureWebGpuAvailable() {
  const gpu = (navigator as NavigatorWithGpu).gpu;
  if (!gpu?.requestAdapter) {
    throw new Error('Mimi richiede WebGPU');
  }

  let adapter: unknown;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (error) {
    throw new Error(`Mimi WebGPU adapter: ${errorMessage(error)}`);
  }

  if (!adapter) {
    throw new Error('Mimi WebGPU adapter non disponibile');
  }
}

async function loadMimiAssets() {
  const loadedAssets: Partial<Record<MimiAssetKey, Uint8Array>> = {};

  for (const asset of MIMI_RUNTIME_ASSETS) {
    loadedAssets[asset.key] = await loadAsset(asset, () => undefined);
  }

  return loadedAssets as Record<MimiAssetKey, Uint8Array>;
}

async function loadAsset(asset: RuntimeCacheAsset, onProgress: (loadedBytes: number) => void) {
  const url = assetUrl(asset.path);
  const cached = await readCachedAsset(asset, url);
  if (cached) {
    onProgress(asset.size);
    return cached;
  }

  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok) {
    throw new Error(`${asset.path}: HTTP ${response.status}`);
  }

  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    await validateAndCacheAsset(asset, url, bytes, response.headers);
    onProgress(asset.size);
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    chunks.push(value);
    loadedBytes += value.byteLength;
    onProgress(Math.min(asset.size, loadedBytes));
  }

  const bytes = concatUint8(chunks, loadedBytes);
  await validateAndCacheAsset(asset, url, bytes, response.headers);
  return bytes;
}

async function assetCached(asset: RuntimeCacheAsset) {
  if (!('caches' in window)) {
    return false;
  }

  try {
    return Boolean(await readCachedAsset(asset, assetUrl(asset.path)));
  } catch {
    return false;
  }
}

async function readCachedAsset(asset: RuntimeCacheAsset, url: string) {
  const memoryKey = assetMemoryKey(asset);
  const memoryAsset = runtimeAssetMemoryCache.get(memoryKey);
  if (memoryAsset?.byteLength === asset.size) {
    return memoryAsset;
  }

  if (!('caches' in window)) {
    return null;
  }

  try {
    const cache = await caches.open(asset.cacheName);
    const response = await cache.match(url, { ignoreSearch: true });
    if (!response?.ok) {
      return null;
    }

    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength !== asset.size) {
      await cache.delete(url, { ignoreSearch: true });
      return null;
    }

    runtimeAssetMemoryCache.set(memoryKey, bytes);
    return bytes;
  } catch {
    return null;
  }
}

async function validateAndCacheAsset(asset: RuntimeCacheAsset, url: string, bytes: Uint8Array, headers: Headers) {
  if (bytes.byteLength !== asset.size) {
    throw new Error(`${asset.path}: ${bytes.byteLength}/${asset.size} bytes`);
  }

  runtimeAssetMemoryCache.set(assetMemoryKey(asset), bytes);

  if (!('caches' in window)) {
    return;
  }

  try {
    const cache = await caches.open(asset.cacheName);
    const cachedHeaders = new Headers(headers);
    cachedHeaders.set('content-length', String(bytes.byteLength));
    await cache.put(url, new Response(arrayBufferFromBytes(bytes), { status: 200, headers: cachedHeaders }));
  } catch {
    return;
  }
}

function configureOrt(ort: MimiOrtModule) {
  ort.env.logLevel = 'error';
  ort.env.wasm.wasmPaths = new URL(`${ORT_ASSET_BASE}/`, location.origin).href;

  const webgpu = ort.env.webgpu as {
    powerPreference?: 'high-performance' | 'low-power';
  };
  webgpu.powerPreference = 'high-performance';
}

function assertCodes(tensor: MimiOrtTensor) {
  if (tensor.dims.length !== 3 || tensor.dims[0] !== 1 || tensor.dims[1] !== MIMI_CODEBOOKS) {
    throw new Error(`Mimi audio_codes shape inattesa: [${tensor.dims.join(', ')}]`);
  }
}

function requiredTensor(outputs: Ort.InferenceSession.ReturnType, name: string) {
  const tensor = outputs[name];
  if (!tensor) {
    throw new Error(`Mimi output mancante: ${name}`);
  }

  return tensor;
}

function syntheticWaveform(frame: number) {
  const output = new Float32Array(MIMI_FRAME_SAMPLES);
  const phaseOffset = frame * MIMI_FRAME_SAMPLES;
  for (let i = 0; i < output.length; i += 1) {
    const t = (phaseOffset + i) / 24_000;
    output[i] = 0.08 * Math.sin(2 * Math.PI * 220 * t) + 0.025 * Math.sin(2 * Math.PI * 440 * t);
  }
  return output;
}

function summarize(values: number[]): TimingSummary {
  const sorted = [...values].sort((left, right) => left - right);
  const sum = values.reduce((total, value) => total + value, 0);
  return {
    avg: sum / Math.max(1, values.length),
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    max: sorted[sorted.length - 1] ?? 0,
  };
}

function percentile(sorted: number[], percent: number) {
  if (sorted.length === 0) {
    return 0;
  }

  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * percent) - 1));
  return sorted[index];
}

export function disposeState(state: MimiOrtFeeds) {
  for (const tensor of Object.values(state)) {
    disposeTensor(tensor);
  }
}

export function disposeTensor(tensor: unknown) {
  const disposable = tensor as { dispose?: () => void } | null;
  disposable?.dispose?.();
}

function assetUrl(path: string) {
  return `${path}?v=${MIMI_ASSET_VERSION}`;
}

function assetMemoryKey(asset: RuntimeCacheAsset) {
  return `${asset.cacheName}:${asset.version}:${asset.path}`;
}

function totalRequiredBytes() {
  return REQUIRED_RUNTIME_ASSETS.reduce((total, asset) => total + asset.size, 0);
}

function progress(
  status: CodecModelCacheProgress['status'],
  loadedBytes: number,
  totalBytes: number,
): CodecModelCacheProgress {
  return {
    status,
    loadedBytes,
    totalBytes,
    percent: totalBytes === 0 ? 1 : Math.min(1, loadedBytes / totalBytes),
  };
}

function concatUint8(chunks: Uint8Array[], totalBytes: number) {
  const output = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

function textFromBytes(bytes: Uint8Array) {
  return new TextDecoder().decode(bytes);
}

function arrayBufferFromBytes(bytes: Uint8Array) {
  return new Uint8Array(bytes).buffer;
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

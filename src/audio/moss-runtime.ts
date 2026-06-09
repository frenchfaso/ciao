import type * as Ort from 'onnxruntime-web';
import { CIAO_ACTIVE_CODEC, type CodecModelCacheProgress } from './codec';

export type OrtModule = typeof Ort;
export type OrtTensor = Ort.Tensor;
export type OrtOutputs = Ort.InferenceSession.ReturnType;
export type OrtFeeds = Record<string, OrtTensor>;

export type MossMeta = {
  runtime_files?: {
    encode: string;
    decode_step: string;
  };
  codec_config: {
    sample_rate: number;
    channels: number;
    downsample_rate: number;
    num_quantizers: number;
  };
  streaming_decode: {
    transformer_offsets: Array<{
      input_name: string;
      output_name: string;
      shape: number[];
      dtype: 'int32';
    }>;
    attention_caches: Array<{
      offset_input_name: string;
      offset_output_name: string;
      cached_keys_input_name: string;
      cached_keys_output_name: string;
      cached_values_input_name: string;
      cached_values_output_name: string;
      cached_positions_input_name: string;
      cached_positions_output_name: string;
      offset_shape: number[];
      cache_shape: number[];
      positions_shape: number[];
      cache_dtype: 'float32' | 'float16';
      positions_dtype: 'int32';
    }>;
  };
};

type NavigatorWithGpu = Navigator & {
  gpu?: {
    requestAdapter?: (options?: { powerPreference?: 'high-performance' | 'low-power' }) => Promise<unknown>;
  };
};

type RuntimeCacheAsset = {
  key: string;
  label: string;
  path: string;
  version: string;
  size: number;
  cacheName: string;
};

export type MossRuntimeAssets = Record<'meta' | 'encoderModel' | 'decoderStepModel', Uint8Array>;

export type MossRuntime = {
  ort: OrtModule;
  meta: MossMeta;
  encoder: Ort.InferenceSession;
  decoder: Ort.InferenceSession;
};

type MossSessionTuning = {
  enableGraphCapture?: boolean;
  freeDimensionOverrides?: Record<string, number>;
  preferredOutputLocation?: Ort.InferenceSession.SessionOptions['preferredOutputLocation'];
};

type MossModelProfile = {
  assetVersion: string;
  cacheName: string;
  encoderModelFile: string;
  encoderModelSize: number;
  decoderModelFile: string;
  decoderModelSize: number;
};

const MOSS_MODEL_BASE = '/models/moss/audio-tokenizer-nano-onnx';
const ORT_ASSET_BASE = '/ort/1.26.0';
const ORT_CACHE_NAME = 'ciao-ort-runtime-v1';
const ORT_ASSET_VERSION = 'ort-1.26.0';
const MOSS_MODEL_PROFILE = {
  assetVersion: 'moss-nano-webgpu-fp16-v9-embedded-steps7',
  cacheName: 'ciao-moss-models-fp16-v9',
  encoderModelFile: 'moss_audio_tokenizer_encode.steps7.webgpu.onnx',
  encoderModelSize: 45_306_869,
  decoderModelFile: 'moss_audio_tokenizer_decode_step.fp16.steps7.webgpu.onnx',
  decoderModelSize: 23_897_423,
} as const satisfies MossModelProfile;

const MOSS_ASSET_VERSION = MOSS_MODEL_PROFILE.assetVersion;
const MOSS_MODEL_CACHE_NAME = MOSS_MODEL_PROFILE.cacheName;

const MOSS_RUNTIME_ASSETS = [
  {
    key: 'meta',
    label: 'meta',
    path: `${MOSS_MODEL_BASE}/codec_browser_onnx_meta.json`,
    version: MOSS_ASSET_VERSION,
    size: 16_118,
    cacheName: MOSS_MODEL_CACHE_NAME,
  },
  {
    key: 'encoderModel',
    label: 'encoder ONNX',
    path: `${MOSS_MODEL_BASE}/${MOSS_MODEL_PROFILE.encoderModelFile}`,
    version: MOSS_ASSET_VERSION,
    size: MOSS_MODEL_PROFILE.encoderModelSize,
    cacheName: MOSS_MODEL_CACHE_NAME,
  },
  {
    key: 'decoderStepModel',
    label: 'decoder ONNX',
    path: `${MOSS_MODEL_BASE}/${MOSS_MODEL_PROFILE.decoderModelFile}`,
    version: MOSS_ASSET_VERSION,
    size: MOSS_MODEL_PROFILE.decoderModelSize,
    cacheName: MOSS_MODEL_CACHE_NAME,
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

const REQUIRED_RUNTIME_ASSETS = [...MOSS_RUNTIME_ASSETS, ORT_WASM_ASSET] as const;

let ortPromise: Promise<OrtModule> | null = null;
let runtimePromise: Promise<MossRuntime> | null = null;
let runtimeCreationQueue: Promise<void> = Promise.resolve();
const runtimeAssetMemoryCache = new Map<string, Uint8Array>();

export async function warmMossModelCache(onProgress: (progress: CodecModelCacheProgress) => void) {
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
    await fetchAndCacheAsset(asset, (assetLoadedBytes) => {
      onProgress(progress('downloading', previousLoadedBytes + assetLoadedBytes, totalBytes));
    });
    loadedBytes = previousLoadedBytes + asset.size;
    onProgress(progress('cached', loadedBytes, totalBytes));
  }

  onProgress(progress('ready', totalBytes, totalBytes));
}

export async function loadMossRuntime(): Promise<MossRuntime> {
  if (runtimePromise) {
    return runtimePromise;
  }

  const nextRuntime = runtimeCreationQueue.then(() => createMossRuntime());
  runtimeCreationQueue = nextRuntime.then(
    () => undefined,
    () => undefined,
  );
  const guardedRuntime = nextRuntime.catch((error: unknown) => {
    if (runtimePromise === guardedRuntime) {
      runtimePromise = null;
    }
    throw error;
  });
  runtimePromise = guardedRuntime;
  return runtimePromise;
}

async function loadMossAssets(onProgress: (loadedBytes: number, totalBytes: number) => void) {
  const totalBytes = totalMossModelBytes();
  const loadedAssets: Record<string, Uint8Array> = {};
  let loadedBytes = 0;

  for (const asset of MOSS_RUNTIME_ASSETS) {
    const bytes = await loadAsset(asset, (assetLoadedBytes) => {
      onProgress(loadedBytes + assetLoadedBytes, totalBytes);
    });

    loadedAssets[asset.key] = bytes;
    loadedBytes += asset.size;
    onProgress(loadedBytes, totalBytes);
  }

  return loadedAssets as MossRuntimeAssets;
}

export async function ensureWebGpuAvailable() {
  const gpu = (navigator as NavigatorWithGpu).gpu;
  if (!gpu?.requestAdapter) {
    throw new Error('MOSS richiede WebGPU');
  }

  let adapter: unknown;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (error) {
    throw new Error(`MOSS WebGPU adapter: ${errorMessage(error)}`);
  }

  if (!adapter) {
    throw new Error('MOSS WebGPU adapter non disponibile');
  }
}

async function loadOrt() {
  if (!ortPromise) {
    const ortAssetPath = `${ORT_ASSET_BASE}/ort.webgpu.bundle.min.mjs`;
    ortPromise = import(/* @vite-ignore */ ortAssetPath)
      .then((ort) => {
        configureOrt(ort as OrtModule);
        return ort as OrtModule;
      })
      .catch((error: unknown) => {
        throw new Error(`MOSS runtime ORT: ${errorMessage(error)}`);
      });
  }

  const ort = await ortPromise;
  configureOrt(ort);
  return ort;
}

async function createModelSession(ort: OrtModule, model: Uint8Array, tuning: MossSessionTuning = {}) {
  const baseOptions = {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
    logVerbosityLevel: 0,
    ...tuning,
  } satisfies Ort.InferenceSession.SessionOptions;

  const optionCandidates = tuning.enableGraphCapture
    ? [baseOptions, { ...baseOptions, enableGraphCapture: false }]
    : [baseOptions];
  let firstError: unknown = null;

  for (const options of optionCandidates) {
    try {
      return await ort.InferenceSession.create(model, options as Ort.InferenceSession.SessionOptions);
    } catch (error) {
      firstError ??= error;
    }
  }

  throw firstError ?? new Error('MOSS session creation failed');
}

function mossEncoderSessionTuning(): MossSessionTuning {
  return {
    freeDimensionOverrides: {
      batch: 1,
      waveform_length: CIAO_ACTIVE_CODEC.inputSamples,
      code_length: CIAO_ACTIVE_CODEC.tokenSteps,
    },
  };
}

function mossDecoderSessionTuning(meta: MossMeta): MossSessionTuning {
  return {
    freeDimensionOverrides: decoderFreeDimensionOverrides(meta),
    preferredOutputLocation: decoderPreferredOutputLocation(meta),
  };
}

function decoderFreeDimensionOverrides(meta: MossMeta) {
  const overrides: Record<string, number> = {
    code_length: CIAO_ACTIVE_CODEC.tokenSteps,
    audio_length: CIAO_ACTIVE_CODEC.rawDecoderSamples,
    Castaudio_dim_0: 1,
    Castaudio_dim_1: meta.codec_config.channels,
  };

  for (const cache of meta.streaming_decode.attention_caches) {
    const index = cache.cached_keys_output_name.match(/_(\d+)$/)?.[1];
    if (!index) {
      continue;
    }

    overrides[`Sliceattn_cached_keys_out_${index}_dim_2`] = cache.cache_shape[2];
    overrides[`Sliceattn_cached_values_out_${index}_dim_2`] = cache.cache_shape[2];
    overrides[`Castattn_cached_positions_out_${index}_dim_1`] = cache.positions_shape[1];
  }

  return overrides;
}

function decoderPreferredOutputLocation(meta: MossMeta) {
  const locations: Record<string, 'cpu' | 'gpu-buffer'> = {
    audio: 'cpu',
    audio_lengths: 'cpu',
  };

  for (const offset of meta.streaming_decode.transformer_offsets) {
    locations[offset.output_name] = 'gpu-buffer';
  }

  for (const cache of meta.streaming_decode.attention_caches) {
    locations[cache.offset_output_name] = 'gpu-buffer';
    locations[cache.cached_keys_output_name] = 'gpu-buffer';
    locations[cache.cached_values_output_name] = 'gpu-buffer';
    locations[cache.cached_positions_output_name] = 'gpu-buffer';
  }

  return locations;
}

export function createInitialMossDecoderState(ort: OrtModule, meta: MossMeta): OrtFeeds {
  const state: OrtFeeds = {};

  for (const offset of meta.streaming_decode.transformer_offsets) {
    state[offset.input_name] = new ort.Tensor('int32', new Int32Array(product(offset.shape)), offset.shape);
  }

  for (const cache of meta.streaming_decode.attention_caches) {
    state[cache.offset_input_name] = new ort.Tensor('int32', new Int32Array(product(cache.offset_shape)), cache.offset_shape);
    state[cache.cached_keys_input_name] = createZeroCacheTensor(ort, cache.cache_shape);
    state[cache.cached_values_input_name] = createZeroCacheTensor(ort, cache.cache_shape);
    state[cache.cached_positions_input_name] = new ort.Tensor(
      'int32',
      new Int32Array(product(cache.positions_shape)),
      cache.positions_shape,
    );
  }

  return state;
}

function createZeroCacheTensor(ort: OrtModule, shape: number[]) {
  const size = product(shape);
  return new ort.Tensor('float16', new Uint16Array(size), shape);
}

export function updateMossDecoderState(outputs: OrtOutputs, meta: MossMeta): OrtFeeds {
  const state: OrtFeeds = {};

  for (const offset of meta.streaming_decode.transformer_offsets) {
    state[offset.input_name] = requiredTensor(outputs, offset.output_name);
  }

  for (const cache of meta.streaming_decode.attention_caches) {
    state[cache.offset_input_name] = requiredTensor(outputs, cache.offset_output_name);
    state[cache.cached_keys_input_name] = requiredTensor(outputs, cache.cached_keys_output_name);
    state[cache.cached_values_input_name] = requiredTensor(outputs, cache.cached_values_output_name);
    state[cache.cached_positions_input_name] = requiredTensor(outputs, cache.cached_positions_output_name);
  }

  return state;
}

export function disposeState(state: OrtFeeds) {
  for (const tensor of Object.values(state)) {
    disposeTensor(tensor);
  }
}

export function disposeTensor(tensor: unknown) {
  const disposable = tensor as { dispose?: () => void } | null;
  disposable?.dispose?.();
}

function totalMossModelBytes() {
  return MOSS_RUNTIME_ASSETS.reduce((total, asset) => total + asset.size, 0);
}

function totalRequiredBytes() {
  return REQUIRED_RUNTIME_ASSETS.reduce((total, asset) => total + asset.size, 0);
}

function configureOrt(ort: OrtModule) {
  ort.env.logLevel = 'error';
  ort.env.wasm.wasmPaths = `${ORT_ASSET_BASE}/`;

  const webgpu = ort.env.webgpu as {
    powerPreference?: 'high-performance' | 'low-power';
  };
  webgpu.powerPreference = 'high-performance';
}

async function createMossRuntime(): Promise<MossRuntime> {
  await ensureWebGpuAvailable();
  const assets = await loadMossAssets(() => undefined);
  const meta = JSON.parse(textFromBytes(assets.meta)) as MossMeta;
  const ort = await loadOrt();

  let encoderModel = assets.encoderModel;
  const encoder = await createModelSession(
    ort,
    encoderModel,
    mossEncoderSessionTuning(),
  );
  encoderModel = new Uint8Array();

  let decoderModel = assets.decoderStepModel;
  const decoder = await createModelSession(
    ort,
    decoderModel,
    mossDecoderSessionTuning(meta),
  );
  decoderModel = new Uint8Array();

  return { ort, meta, encoder, decoder };
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
      await deleteCachedAsset(asset);
      return null;
    }

    runtimeAssetMemoryCache.set(memoryKey, bytes);
    return bytes;
  } catch {
    return null;
  }
}

async function fetchAndCacheAsset(asset: RuntimeCacheAsset, onProgress: (loadedBytes: number) => void) {
  const url = assetUrl(asset.path);
  const response = await fetch(url, { cache: 'force-cache' });

  if (!response.ok) {
    throw new Error(`MOSS download ${asset.label}: HTTP ${response.status}`);
  }

  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onProgress(asset.size);
    await validateAndCacheAsset(asset, url, bytes, response.headers);
    return;
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

  await validateAndCacheAsset(asset, url, concatUint8(chunks, loadedBytes), response.headers);
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

async function deleteCachedAsset(asset: RuntimeCacheAsset) {
  const cache = await openAssetCache(asset);
  await cache?.delete(assetUrl(asset.path), { ignoreSearch: true });
}

async function openAssetCache(asset: RuntimeCacheAsset) {
  if (!('caches' in window)) {
    return null;
  }

  try {
    return await caches.open(asset.cacheName);
  } catch {
    return null;
  }
}

function requiredTensor(outputs: OrtOutputs, name: string) {
  const tensor = outputs[name];
  if (!tensor) {
    throw new Error(`MOSS decoder output mancante: ${name}`);
  }

  return tensor;
}

function progress(
  status: CodecModelCacheProgress['status'],
  loadedBytes: number,
  totalBytes: number,
): CodecModelCacheProgress {
  const percent = totalBytes > 0 ? Math.max(0, Math.min(1, loadedBytes / totalBytes)) : 0;
  return {
    status,
    loadedBytes,
    totalBytes,
    percent,
  };
}

function assetUrl(path: string) {
  const url = new URL(path, location.href);
  const asset = REQUIRED_RUNTIME_ASSETS.find((candidate) => candidate.path === path);
  if (asset?.version) {
    url.searchParams.set('v', asset.version);
  }
  return url.toString();
}

function assetMemoryKey(asset: RuntimeCacheAsset) {
  return `${asset.cacheName}:${asset.key}`;
}

function textFromBytes(bytes: Uint8Array) {
  return new TextDecoder().decode(bytes);
}

function product(values: number[]) {
  return values.reduce((total, value) => total * value, 1);
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

function arrayBufferFromBytes(bytes: Uint8Array) {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function errorMessage(error: unknown) {
  if (error instanceof Error && error.message) {
    return error.message;
  }

  return String(error || 'errore sconosciuto');
}

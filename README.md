# ciao

Minimal offline-first PWA for very-low-bandwidth peer-to-peer voice chat.

## Stack

- Vite + TypeScript
- `vite-plugin-pwa` with Workbox service worker and installable manifest
- WebRTC `RTCDataChannel` for P2P media transport
- Ephemeral WebSocket signaling for room links
- Go deploy server for static assets, signaling, and echo test
- MOSS Audio Tokenizer Nano through ONNX Runtime WebGPU

## Development

Open the repository in the dev container so dependencies stay out of the host
checkout.

```sh
npm run dev
```

## Build

```sh
npm run build
npm run preview
```

## PWA Behavior

CIAO ships an installable manifest, waving-hand SVG icon/favicons, and a Workbox
service worker. Chrome/Brave expose the native install UI when the app meets PWA
installability criteria; Safari/iOS installs from Share -> Add to Home Screen.

Service worker updates are accepted automatically only when the runtime is idle.
Active calls, echo tests, signaling sessions, and audio engines defer page reload
until the app is safe to refresh.

The MOSS and ORT assets are cached outside the Workbox precache. On startup the
app verifies Cache Storage, downloads missing complete files, shows progress, and
keeps the microphone disabled until WebGPU sessions are ready.

## Codec

CIAO uses one production codec profile:

- MOSS Audio Tokenizer Nano ONNX
- 48 kHz model rate
- mono capture/playback bridged to the stereo MOSS model
- 7 token steps per superframe
- 480 ms audio frame
- 16 RVQ codebooks, 10-bit code indices
- 140 raw codec bytes per frame, about 2.33 kbps before transport overhead
- fp32 encoder graph, fp16 decoder graph, fp16 K/V cache boundaries
- WebGPU execution provider only; no PCM fallback

The active vendored model files are:

```text
public/models/moss/audio-tokenizer-nano-onnx/codec_browser_onnx_meta.json
public/models/moss/audio-tokenizer-nano-onnx/moss_audio_tokenizer_encode.steps7.webgpu.onnx
public/models/moss/audio-tokenizer-nano-onnx/moss_audio_tokenizer_encode.data
public/models/moss/audio-tokenizer-nano-onnx/moss_audio_tokenizer_decode_step.fp16.steps7.webgpu.onnx
public/models/moss/audio-tokenizer-nano-onnx/moss_audio_tokenizer_decode_shared.fp16.data
```

Regenerate pinned assets from the dev container:

```sh
npm run moss:download
npm run moss:optimize
npm run moss:fp16
npm run moss:steps
npm run build
```

`moss:download` fetches upstream source files from Hugging Face.
`moss:optimize` writes static-shape WebGPU source graphs.
`moss:fp16` converts the decoder to the balanced fp16 profile and validates it
against the fp32 baseline.
`moss:steps` writes the active step-7 production graphs and removes transient
fp32/source variants from `public/models`.

Sources:

- MOSS Audio Tokenizer code: https://github.com/OpenMOSS/MOSS-Audio-Tokenizer
- MOSS Audio Tokenizer Nano ONNX export: https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX

## Deploy Container

The deploy image is multi-stage. Node builds `dist`, Go builds a static server,
and the final Alpine runtime contains only the server binary, static PWA assets,
and `wget` for the healthcheck.

```sh
npm run container:build
npm run container:run
```

Cloudflare named tunnel for `ciao.netstead.xyz`:

1. In Cloudflare Zero Trust, add a public hostname to the remotely managed
   tunnel:

   ```text
   Hostname: ciao.netstead.xyz
   Service:  http://ciao:8080
   ```

2. Start CIAO and the tunnel connector as separate Compose services:

   ```sh
   TUNNEL_TOKEN=... npm run compose:cloudflare
   ```

   Or put the token in a local `.env` file, which is ignored by git:

   ```sh
   cp .env.example .env
   npm run compose:cloudflare
   ```

With plain Docker:

```sh
docker build -t ciao:latest .
docker run --rm -p 8080:8080 ciao:latest
```

## Runtime Flow

1. The first peer taps copy-link, creating and copying a room URL.
2. The second peer opens the room URL.
3. The server exchanges WebRTC offer, answer, and ICE candidates over `/ws`.
4. Voice flows P2P through a WebRTC `RTCDataChannel`.
5. The `/echo` WebSocket endpoint echoes CIAO protocol audio frames for local
   setup testing without a second peer.

The WebRTC VoIP media stack is not used. Audio is MOSS token data carried by the
custom `CIAO` binary frame protocol over the DataChannel.

## Streaming Protocol

`ciao/moss-nano-native/v6/s7/cb16` carries MOSS-native audio payloads:

- raw packed MOSS token data, without a second codec packet header
- 1-2 superframes per DataChannel message
- VAD/DTX for silence
- receiver jitter buffer with adaptive 1-3 frame target
- packet-loss concealment by repeat/fade of the last decoded frame
- receiver feedback in `State` frames for loss, jitter, and buffer depth

The signaling server is intentionally ephemeral. It stores room participants only
while their WebSocket connections are open and removes empty rooms immediately.

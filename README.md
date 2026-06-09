<div align="center">

# 👋

# ciao

**A tiny offline-first PWA for very-low-bandwidth peer-to-peer voice chat.**

Share a link, connect through WebRTC DataChannel, and stream MOSS neural audio
tokens instead of using the browser VoIP media stack.

</div>

## Beta

- installable mobile PWA
- offline-ready app shell
- ephemeral signaling rooms
- P2P audio over `RTCDataChannel`
- server echo test for setup checks
- MOSS Audio Tokenizer Nano on WebGPU only
- no PCM fallback

## Development

Use the dev container. Do not run npm install/build commands on the host.

```sh
npm run dev
```

Npm app/model commands are guarded and fail outside a containerized environment
so `node_modules` and build tooling stay off the host checkout.

## Deploy

```sh
podman compose -f compose.cloudflare.yml up -d --build
```

The deploy image contains the built PWA, the Go signaling/echo server, and one
vendored MOSS encoder/decoder pair.

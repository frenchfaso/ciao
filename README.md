<div align="center">

# 👋

# ciao

**A tiny offline-first PWA for very-low-bandwidth peer-to-peer voice chat.**

Share a link, connect through WebRTC DataChannel, and stream Mimi neural audio
tokens instead of using the browser VoIP media stack.

</div>

## Beta

- installable mobile PWA
- offline-ready app shell
- ephemeral signaling rooms
- P2P audio over `RTCDataChannel`
- server echo test for setup checks
- Mimi streaming codec on WebGPU only
- no PCM fallback

## Development

Use the dev container. Do not run npm install/build commands on the host.

```sh
npm run dev
```

Npm app/model commands are guarded and fail outside a containerized environment
so `node_modules` and build tooling stay off the host checkout.

## Deploy

Local or LAN:

```sh
podman compose up -d --build
```

Optional Cloudflare Tunnel:

```sh
cp .env.example .env
podman compose -f compose.yml -f compose.cloudflare.yml up -d --build
```

The deploy image contains the built PWA, the Go signaling/echo server, and one
vendored Mimi encoder/decoder pair. Cloudflare is optional; keep tunnel tokens
in `.env`, which is ignored by git. Configure the Cloudflare public hostname in
the Cloudflare dashboard and point it to `http://ciao:8080`.

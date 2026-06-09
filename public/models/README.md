# MOSS codec assets

The active browser codec assets are vendored under:

```text
public/models/moss/audio-tokenizer-nano-onnx/
```

The pinned source is `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`.
CIAO uses WebGPU-specialized copies of the streaming encode and decode-step
ONNX export with static MOSS superframe shapes. The production profile uses the
fp32 encoder graph and fp16 decoder graph with fp16 attention K/V cache
input/output boundaries.

The selected export runs at 48 kHz, 12.5 token steps per second, 16 codebooks,
and 10-bit code indices. CIAO currently defaults to 7 token steps per 480 ms
superframe. The static decoder graph declares `[1, 2, 26880]` raw audio output
for seven token steps; the runtime trims that output to the original
23040-sample frame before playout.

The app expects one encoder model, one decoder model, and the small MOSS
metadata file. The ONNX external weight data is embedded into the two active
model files:

```text
public/models/moss/audio-tokenizer-nano-onnx/codec_browser_onnx_meta.json
public/models/moss/audio-tokenizer-nano-onnx/moss_audio_tokenizer_encode.steps7.webgpu.onnx
public/models/moss/audio-tokenizer-nano-onnx/moss_audio_tokenizer_decode_step.fp16.steps7.webgpu.onnx
```

Re-fetch or verify the pinned assets from the dev container:

```sh
npm run moss:download
npm run moss:optimize
npm run moss:fp16
npm run moss:steps
npm run build
```

The upstream `.onnx` source files, fp32 decoder files, and external `.data`
weight files are transient regeneration inputs. They are downloaded by
`moss:download`, converted by `moss:optimize` and `moss:fp16`, then embedded and
removed by `moss:steps` after the active step-7 production files have been
written.
The fp16 files are regenerated from the static fp32 `.webgpu.onnx` files and are
not downloaded from upstream. The conversion validates exact encoder token
match and decoder SNR against the fp32 baseline before replacing the vendored
fp16 files.

If those files are missing, `src/audio/moss-codec.ts` refuses to create an
audio codec. CIAO can still exercise signaling and DataChannel behavior, but it
does not send PCM audio.

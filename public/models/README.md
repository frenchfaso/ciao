# Mimi Assets

CIAO vendors the active Mimi streaming browser codec under:

```text
public/models/mimi/streaming-8cb-fp16/
```

The runtime set is generated before `dev` and `build`:

```text
encoder_model.onnx
decoder_model.onnx
state_spec.txt
```

The current models come from `BMekiker/mimi-onnx-streaming`, using the
`streaming-8cb-fp16` export with explicit convolution state and KV cache.
The large ONNX files are stored in Git as chunks under `vendor/models/` and
reassembled automatically inside the dev/deploy container.

Regenerate from the dev container:

```sh
npm run mimi:download
npm run build
```

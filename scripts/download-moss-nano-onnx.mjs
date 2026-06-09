import { createWriteStream, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(new URL('../package.json', import.meta.url)));
const targetDir = join(root, 'public', 'models', 'moss', 'audio-tokenizer-nano-onnx');
const sourceBase = 'https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX/resolve/main';

const files = [
  ['codec_browser_onnx_meta.json', 17_036],
  ['moss_audio_tokenizer_encode.onnx', 815_775],
  ['moss_audio_tokenizer_encode.data', 44_507_136],
  ['moss_audio_tokenizer_decode_step.onnx', 351_400],
  ['moss_audio_tokenizer_decode_shared.data', 44_198_912],
];

mkdirSync(targetDir, { recursive: true });

for (const [file, expectedBytes] of files) {
  const targetPath = join(targetDir, file);
  if (existsSync(targetPath) && statSync(targetPath).size === expectedBytes) {
    console.log(`cached ${file}`);
    continue;
  }

  const url = `${sourceBase}/${file}`;
  console.log(`downloading ${file}`);
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`download failed ${file}: HTTP ${response.status}`);
  }

  await pipeline(response.body, createWriteStream(targetPath));

  const actualBytes = statSync(targetPath).size;
  if (actualBytes !== expectedBytes) {
    throw new Error(`${file}: expected ${expectedBytes} bytes, got ${actualBytes}`);
  }
}

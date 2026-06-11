import { createReadStream, createWriteStream, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { once } from 'node:events';
import { dirname, join } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(new URL('../package.json', import.meta.url)));
const targetDir = join(root, 'public', 'models', 'mimi', 'streaming-8cb-fp16');
const vendorDir = join(root, 'vendor', 'models', 'mimi', 'streaming-8cb-fp16');
const sourceBase = 'https://huggingface.co/BMekiker/mimi-onnx-streaming/resolve/main/streaming-8cb-fp16';
const chunkBytes = 40_000_000;

const files = [
  ['encoder_model.onnx', 124_768_461],
  ['decoder_model.onnx', 97_478_284],
  ['state_spec.txt', 534],
];

mkdirSync(targetDir, { recursive: true });
mkdirSync(vendorDir, { recursive: true });

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

for (const [file] of files) {
  if (file.endsWith('.onnx')) {
    const targetPath = join(targetDir, file);
    if (existsSync(targetPath)) {
      await writeChunks(file, targetPath);
    }
  }
}

async function writeChunks(file, sourcePath) {
  for (const existing of readdirSync(vendorDir)) {
    if (existing.startsWith(`${file}.part`)) {
      rmSync(join(vendorDir, existing), { force: true });
    }
  }

  let partIndex = 0;
  let partBytes = 0;
  let output = createPartStream(file, partIndex);

  try {
    for await (const chunk of createReadStream(sourcePath)) {
      let offset = 0;
      while (offset < chunk.byteLength) {
        const availableBytes = chunkBytes - partBytes;
        const bytesToWrite = Math.min(availableBytes, chunk.byteLength - offset);
        const slice = chunk.subarray(offset, offset + bytesToWrite);
        if (!output.write(slice)) {
          await once(output, 'drain');
        }
        offset += bytesToWrite;
        partBytes += bytesToWrite;

        if (partBytes === chunkBytes && offset < chunk.byteLength) {
          output.end();
          await once(output, 'finish');
          partIndex += 1;
          partBytes = 0;
          output = createPartStream(file, partIndex);
        }
      }
    }

    output.end();
    await once(output, 'finish');
    console.log(`chunked ${file}`);
  } catch (error) {
    output.destroy();
    throw error;
  }
}

function createPartStream(file, partIndex) {
  const partName = `${file}.part${String(partIndex).padStart(2, '0')}`;
  return createWriteStream(join(vendorDir, partName));
}

import { createReadStream, createWriteStream, existsSync, mkdirSync, renameSync, rmSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(new URL('../package.json', import.meta.url)));
const modelDir = join(root, 'public', 'models', 'mimi', 'streaming-8cb-fp16');
const vendorDir = join(root, 'vendor', 'models', 'mimi', 'streaming-8cb-fp16');

const models = [
  {
    file: 'encoder_model.onnx',
    bytes: 124_768_461,
    parts: ['encoder_model.onnx.part00', 'encoder_model.onnx.part01', 'encoder_model.onnx.part02', 'encoder_model.onnx.part03'],
  },
  {
    file: 'decoder_model.onnx',
    bytes: 97_478_284,
    parts: ['decoder_model.onnx.part00', 'decoder_model.onnx.part01', 'decoder_model.onnx.part02'],
  },
];

mkdirSync(modelDir, { recursive: true });

for (const model of models) {
  const targetPath = join(modelDir, model.file);
  if (fileHasSize(targetPath, model.bytes)) {
    console.log(`model ready ${model.file}`);
    continue;
  }

  await assembleModel(model, targetPath);
  console.log(`model assembled ${model.file}`);
}

function fileHasSize(path, bytes) {
  return existsSync(path) && statSync(path).size === bytes;
}

async function assembleModel(model, targetPath) {
  const partPaths = model.parts.map((part) => join(vendorDir, part));
  const missingPart = partPaths.find((partPath) => !existsSync(partPath));
  if (missingPart) {
    throw new Error(`missing model chunk: ${missingPart}`);
  }

  const totalBytes = partPaths.reduce((sum, partPath) => sum + statSync(partPath).size, 0);
  if (totalBytes !== model.bytes) {
    throw new Error(`${model.file}: expected ${model.bytes} bytes from chunks, got ${totalBytes}`);
  }

  const tmpPath = `${targetPath}.tmp`;
  rmSync(tmpPath, { force: true });
  const output = createWriteStream(tmpPath);

  try {
    for (const partPath of partPaths) {
      await appendFile(output, partPath);
    }
    output.end();
    await once(output, 'finish');

    if (!fileHasSize(tmpPath, model.bytes)) {
      throw new Error(`${model.file}: assembled size mismatch`);
    }

    renameSync(tmpPath, targetPath);
  } catch (error) {
    output.destroy();
    rmSync(tmpPath, { force: true });
    throw error;
  }
}

async function appendFile(output, path) {
  for await (const chunk of createReadStream(path)) {
    if (!output.write(chunk)) {
      await once(output, 'drain');
    }
  }
}

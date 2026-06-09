import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(new URL('../package.json', import.meta.url)));
const ortAssetVersion = '1.26.0';
const sourceDir = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const targetRoot = join(root, 'public', 'ort');
const targetDir = join(targetRoot, ortAssetVersion);
const runtimeAssets = new Set([
  'ort.webgpu.bundle.min.mjs',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
]);

mkdirSync(targetRoot, { recursive: true });

for (const file of readdirSync(targetRoot)) {
  if (file !== '.gitkeep') {
    rmSync(join(targetRoot, file), { force: true, recursive: true });
  }
}

if (
  !existsSync(sourceDir) ||
  ![...runtimeAssets].every((asset) => existsSync(join(sourceDir, asset)))
) {
  process.exit(0);
}

mkdirSync(targetDir, { recursive: true });

for (const file of readdirSync(sourceDir)) {
  if (runtimeAssets.has(file)) {
    copyFileSync(join(sourceDir, file), join(targetDir, file));
  }
}

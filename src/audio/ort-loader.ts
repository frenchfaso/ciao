import type * as Ort from 'onnxruntime-web';

export type OrtWebGpuModule = typeof Ort;

const ortModulePromises = new Map<string, Promise<OrtWebGpuModule>>();

export async function importOrtWebGpuModule(path: string): Promise<OrtWebGpuModule> {
  const url = new URL(path, location.origin).href;
  const cached = ortModulePromises.get(url);
  if (cached) {
    return cached;
  }

  const promise = importOrtModule(url).catch((error: unknown) => {
    ortModulePromises.delete(url);
    throw error;
  });
  ortModulePromises.set(url, promise);
  return promise;
}

async function importOrtModule(url: string) {
  if (!shouldImportOrtViaBlob()) {
    return (await import(/* @vite-ignore */ url)) as OrtWebGpuModule;
  }

  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status}`);
  }

  const blobUrl = URL.createObjectURL(
    new Blob([await response.blob()], {
      type: 'text/javascript',
    }),
  );

  try {
    return (await import(/* @vite-ignore */ blobUrl)) as OrtWebGpuModule;
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

function shouldImportOrtViaBlob() {
  return location.hostname === 'localhost' || location.hostname === '127.0.0.1';
}

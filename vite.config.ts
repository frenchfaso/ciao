import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

const rootDir = fileURLToPath(new URL('.', import.meta.url));
const packageJson = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8')) as {
  version?: string;
};

export default defineConfig({
  define: {
    __CIAO_APP_VERSION__: JSON.stringify(buildAppVersion()),
  },
  build: {
    emptyOutDir: true,
  },
  plugins: [
    VitePWA({
      registerType: 'prompt',
      includeAssets: [
        'icons/icon.svg',
        'icons/apple-touch-icon.svg',
        'worklets/ciao-audio-worklet.js',
        'models/README.md',
      ],
      manifest: {
        id: '/',
        name: 'ciao',
        short_name: 'ciao',
        lang: 'it-IT',
        description: 'Offline-first PWA per voicechat P2P su WebRTC DataChannel.',
        start_url: '/',
        scope: '/',
        display: 'standalone',
        orientation: 'portrait',
        background_color: '#101418',
        theme_color: '#101418',
        categories: ['communication', 'utilities'],
        icons: [
          {
            src: '/icons/icon.svg',
            sizes: 'any',
            type: 'image/svg+xml',
            purpose: 'any maskable',
          },
          {
            src: '/icons/apple-touch-icon.svg',
            sizes: '180x180',
            type: 'image/svg+xml',
            purpose: 'any',
          },
        ],
      },
      workbox: {
        cleanupOutdatedCaches: true,
        clientsClaim: true,
        skipWaiting: false,
        navigateFallback: null,
        maximumFileSizeToCacheInBytes: 32 * 1024 * 1024,
        globPatterns: ['**/*.{js,mjs,css,html,svg,png,ico,webmanifest,wasm,json,md,txt}'],
        runtimeCaching: [
          {
            urlPattern: ({ request }) => request.mode === 'navigate',
            handler: 'NetworkFirst',
            options: {
              cacheName: 'ciao-pages',
              networkTimeoutSeconds: 2,
            },
          },
          {
            urlPattern: ({ url }) => url.pathname.startsWith('/ort/'),
            handler: 'CacheFirst',
            options: {
              cacheName: 'ciao-ort-runtime-v1',
              expiration: {
                maxEntries: 12,
                maxAgeSeconds: 60 * 60 * 24 * 365,
              },
            },
          },
          {
            urlPattern: ({ request }) =>
              ['script', 'style', 'worker', 'manifest', 'image'].includes(request.destination),
            handler: 'CacheFirst',
            options: {
              cacheName: 'ciao-static',
              expiration: {
                maxEntries: 80,
                maxAgeSeconds: 60 * 60 * 24 * 30,
              },
            },
          },
        ],
      },
      devOptions: {
        enabled: true,
        type: 'module',
      },
    }),
  ],
});

function buildAppVersion() {
  const version = packageJson.version ?? '0.0.0';
  const gitHash = readGitHash();
  const buildStamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
  return `v${version}+${gitHash ? `${gitHash}.` : ''}${buildStamp}`;
}

function readGitHash() {
  try {
    return execFileSync('git', ['rev-parse', '--short=8', 'HEAD'], {
      cwd: rootDir,
      stdio: ['ignore', 'pipe', 'ignore'],
    })
      .toString()
      .trim();
  } catch {
    return '';
  }
}

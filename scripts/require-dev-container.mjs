import { existsSync } from 'node:fs';

const isContainer =
  existsSync('/.dockerenv') ||
  existsSync('/run/.containerenv') ||
  process.env.CODESPACES === 'true' ||
  process.env.DEVCONTAINER === 'true' ||
  process.env.CI === 'true';

if (!isContainer) {
  console.error(
    [
      'CIAO npm commands must run inside the dev container or another containerized build.',
      '',
      'Open this repository in the configured dev container, then run npm commands there.',
      'This prevents node_modules and build tooling from being installed on the host.',
    ].join('\n'),
  );
  process.exit(1);
}

import { CIAO_ACTIVE_CODEC } from '../audio/codec';

export const CIAO_PROTOCOL = `ciao/${CIAO_ACTIVE_CODEC.id}/v9/s${CIAO_ACTIVE_CODEC.tokenSteps}/cb${CIAO_ACTIVE_CODEC.codebooks}`;

const MAGIC = 0x4349414f;
const VERSION = 1;
const HEADER_BYTES = 16;

export enum CiaoFrameType {
  Hello = 1,
  State = 3,
  Bye = 4,
}

export type CiaoFrame = {
  type: CiaoFrameType;
  sequence: number;
  timestamp: number;
  payload: Uint8Array;
};

export function encodeFrame(frame: CiaoFrame) {
  const output = new ArrayBuffer(HEADER_BYTES + frame.payload.byteLength);
  const view = new DataView(output);
  view.setUint32(0, MAGIC, false);
  view.setUint8(4, VERSION);
  view.setUint8(5, frame.type);
  view.setUint8(6, 0);
  view.setUint8(7, HEADER_BYTES);
  view.setUint32(8, frame.sequence, true);
  view.setUint32(12, frame.timestamp, true);
  new Uint8Array(output, HEADER_BYTES).set(frame.payload);
  return output;
}

export function decodeFrame(data: ArrayBuffer): CiaoFrame | null {
  if (data.byteLength < HEADER_BYTES) {
    return null;
  }

  const view = new DataView(data);
  if (view.getUint32(0, false) !== MAGIC || view.getUint8(4) !== VERSION) {
    return null;
  }

  const headerBytes = view.getUint8(7);
  if (headerBytes < HEADER_BYTES || headerBytes > data.byteLength) {
    return null;
  }

  const type = view.getUint8(5);
  if (!isFrameType(type)) {
    return null;
  }

  return {
    type,
    sequence: view.getUint32(8, true),
    timestamp: view.getUint32(12, true),
    payload: new Uint8Array(data.slice(headerBytes)),
  };
}

function isFrameType(value: number): value is CiaoFrameType {
  return (
    value === CiaoFrameType.Hello ||
    value === CiaoFrameType.State ||
    value === CiaoFrameType.Bye
  );
}

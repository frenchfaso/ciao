class CiaoCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSize = Math.max(128, Math.round(sampleRate * 0.08));
    this.buffer = new Float32Array(this.frameSize);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];

    if (!input) {
      return true;
    }

    for (let i = 0; i < input.length; i += 1) {
      this.buffer[this.offset] = input[i];
      this.offset += 1;

      if (this.offset === this.frameSize) {
        const frame = this.buffer;
        this.port.postMessage(frame, [frame.buffer]);
        this.buffer = new Float32Array(this.frameSize);
        this.offset = 0;
      }
    }

    return true;
  }
}

class CiaoPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.current = null;
    this.offset = 0;
    this.port.onmessage = (event) => {
      this.queue.push(event.data);
    };
  }

  process(_, outputs) {
    const output = outputs[0]?.[0];

    if (!output) {
      return true;
    }

    for (let i = 0; i < output.length; i += 1) {
      if (!this.current || this.offset >= this.current.length) {
        this.current = this.queue.shift() ?? null;
        this.offset = 0;
      }

      output[i] = this.current ? this.current[this.offset] : 0;
      this.offset += 1;
    }

    return true;
  }
}

registerProcessor('ciao-capture', CiaoCaptureProcessor);
registerProcessor('ciao-playback', CiaoPlaybackProcessor);

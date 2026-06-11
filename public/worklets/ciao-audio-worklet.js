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
    this.lastOutput = 0;
    this.previousTail = null;
    this.rampStart = 0;
    this.rampOffset = 0;
    this.rampSamples = 0;
    this.crossfadeSamples = 0;
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
        if (this.current && this.current.length > 0) {
          const tailSamples = Math.min(this.crossfadeSamples, this.current.length);
          this.previousTail = this.current.slice(this.current.length - tailSamples);
        }

        this.current = this.queue.shift() ?? null;
        this.offset = 0;
        if (this.current) {
          this.rampStart = this.lastOutput;
          this.rampOffset = 0;
          this.rampSamples = Math.min(this.current.length, this.previousTail ? this.previousTail.length : Math.round(sampleRate * 0.004));
        } else {
          this.previousTail = null;
          this.rampOffset = 0;
          this.rampSamples = 0;
        }
      }

      let sample = this.current ? this.current[this.offset] : 0;
      if (this.current && this.rampOffset < this.rampSamples) {
        const alpha = (this.rampOffset + 1) / this.rampSamples;
        const previous = this.previousTail?.[this.rampOffset] ?? this.rampStart;
        sample = previous * (1 - alpha) + sample * alpha;
        this.rampOffset += 1;
        if (this.rampOffset >= this.rampSamples) {
          this.previousTail = null;
        }
      }

      output[i] = sample;
      this.lastOutput = sample;
      this.offset += 1;
    }

    return true;
  }
}

registerProcessor('ciao-capture', CiaoCaptureProcessor);
registerProcessor('ciao-playback', CiaoPlaybackProcessor);

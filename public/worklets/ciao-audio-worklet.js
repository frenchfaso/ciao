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
    this.rampStart = 0;
    this.rampOffset = 0;
    this.rampSamples = 0;
    this.wasUnderrun = true;
    this.rampInSamples = Math.max(1, Math.round(sampleRate * 0.002));
    this.silenceRampOffset = 0;
    this.silenceRampSamples = Math.max(1, Math.round(sampleRate * 0.002));
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
      if (this.current && this.offset >= this.current.length) {
        this.current = null;
        this.offset = 0;
        this.silenceRampOffset = 0;
        this.port.postMessage({ type: 'played' });
      }

      if (!this.current && this.queue.length > 0) {
        this.current = this.queue.shift() ?? null;
        this.offset = 0;
        if (this.current) {
          this.rampStart = this.lastOutput;
          this.rampOffset = 0;
          this.rampSamples = this.wasUnderrun ? Math.min(this.current.length, this.rampInSamples) : 0;
          this.silenceRampOffset = 0;
          this.wasUnderrun = false;
        } else {
          this.rampOffset = 0;
          this.rampSamples = 0;
        }
      }

      let sample = this.current ? this.current[this.offset] : 0;
      if (this.current && this.rampOffset < this.rampSamples) {
        const alpha = (this.rampOffset + 1) / this.rampSamples;
        sample = this.rampStart * (1 - alpha) + sample * alpha;
        this.rampOffset += 1;
      }
      if (!this.current && this.silenceRampOffset < this.silenceRampSamples) {
        const alpha = (this.silenceRampOffset + 1) / this.silenceRampSamples;
        sample = this.lastOutput * (1 - alpha);
        this.silenceRampOffset += 1;
        this.wasUnderrun = true;
      } else if (!this.current) {
        this.wasUnderrun = true;
      }

      output[i] = sample;
      this.lastOutput = sample;
      if (this.current) {
        this.offset += 1;
      }
    }

    return true;
  }
}

registerProcessor('ciao-capture', CiaoCaptureProcessor);
registerProcessor('ciao-playback', CiaoPlaybackProcessor);

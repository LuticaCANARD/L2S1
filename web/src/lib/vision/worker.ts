/// <reference lib="webworker" />
import { env, pipeline, type ObjectDetectionPipeline } from '@huggingface/transformers';
import config from './detector.json';
import type { DetectorOutput } from './types';
let detector: ObjectDetectionPipeline | undefined;
let busy = false;
const post = (message: DetectorOutput) => self.postMessage(message);
self.onmessage = async (event: MessageEvent<{ type: 'detect'; image: string }>) => {
  if (busy || event.data.type !== 'detect') return;
  busy = true;
  try {
    if (!detector) {
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      if (env.backends.onnx.wasm) env.backends.onnx.wasm.numThreads = 1;
      detector = await pipeline('object-detection', config.model, {
        revision: config.revision, dtype: 'q8', device: 'wasm',
        progress_callback: (info) => {
          if ('file' in info) post({ type: 'progress', file: info.file, progress: 'progress' in info ? info.progress : info.status === 'done' ? 100 : 0 });
        }
      });
    }
    post({ type: 'ready' });
    const start = performance.now();
    const detections = await detector(event.data.image, { threshold: config.minimum_score, percentage: true });
    post({ type: 'result', detections, elapsed_ms: performance.now() - start });
  } catch (cause) {
    post({ type: 'error', message: cause instanceof Error ? cause.message : String(cause) });
  } finally { busy = false; }
};

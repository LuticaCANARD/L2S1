// Actual model inference, not hand-authored bounding boxes. Run from web/.
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pipeline, env, RawImage } from '@huggingface/transformers';
const config = JSON.parse(await readFile(new URL('../src/lib/vision/detector.json', import.meta.url)));
const gallery = JSON.parse(await readFile(new URL('../static/trashnet/recorded.json', import.meta.url)));
env.cacheDir = process.env.L2S1_DETECT_CACHE || '/tmp/l2s1-detect-cache';
const detector = await pipeline('object-detection', config.model, { revision: config.revision, dtype: config.dtype, device: 'cpu' });
const samples = [];
for (const label of gallery.source.classes) {
  for (const item of gallery.source.records.filter((row) => row.label === label).slice(0, 2)) {
    const file = new URL('../static' + item.image_url, import.meta.url);
    const image = await RawImage.read(file.pathname);
    const start = performance.now();
    const detections = await detector(image, { threshold: config.minimum_score, percentage: true });
    const elapsed_ms = performance.now() - start;
    samples.push({ ...item, width: image.width, height: image.height, elapsed_ms, detections });
    console.log(item.name, detections.filter((row) => row.score >= 0.5).map((row) => [row.label, row.score]));
  }
}
const modelPath = `${env.cacheDir}/${config.model}/${config.revision}/onnx/model_quantized.onnx`;
const modelBytes = await readFile(modelPath);
const model_sha256 = createHash('sha256').update(modelBytes).digest('hex');
await writeFile(new URL('../static/trashnet/detections.json', import.meta.url), JSON.stringify({ ...config, model_sha256, model_bytes: modelBytes.length, recorded_at: new Date().toISOString(), runtime: 'Transformers.js 4.3.0 / ONNX Runtime / CPU', coordinates: 'normalized_xyxy', samples }, null, 2) + '\n');
await detector.dispose();

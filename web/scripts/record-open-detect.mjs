// Frozen vocabulary and real inference; labels/filenames are never model inputs.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pipeline, env, RawImage } from '@huggingface/transformers';
const config = JSON.parse(await readFile(new URL('../src/lib/vision/open-detector.json', import.meta.url)));
const gallery = JSON.parse(await readFile(new URL('../static/trashnet/recorded.json', import.meta.url)));
env.cacheDir = process.env.L2S1_DETECT_CACHE || '../results/vision-improvement-20260927/onnx-cache';
const detector = await pipeline('zero-shot-object-detection', config.model, { revision: config.revision, dtype: 'q8', device: 'cpu' });
const samples = [];
const { suppressOverlaps } = await import('../src/lib/vision/nms.js');
// Fixed first two images of every class, identical to the original DETR recording.
for (const label of gallery.source.classes) {
  for (const item of gallery.source.records.filter(row => row.label === label).slice(0, 2)) {
    const image = await RawImage.read(new URL('../static' + item.image_url, import.meta.url).pathname);
    const start = performance.now();
    const raw = await detector(image, config.labels, { threshold: config.minimum_score, percentage: true });
    const detections = suppressOverlaps(raw, config.iou_threshold);
    const elapsed_ms = performance.now() - start;
    samples.push({ ...item, width: image.width, height: image.height, elapsed_ms, detections });
    console.log(item.name, detections.filter(row => row.score >= config.display_score).map(row => [row.label, row.score]));
  }
}
const bytes = await readFile(`${env.cacheDir}/${config.model}/${config.revision}/onnx/model_quantized.onnx`);
const output = { ...config, model_sha256: createHash('sha256').update(bytes).digest('hex'), model_bytes: bytes.length, recorded_at: new Date().toISOString(), runtime: 'Transformers.js 4.3.0 / ONNX Runtime / CPU / Q8', coordinates: 'normalized_xyxy', samples };
await mkdir(new URL('../static/trashnet/', import.meta.url), { recursive: true });
await writeFile(new URL('../static/trashnet/open-detections.json', import.meta.url), JSON.stringify(output, null, 2) + '\n');
await detector.dispose();

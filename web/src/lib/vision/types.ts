export type Sample = { name: string; label: string; sha256: string; image_url: string };
export type Observation = { image: string; ground_truth: string; selected: string | null; raw_top1: string; candidate_mass: number | null; top_option_probability: number; abstention_reasons: string[]; latency_ms: number | null; scores: { id: string; option_probability: number }[] };
export type GalleryModel = { id: string; name: string; observations: Record<string, Observation>; runtime?: string; recorded_at?: string; supervised?: boolean };
export type Improvement = { model: GalleryModel; test_records: Sample[]; report: { classifier: string; split_counts: Record<string, number>; evaluations: Record<string, { correct: number; total: number; accuracy: number; balanced_accuracy: number; confusion_matrix: number[][] }> }; };
export type Gallery = {
  recorded_at: string; runtime: string; prompt: string;
  policy: { min_top_probability: number; min_candidate_mass: number };
  source: { source_commit: string; classes: string[]; records: Sample[] };
  models: GalleryModel[];
};
export type Detection = { label: string; score: number; box: { xmin: number; ymin: number; xmax: number; ymax: number } };
export type DetectionSample = Sample & { width: number; height: number; elapsed_ms: number; detections: Detection[] };
export type DetectionRecording = { model: string; revision: string; recorded_at: string; runtime: string; samples: DetectionSample[] };
export type DetectorOutput =
  | { type: 'progress'; file: string; progress: number }
  | { type: 'ready' }
  | { type: 'result'; detections: Detection[]; elapsed_ms: number }
  | { type: 'error'; message: string };
export function outcome(row: Observation): 'correct' | 'wrong' | 'abstained' {
  return row.selected === null ? 'abstained' : row.selected === row.ground_truth ? 'correct' : 'wrong';
}
export function visibleDetections(rows: Detection[], threshold: number) {
  return rows.filter((row) => Number.isFinite(row.score) && row.score >= threshold && Object.values(row.box).every(Number.isFinite))
    .map((row) => ({ ...row, box: {
      xmin: Math.max(0, Math.min(1, row.box.xmin)), ymin: Math.max(0, Math.min(1, row.box.ymin)),
      xmax: Math.max(0, Math.min(1, row.box.xmax)), ymax: Math.max(0, Math.min(1, row.box.ymax))
    } })).filter((row) => row.box.xmax > row.box.xmin && row.box.ymax > row.box.ymin)
    .sort((a, b) => b.score - a.score);
}

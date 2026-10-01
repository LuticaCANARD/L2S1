/** Class-agnostic suppression: one physical object should not get many labels.
 * @param {import('./types').Detection[]} rows
 * @param {number} threshold
 */
export function suppressOverlaps(rows, threshold = 0.4) {
  /** @type {import('./types').Detection[]} */
  const kept = [];
  for (const row of [...rows].sort((a, b) => b.score - a.score)) {
    const a = row.box;
    if (!Number.isFinite(row.score) || !Object.values(a).every(Number.isFinite) || a.xmax <= a.xmin || a.ymax <= a.ymin) continue;
    if (kept.some(other => {
      const b = other.box;
      const overlap = Math.max(0, Math.min(a.xmax, b.xmax) - Math.max(a.xmin, b.xmin)) * Math.max(0, Math.min(a.ymax, b.ymax) - Math.max(a.ymin, b.ymin));
      const union = (a.xmax - a.xmin) * (a.ymax - a.ymin) + (b.xmax - b.xmin) * (b.ymax - b.ymin) - overlap;
      return overlap / union > threshold;
    })) continue;
    kept.push(row);
  }
  return kept;
}

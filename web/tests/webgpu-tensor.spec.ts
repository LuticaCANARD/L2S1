import { expect, test } from '@playwright/test';
import { halfToNumber, lastLogits, type LogitTensor } from '../src/lib/webgpu/tensor';

function cpuTensor(type: string, dims: number[], data: ArrayLike<number>): LogitTensor {
  return { type, dims, data, location: 'cpu', ort_tensor: { getData: async () => { throw new Error('CPU tensor must not request GPU readback'); } } };
}

test('IEEE binary16 decodes signed zeros, subnormal boundaries, normal limits and nonfinite values', () => {
  expect(Object.is(halfToNumber(0x0000), 0)).toBeTruthy();
  expect(Object.is(halfToNumber(0x8000), -0)).toBeTruthy();
  for (const [bits, expected] of [
    [0x0001, 2 ** -24], [0x03ff, 1023 * 2 ** -24], [0x0400, 2 ** -14],
    [0x3c00, 1], [0xbc00, -1], [0x3800, 0.5], [0x7bff, 65504], [0xfbff, -65504]
  ]) expect(halfToNumber(bits)).toBe(expected);
  expect(halfToNumber(0x7c00)).toBe(Infinity);
  expect(halfToNumber(0xfc00)).toBe(-Infinity);
  expect(Number.isNaN(halfToNumber(0x7e00))).toBeTruthy();
  expect(Number.isNaN(halfToNumber(0xfe01))).toBeTruthy();
});

test('full-prefill float32 and float64 outputs select the last sequence row and copy the values', async () => {
  for (const [type, data] of [
    ['float32', Float32Array.from([90, 91, 92, -2, 0, 3])],
    ['float64', Float64Array.from([90, 91, 92, -2, 0, 3])]
  ] as const) {
    const output = await lastLogits(cpuTensor(type, [1, 2, 3], data));
    expect(output).toBeInstanceOf(Float64Array);
    expect(Array.from(output)).toEqual([-2, 0, 3]);
    data[3] = 100;
    expect(output[0]).toBe(-2);
  }
});

test('packed float16 logits decode the final row rather than treating bit patterns as numeric scores', async () => {
  const data = Uint16Array.from([0x7bff, 0x7bff, 0x7bff, 0x7bff, 0xbc00, 0x0001, 0x3800, 0x4000]);
  expect(Array.from(await lastLogits(cpuTensor('float16', [1, 2, 4], data)))).toEqual([-1, 2 ** -24, 0.5, 2]);
  // Runtimes exposing already decoded Float16Array values also follow this path.
  expect(Array.from(await lastLogits(cpuTensor('float16', [1, 1, 3], Float64Array.from([-1, 0.5, 2]))))).toEqual([-1, 0.5, 2]);
});

test('GPU logits await readback and use the returned data without accessing the CPU data property', async () => {
  let resolveReadback!: (data: ArrayLike<number>) => void;
  let calls = 0;
  let completed = false;
  const readback = new Promise<ArrayLike<number>>((resolve) => { resolveReadback = resolve; });
  const tensor: LogitTensor = {
    type: 'float16', dims: [1, 2, 2], location: 'gpu-buffer',
    get data(): ArrayLike<number> { throw new Error('GPU data must be read back asynchronously'); },
    ort_tensor: { getData: () => { calls++; return readback; } }
  };
  const pending = lastLogits(tensor).then((result) => { completed = true; return result; });
  await Promise.resolve();
  expect(calls).toBe(1);
  expect(completed).toBeFalsy();
  resolveReadback(Uint16Array.from([0x7bff, 0x7bff, 0xbc00, 0x3c00]));
  expect(Array.from(await pending)).toEqual([-1, 1]);
  expect(completed).toBeTruthy();
});

test('unsupported dtype, batch size, dimensions and truncated output fail explicitly', async () => {
  for (const dims of [[1, 2], [2, 1, 2], [1, 0, 2], [1, -1, 2], [1, 1.5, 2], [1, 1, NaN]]) {
    await expect(lastLogits(cpuTensor('float32', dims, [0, 1]))).rejects.toThrow();
  }
  await expect(lastLogits(cpuTensor('int64', [1, 1, 2], [0, 1]))).rejects.toThrow('dtype');
  await expect(lastLogits(cpuTensor('float32', [1, 2, 2], [0, 1, 2]))).rejects.toThrow('차원');
});

test('failed GPU readback propagates the failure without returning fabricated logits', async () => {
  const tensor: LogitTensor = { ...cpuTensor('float32', [1, 1, 2], []), location: 'gpu-buffer', ort_tensor: { getData: async () => { throw new Error('device lost'); } } };
  await expect(lastLogits(tensor)).rejects.toThrow('device lost');
});

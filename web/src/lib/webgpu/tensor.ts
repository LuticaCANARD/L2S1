import { WebgpuMessageError, type WebgpuLocale } from '../i18n/webgpu';
export type LogitTensor = { type: string; dims: readonly number[]; location: string; data: ArrayLike<number | bigint | string>; ort_tensor: { getData(): Promise<ArrayLike<number | bigint | string>> } };
export function halfToNumber(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1; const exponent = bits >>> 10 & 31; const fraction = bits & 1023;
  return exponent === 0 ? sign * 2 ** -14 * (fraction / 1024) : exponent === 31 ? fraction ? NaN : sign * Infinity : sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}
export async function lastLogits(tensor: LogitTensor, locale: WebgpuLocale = 'ko'): Promise<Float64Array> {
  if (!['float32', 'float16', 'float64'].includes(tensor.type)) throw new WebgpuMessageError('errorLogitsDtype', { dtype: tensor.type }, locale);
  if (tensor.dims.length !== 3 || tensor.dims[0] !== 1 || tensor.dims.some((size) => !Number.isInteger(size) || size < 1)) throw new WebgpuMessageError('errorLogitsShape', {}, locale);
  const data = tensor.location === 'gpu-buffer' ? await tensor.ort_tensor.getData() : tensor.data;
  const vocabulary = tensor.dims[2];
  if (data.length !== tensor.dims.reduce((product, size) => product * size, 1)) throw new WebgpuMessageError('errorLogitsDimensions', {}, locale);
  const offset = data.length - vocabulary;
  const values = new Float64Array(vocabulary);
  for (let i = 0; i < vocabulary; i++) values[i] = tensor.type === 'float16' && data instanceof Uint16Array ? halfToNumber(Number(data[offset+i])) : Number(data[offset+i]);
  return values;
}

import { writable } from 'svelte/store';
export const locales = ['ko', 'en', 'ja'] as const;
export type Locale = typeof locales[number];
export type Theme = 'system' | 'light' | 'dark';
export const locale = writable<Locale>('en');
export const theme = writable<Theme>('system');
export function isLocale(value: unknown): value is Locale { return locales.some((item) => item === value); }
export function isTheme(value: unknown): value is Theme { return value === 'system' || value === 'light' || value === 'dark'; }
export function translate<T extends Record<string, string>>(language: Locale, catalog: { en: T; ko: Record<keyof T, string>; ja: Record<keyof T, string> }, key: keyof T, parameters: Record<string, string | number> = {}): string {
  return catalog[language][key].replace(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g, (placeholder, name: string) => parameters[name] === undefined ? placeholder : String(parameters[name]));
}
export const preferenceKeys = { locale: 'l2s1-locale', theme: 'l2s1-theme' };

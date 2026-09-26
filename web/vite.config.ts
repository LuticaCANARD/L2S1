import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
export default defineConfig({
  plugins: [sveltekit()],
  worker: { format: 'es' },
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'] },
  server: { proxy: {
    '/quality-inference': { target: process.env.L2S1_QUALITY_URL || 'http://127.0.0.1:8082', rewrite: (path) => path.replace(/^\/quality-inference/, '') },
    '/inference': { target: 'http://127.0.0.1:8080', rewrite: (path) => path.replace(/^\/inference/, '') },
    '/text-inference': { target: 'http://127.0.0.1:8081', rewrite: (path) => path.replace(/^\/text-inference/, '') }
  } }
});

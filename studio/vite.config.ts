import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/',
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    proxy: {
      '/v1': {
        target: (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.NUR_API
          || 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
});

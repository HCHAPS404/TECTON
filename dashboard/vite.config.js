import {defineConfig} from 'vite';
import {nodePolyfills} from 'vite-plugin-node-polyfills';

export default defineConfig({
  base: './',
  plugins: [nodePolyfills({include:['assert','events','buffer','process','util'],globals:{Buffer:true,process:true,global:false}})],
  define: {'process.env.NODE_ENV': JSON.stringify('production'), global: 'globalThis'},
  build: {target: 'es2022', chunkSizeWarningLimit: 2500},
});

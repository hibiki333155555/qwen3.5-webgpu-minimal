# Qwen3.5 WebGPU Minimal

An intentionally minimal local WebGPU LLM demo for Qwen3.5 0.8B ONNX through Transformers.js.

## Demo

<video src="assets/qwen35-webgpu-minimal-demo.mp4" poster="assets/qwen35-webgpu-minimal-demo-poster.png" controls muted width="100%"></video>

Prompt: `describe this image`

## Run

```sh
bun run dev
```

Open http://localhost:3000.

The page downloads Transformers.js and the ONNX model in the browser. Inference runs locally with WebGPU after the assets are loaded.

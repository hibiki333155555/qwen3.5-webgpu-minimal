import {
  AutoProcessor,
  env,
  InterruptableStoppingCriteria,
  Qwen3_5ForConditionalGeneration,
  RawImage,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const modelId = "onnx-community/Qwen3.5-0.8B-ONNX-OPT";
const plainAnswerInstruction =
  "Answer as simply as possible. Do not use Markdown. Output plain text only.";

env.useBrowserCache = true;
env.useWasmCache = true;
env.allowRemoteModels = true;

const messages = document.querySelector("#messages");
const loadStatus = document.querySelector("#loadStatus");
const loadText = document.querySelector("#loadText");
const progressBar = document.querySelector("#progressBar");
const imagePreview = document.querySelector("#imagePreview");
const previewImage = document.querySelector("#previewImage");
const form = document.querySelector("#composer");
const imageInput = document.querySelector("#imageInput");
const imageButton = document.querySelector("#imageButton");
const promptInput = document.querySelector("#prompt");
const sendButton = document.querySelector("#sendButton");

if (
  !messages ||
  !loadStatus ||
  !loadText ||
  !progressBar ||
  !imagePreview ||
  !previewImage ||
  !form ||
  !imageInput ||
  !imageButton ||
  !promptInput ||
  !sendButton
) {
  throw new Error("Missing UI element");
}

let processor = null;
let model = null;
let selectedImage = null;
let loadingPromise = null;
let isGenerating = false;

const stoppingCriteria = new InterruptableStoppingCriteria();
const progressByFile = new Map();

function updateProgress(event) {
  const file = event.file ?? event.name ?? event.url ?? event.status ?? "model";

  if (typeof event.progress === "number") {
    progressByFile.set(file, Math.max(0, Math.min(100, event.progress)));
  } else if (event.status === "done" || event.status === "ready") {
    progressByFile.set(file, 100);
  } else if (!progressByFile.has(file)) {
    progressByFile.set(file, 0);
  }

  const values = [...progressByFile.values()];
  const percent = values.length
    ? Math.round(values.reduce((total, value) => total + value, 0) / values.length)
    : 0;

  progressBar.style.width = `${percent}%`;
  loadText.textContent = `Loading model... ${percent}%`;
}

function markModelReady() {
  progressBar.style.width = "100%";
  loadText.textContent = "Model ready";
  window.setTimeout(() => loadStatus.classList.add("ready"), 500);
}

function appendMessage(className, text = "", imageUrl) {
  const node = document.createElement("div");
  node.className = `message ${className}`;

  if (imageUrl) {
    const image = document.createElement("img");
    image.src = imageUrl;
    image.alt = "";
    node.append(image);
  }

  const content = document.createElement("div");
  content.textContent = text;
  node.append(content);
  messages.append(node);
  messages.scrollTop = messages.scrollHeight;
  return content;
}

function setBusy(busy) {
  isGenerating = busy;
  sendButton.textContent = busy ? "Stop" : "Send";
  sendButton.disabled = false;
  promptInput.disabled = busy;
  imageButton.disabled = busy;
}

function updateComposer() {
  imageButton.classList.toggle("has-image", selectedImage !== null);
  imagePreview.hidden = selectedImage === null;
  if (selectedImage) {
    previewImage.src = selectedImage.url;
  } else {
    previewImage.removeAttribute("src");
  }
  sendButton.disabled = isGenerating ? false : !promptInput.value.trim() && !selectedImage;
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`;
}

async function loadModel() {
  if (processor && model) return;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    loadStatus.classList.remove("ready");
    updateProgress({ status: "start", progress: 0, file: "model" });
    processor = await AutoProcessor.from_pretrained(modelId, {
      progress_callback: updateProgress,
    });
    model = await Qwen3_5ForConditionalGeneration.from_pretrained(modelId, {
      dtype: {
        embed_tokens: "q4",
        vision_encoder: "fp16",
        decoder_model_merged: "q4",
      },
      device: "webgpu",
      progress_callback: updateProgress,
    });
    markModelReady();
  })();

  return loadingPromise;
}

async function generate(userText, attachment, output) {
  await loadModel();
  if (!processor || !model) throw new Error("Model was not loaded");

  const promptText = `${plainAnswerInstruction}\n\n${userText || "Briefly describe this image."}`;
  let prompt = "<|im_start|>user\n";

  if (attachment) {
    prompt += "<|vision_start|><|image_pad|><|vision_end|>";
  }

  prompt += `${promptText}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;

  const inputs = attachment ? await processor(prompt, attachment.raw) : await processor(prompt);
  let streamedText = "";

  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      streamedText += token;
      output.textContent = streamedText.replace(/<\|im_end\|>/g, "").trimStart();
      messages.scrollTop = messages.scrollHeight;
    },
  });

  const result = await model.generate({
    ...inputs,
    max_new_tokens: 512,
    do_sample: true,
    temperature: 0.7,
    top_p: 0.8,
    top_k: 20,
    presence_penalty: 1.5,
    streamer,
    stopping_criteria: stoppingCriteria,
  });

  if (!streamedText.trim()) {
    const inputLength = inputs.input_ids.dims.at(-1) ?? 0;
    const decoded = processor.batch_decode(result.slice(null, [inputLength, null]), {
      skip_special_tokens: true,
    });
    output.textContent = decoded[0]?.trim() ?? "";
  }
}

imageButton.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", async () => {
  const file = imageInput.files?.[0];
  if (!file) return;

  if (selectedImage) URL.revokeObjectURL(selectedImage.url);

  const url = URL.createObjectURL(file);
  const raw = await RawImage.read(url);
  selectedImage = { raw: await raw.resize(448, 448), url };
  imageInput.value = "";
  updateComposer();
});

promptInput.addEventListener("input", updateComposer);
promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (isGenerating) {
    stoppingCriteria.interrupt();
    return;
  }

  const text = promptInput.value.trim();
  if (!text && !selectedImage) return;

  const attachment = selectedImage;
  appendMessage("user", text, attachment?.url);
  promptInput.value = "";
  selectedImage = null;
  updateComposer();

  const output = appendMessage("ai", "");

  try {
    setBusy(true);
    await generate(text, attachment, output);
  } catch (error) {
    output.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    setBusy(false);
    stoppingCriteria.reset();
    updateComposer();
  }
});

updateComposer();
loadModel().catch((error) => {
  loadText.textContent = error instanceof Error ? error.message : String(error);
});

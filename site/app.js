// LLM_local playground — wires the UI to the in-browser GPT.
import { Tokenizer } from "./llm-tokenizer.js";
import { GPT } from "./llm-core.js";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const metaEl = $("modelMeta");

let tok = null, model = null, stopFlag = false;

async function init() {
  try {
    statusEl.className = "status load";
    statusEl.textContent = "loading tokenizer…";
    tok = await Tokenizer.load("tokenizer.json");
    statusEl.textContent = "loading weights…";
    model = await GPT.load("");
    const c = model.cfg;
    metaEl.textContent =
      `vocab ${c.vocab_size} · ${c.n_layer} layers · ${c.n_head} heads · ` +
      `${c.n_embd}d · ctx ${c.block_size} · ${(await (await fetch("model.bin")).arrayBuffer()).byteLength / 1e3 | 0} KB`;
    statusEl.className = "status ok";
    statusEl.textContent = "ready ✓";
    $("gen").disabled = false;
  } catch (e) {
    statusEl.className = "status err";
    statusEl.textContent = "load failed: " + e.message;
    console.error(e);
  }
}

let running = false;
async function generate() {
  if (!model || running) return;
  running = true; stopFlag = false;
  $("gen").disabled = true; $("stop").disabled = false;

  const prompt = $("prompt").value;
  const maxNew = +$("tokens").value;
  const temperature = +$("temp").value;
  const topK = +$("topk").value;
  const topP = +$("topp").value;

  $("out").textContent = prompt;
  const t0 = performance.now();
  // generate in chunks so the UI can update + stop
  let produced = 0;
  const chunk = 8;
  try {
    while (produced < maxNew && !stopFlag) {
      const text = model.generate(tok, prompt + $("out").textContent.slice(prompt.length),
        Math.min(chunk, maxNew - produced), { temperature, topK, topP, seed: 1 });
      // re-derive only the newly produced tail
      const full = text;
      $("out").textContent = full;
      produced = full.length - prompt.length;
      await new Promise((r) => setTimeout(r, 0));
    }
  } catch (e) {
    $("out").textContent += "\n[error] " + e.message;
  }
  const dt = ((performance.now() - t0) / 1000);
  $("timing").textContent = `${produced} chars in ${dt.toFixed(1)}s (${produced / Math.max(dt, 0.1) | 0} char/s on CPU)`;
  running = false; $("gen").disabled = false; $("stop").disabled = true;
}

$("gen").onclick = generate;
$("stop").onclick = () => { stopFlag = true; };
$("clear").onclick = () => { $("out").textContent = ""; $("timing").textContent = ""; };

// chat mode (uses the SFT-style instruction template)
$("chatIn").addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || !model || running) return;
  const user = $("chatIn").value.trim();
  if (!user) return;
  $("chatIn").value = "";
  const prompt = `User: ${user}\nAssistant:`;
  const out = model.generate(tok, prompt, 160, { temperature: 0.8, topK: 40, topP: 0.9, seed: 2 });
  const reply = out.slice(prompt.length).split("User:")[0].trim();
  $("chat").textContent += `\nYou: ${user}\nLLM_local: ${reply}\n`;
  $("chat").scrollTop = $("chat").scrollHeight;
});

init();

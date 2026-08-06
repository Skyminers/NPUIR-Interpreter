"use strict";

const byId = (id) => document.getElementById(id);
const ui = {
  showRun: byId("show-run"), showReplay: byId("show-replay"), loadSession: byId("load-session-button"),
  runView: byId("run-view"), replayView: byId("replay-view"), backendStatus: byId("backend-status"),
  runForm: byId("run-form"), kernelFile: byId("kernel-file"), kernelFileName: byId("kernel-file-name"), kernelSource: byId("kernel-source"),
  irStage: byId("ir-stage-status"), argumentFiles: byId("argument-files"), argumentFileList: byId("argument-file-list"),
  runArgs: byId("run-args"), runBlockDim: byId("run-block-dim"), runSubBlock: byId("run-sub-block"), runGmElems: byId("run-gm-elems"),
  runSched: byId("run-sched"), runSeed: byId("run-seed"), runDebugCore: byId("run-debug-core"), runArch: byId("run-arch"), runSubmit: byId("run-submit"),
  expectedValues: byId("expected-values"), expectedArg: byId("expected-arg"), expectedAtol: byId("expected-atol"), expectedRtol: byId("expected-rtol"),
  runResult: byId("run-result"), runResultTitle: byId("run-result-title"), runResultDetail: byId("run-result-detail"),
  loweringResult: byId("lowering-result"), loweringTitle: byId("lowering-title"), loweringDetail: byId("lowering-detail"),
  downloadLowered: byId("download-lowered"), loadLowered: byId("load-lowered"),
  outputResults: byId("output-results"), processOutput: byId("process-output"), replayRun: byId("replay-run"),
  file: byId("session-file"), empty: byId("empty-state"), debugger: byId("debugger"),
  timeline: byId("timeline"), current: byId("step-current"), total: byId("step-total"),
  first: byId("first"), previous: byId("previous"), play: byId("play"), next: byId("next"), last: byId("last"),
  schedule: byId("schedule"), program: byId("program-instance"), selectedCore: byId("selected-core"), result: byId("step-result"), finish: byId("finish-result"),
  sequence: byId("event-sequence"), executedCore: byId("executed-core"), executed: byId("executed-op"), cores: byId("cores"), pipes: byId("pipes"), pipeCore: byId("pipe-core"),
  aicIr: byId("aic-ir-source"), aivIr: byId("aiv-ir-source"),
  followAicIr: byId("follow-aic-ir"), followAivIr: byId("follow-aiv-ir"),
  waits: byId("waits"), sync: byId("sync"), arena: byId("arena"), offset: byId("memory-offset"), length: byId("memory-length"),
  allocations: byId("allocations"), typedValues: byId("typed-values"), memory: byId("memory")
};

let meta = null;
let states = [];
let finish = null;
let step = 0;
let selectedCoreId = null;
let selectedProgramId = 0;
let playing = null;
let memory = new Map();
let replayedThrough = -1;
let highlightedIrLines = {AIC: 0, AIV: 0};
let lastExecutedIrLines = {AIC: [], AIV: []};
let backendAvailable = false;
let latestSessionUrl = null;
let latestLoweredUrl = null;
let selectedArgumentFiles = [];

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#039;");

function parseSession(text) {
  const events = [];
  text.split(/\r?\n/).forEach((line, index) => {
    if (!line.trim()) return;
    try { events.push(JSON.parse(line)); }
    catch (error) { throw new Error(`第 ${index + 1} 行不是合法 JSON：${error.message}`); }
  });
  const sessionMeta = events.find((event) => event.event === "meta");
  const sessionStates = events.filter((event) => event.event === "state");
  if (!sessionMeta || sessionMeta.schema !== "npuir-interp-debug/v1") throw new Error("不支持的调试协议");
  if (sessionMeta.target_model !== "ascend-single-core/v2") {
    throw new Error("该会话由旧版调试器生成；请重新构建 npuir-interp-debug 后再运行用例");
  }
  if (!sessionStates.length) throw new Error("调试会话中没有 state 事件");
  return { meta: sessionMeta, states: sessionStates, finish: events.findLast((event) => event.event === "finish") };
}

function coreCoordinates(core) {
  if (Number.isInteger(core?.program_id) && core?.kind) {
    return { program: core.program_id, kind: core.kind, sub: core.sub_core_id ?? 0 };
  }
  const id = typeof core === "string" ? core : core?.id;
  const match = /^(AIC|AIV)#(\d+)(?:\.(\d+))?$/.exec(id ?? "");
  return match ? { program: Number(match[2]), kind: match[1], sub: Number(match[3] ?? 0) } : null;
}

function coreLabel(core) {
  const coordinates = coreCoordinates(core);
  if (!coordinates) return core?.id ?? "—";
  if (coordinates.kind === "AIC") return "AIC";
  return "AIV";
}

function visibleCores(state) {
  const cores = state.cores.filter((core) => {
    const coordinates = coreCoordinates(core);
    return coordinates?.program === selectedProgramId &&
      (coordinates.kind !== "AIV" || coordinates.sub === 0);
  });
  const inactive = (kind, sub = 0) => ({
    id: `inactive-${kind}-${sub}`, program_id: selectedProgramId, kind,
    sub_core_id: sub, status: "inactive", steps: 0, clock: "—",
    current: null, blocked: {what: "", operation: null}, flags: [],
    pipes: (kind === "AIC"
      ? ["PIPE_S", "PIPE_M", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_FIX"]
      : ["PIPE_S", "PIPE_V", "PIPE_MTE2", "PIPE_MTE3"])
      .map((pipe) => ({pipe, tasks: []})),
    inactive: true
  });
  if (!cores.some((core) => core.kind === "AIC")) cores.push(inactive("AIC"));
  if (!cores.some((core) => core.kind === "AIV")) cores.push(inactive("AIV"));
  return cores.sort((a, b) => a.kind.localeCompare(b.kind) || (a.sub_core_id ?? 0) - (b.sub_core_id ?? 0));
}

function isCoreLocalArena(arena) {
  return ["ub", "l1", "l0a", "l0b", "l0c", "cbuf", "ca", "cb", "cc"]
    .includes(String(arena.space).toLowerCase());
}

function visibleArenas(state) {
  return state.arenas.filter((arena) =>
    (!isCoreLocalArena(arena) || arena.owner === selectedProgramId) &&
    (!Number.isInteger(arena.sub_core_id) || arena.sub_core_id === 0));
}

function crossFlagIsVisible(flag) {
  if (Number.isInteger(flag.scope)) return flag.scope < 0 || flag.scope === selectedProgramId;
  if (String(flag.key).includes("inter-block")) return true;
  return String(flag.key).includes(`block=${selectedProgramId}]`);
}

function activateSession(text) {
  const parsed = parseSession(text);
  meta = parsed.meta; states = parsed.states; finish = parsed.finish;
  step = 0;
  const requested = states[0].cores.find((core) => core.id === meta.core_filter);
  selectedProgramId = coreCoordinates(requested ?? states[0].cores[0])?.program ?? 0;
  const requestedKind = coreCoordinates(requested)?.kind;
  const initialCores = visibleCores(states[0]);
  selectedCoreId = initialCores.find((core) => core.kind === requestedKind)?.id ?? initialCores[0]?.id ?? null;
  memory = new Map(); replayedThrough = -1;
  highlightedIrLines = {AIC: 0, AIV: 0};
  lastExecutedIrLines = {AIC: [], AIV: []};
  const latestIrLine = {AIC: 0, AIV: 0};
  states.forEach((state, index) => {
    const kind = coreCoordinates(state.selected_core)?.kind;
    if (kind && state.executed?.ir_line) latestIrLine[kind] = Number(state.executed.ir_line);
    lastExecutedIrLines.AIC[index] = latestIrLine.AIC;
    lastExecutedIrLines.AIV[index] = latestIrLine.AIV;
  });
  const sourceHtml = String(meta.ir ?? "").split("\n").map((line, index) => `
    <div class="ir-line" data-line="${index + 1}"><span class="ir-line-number">${index + 1}</span><code>${escapeHtml(line) || " "}</code></div>`).join("");
  ui.aicIr.innerHTML = sourceHtml;
  ui.aivIr.innerHTML = sourceHtml;
  ui.timeline.max = String(states.length - 1);
  ui.total.textContent = String(states.length - 1);
  ui.schedule.textContent = meta.schedule;
  ui.finish.textContent = resultLabel(finish?.result ?? "未完成");
  ui.finish.style.color = finish?.result === "success" ? "var(--green)" : "var(--red)";
  ui.empty.hidden = true; ui.debugger.hidden = false;
  render();
}

function selectMode(mode) {
  const replay = mode === "replay";
  ui.runView.hidden = replay;
  ui.replayView.hidden = !replay;
  ui.loadSession.hidden = !replay;
  ui.showRun.classList.toggle("active", !replay);
  ui.showReplay.classList.toggle("active", replay);
}

ui.showRun.addEventListener("click", () => selectMode("run"));
ui.showReplay.addEventListener("click", () => selectMode("replay"));

ui.kernelFile.addEventListener("change", async () => {
  const file = ui.kernelFile.files[0];
  if (!file) return;
  ui.kernelSource.value = await file.text();
  ui.kernelFileName.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KiB`;
  updateIrStage();
});

function detectIrStage(source) {
  if (!source.trim()) return "empty";
  if (["global_kernel", "tensor<", "tensor.", "bufferization.", "linalg.", "gpu.", "hfusion."].some((marker) => source.includes(marker))) return "ttadapter";
  if (["hacc.entry", "hivm.func_core_type", "hivm.module_core_type"].every((marker) => source.includes(marker))) return "hivm";
  return "unlowered";
}

function updateIrStage() {
  const stage = detectIrStage(ui.kernelSource.value);
  ui.irStage.className = `ir-stage-status ${stage}`;
  ui.irStage.textContent = ({
    empty: "等待 IR",
    hivm: "已完全下降 · 可直接执行",
    ttadapter: "检测到 TTAdapter · 运行时将自动下降",
    unlowered: "IR 尚未完全下降 · 运行时将自动下降"
  })[stage];
}

ui.kernelSource.addEventListener("input", updateIrStage);

ui.argumentFiles.addEventListener("change", () => {
  const files = Array.from(ui.argumentFiles.files ?? []);
  const specs = ui.runArgs.value.split(",").map((spec) => spec.trim()).filter(Boolean);
  const withoutOldUploads = specs.filter((spec) => !/^@input\d+$/.test(spec));
  if (!selectedArgumentFiles.length) withoutOldUploads.splice(0, Math.min(files.length, withoutOldUploads.length));
  selectedArgumentFiles = files;
  const tokens = files.map((_file, index) => `@input${index}`);
  ui.runArgs.value = [...tokens, ...withoutOldUploads].join(",");
  ui.argumentFileList.innerHTML = files.length
    ? files.map((file, index) => `<span><code>@input${index}</code>${escapeHtml(file.name)}<small>${(file.size / 1024).toFixed(1)} KiB</small></span>`).join("")
    : "<span>尚未选择参数文件</span>";
});

function fileBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] ?? "");
    reader.onerror = () => reject(reader.error ?? new Error(`无法读取 ${file.name}`));
    reader.readAsDataURL(file);
  });
}

function parseExpectedValues(text) {
  if (!text.trim()) return null;
  const values = text.trim().split(/[\s,]+/).filter(Boolean).map(Number);
  if (!values.length || values.some((value) => !Number.isFinite(value))) {
    throw new Error("数学期望值只能包含以逗号或空白分隔的有限数字");
  }
  return {
    arg: Number(ui.expectedArg.value), values,
    atol: Number(ui.expectedAtol.value), rtol: Number(ui.expectedRtol.value)
  };
}

function outputCard(output) {
  const shape = output.shape.length ? `[${output.shape.join(", ")}]` : "scalar";
  const preview = output.preview?.length
    ? output.preview.map((value, index) => `<span><small>${index}</small>${escapeHtml(value)}</span>`).join("")
    : `<p class="muted">${escapeHtml(output.preview_error ?? "无可预览数据")}</p>`;
  return `<article class="panel output-card">
    <div class="output-card-head">
      <div><p class="eyebrow">ARG ${output.arg}</p><h2>${escapeHtml(output.name)}</h2></div>
      <a href="${escapeHtml(output.url)}" download>下载 NPY</a>
    </div>
    <div class="output-meta"><span>dtype <strong>${escapeHtml(output.dtype)}</strong></span><span>shape <strong>${escapeHtml(shape)}</strong></span><span>元素 <strong>${output.count}</strong></span></div>
    <div class="value-preview">${preview}</div>
  </article>`;
}

function renderRunResult(result) {
  const passed = result.status === "passed";
  ui.runResult.hidden = false;
  ui.runResult.classList.toggle("passed", passed);
  ui.runResult.classList.toggle("failed", !passed);
  ui.runResultTitle.textContent = passed ? "运行验证通过" : "运行验证失败";
  const lowerTime = result.lowering?.performed ? ` · 自动下降 ${result.lowering.duration_ms} ms` : "";
  ui.runResultDetail.textContent = `${result.process.duration_ms} ms · 进程 ${result.process.status}${lowerTime} · ${result.verification.message}`;
  const lowering = result.lowering;
  ui.loweringResult.hidden = !lowering;
  if (lowering) {
    ui.loweringResult.className = `panel lowering-result ${lowering.status}`;
    ui.loweringTitle.textContent = lowering.performed
      ? (lowering.status === "passed" ? "自动下降完成" : "自动下降失败")
      : "无需自动下降";
    ui.loweringDetail.textContent = lowering.message;
    latestLoweredUrl = lowering.source_url;
    ui.downloadLowered.hidden = !latestLoweredUrl;
    ui.loadLowered.hidden = !latestLoweredUrl;
    if (latestLoweredUrl) ui.downloadLowered.href = latestLoweredUrl;
  }
  ui.outputResults.innerHTML = result.outputs.length
    ? result.outputs.map(outputCard).join("")
    : '<div class="panel no-output">没有生成输出 NPY</div>';
  const lowerCommand = lowering?.command?.length
    ? `$ ${lowering.command.map((part) => JSON.stringify(part)).join(" ")}\n${lowering.stdout}${lowering.stderr}\n\n`
    : "";
  const command = result.process.command.map((part) => JSON.stringify(part)).join(" ");
  const runLog = command ? `$ ${command}\n${result.process.stdout}${result.process.stderr}` : "Interpreter 未启动";
  ui.processOutput.textContent = `${lowerCommand}${runLog}`.trim();
  latestSessionUrl = result.session_url;
  ui.replayRun.disabled = !latestSessionUrl;
  ui.runResult.scrollIntoView({behavior: "smooth", block: "start"});
}

ui.loadLowered.addEventListener("click", async () => {
  if (!latestLoweredUrl) return;
  try {
    const response = await fetch(latestLoweredUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    ui.kernelSource.value = await response.text();
    ui.kernelFileName.textContent = "已载入本次自动下降结果";
    updateIrStage();
    ui.kernelSource.scrollIntoView({behavior: "smooth", block: "center"});
  } catch (error) {
    window.alert(`无法载入下降后 IR：${error.message}`);
  }
});

ui.runForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!backendAvailable) {
    window.alert("当前页面没有连接到 Interpreter Web 服务，请使用 tools/interpreter_web.py 启动");
    return;
  }
  let expected;
  try { expected = parseExpectedValues(ui.expectedValues.value); }
  catch (error) { window.alert(error.message); return; }
  let inputFiles;
  try {
    inputFiles = await Promise.all(selectedArgumentFiles.map(async (file) => ({
      name: file.name, data: await fileBase64(file)
    })));
  } catch (error) {
    window.alert(`参数文件读取失败：${error.message}`);
    return;
  }
  const payload = {
    source: ui.kernelSource.value,
    args: ui.runArgs.value,
    arch: ui.runArch.value,
    input_files: inputFiles,
    block_dim: Number(ui.runBlockDim.value),
    sub_block_num: Number(ui.runSubBlock.value),
    dyn_gm_elems: Number(ui.runGmElems.value),
    sched: ui.runSched.value,
    seed: Number(ui.runSeed.value),
    debug_core: ui.runDebugCore.value,
    expected
  };
  ui.runSubmit.disabled = true;
  ui.runSubmit.textContent = "正在运行…";
  try {
    const response = await fetch("/api/run", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error ?? `HTTP ${response.status}`);
    renderRunResult(result);
  } catch (error) {
    window.alert(`运行失败：${error.message}`);
  } finally {
    ui.runSubmit.disabled = false;
    ui.runSubmit.textContent = "运行并生成调试日志";
  }
});

async function loadSessionFromUrl(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  activateSession(await response.text());
  selectMode("replay");
}

ui.replayRun.addEventListener("click", async () => {
  if (!latestSessionUrl) return;
  try { await loadSessionFromUrl(latestSessionUrl); }
  catch (error) { window.alert(`无法加载本次调试日志：${error.message}`); }
});

async function discoverBackend() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const config = await response.json();
    backendAvailable = Boolean(config.available);
    ui.backendStatus.textContent = backendAvailable
      ? (config.compiler_available ? "Interpreter 与自动下降已就绪" : "Interpreter 已就绪 · 自动下降不可用")
      : "未找到 Interpreter 二进制";
    ui.backendStatus.className = `backend-status ${backendAvailable ? "ready" : "unavailable"}`;
    ui.runSubmit.disabled = !backendAvailable;
  } catch (_error) {
    backendAvailable = false;
    ui.backendStatus.textContent = "仅重放模式 · 请通过 interpreter_web.py 启动";
    ui.backendStatus.className = "backend-status unavailable";
    ui.runSubmit.disabled = true;
  }
}

discoverBackend();

ui.file.addEventListener("change", async () => {
  const file = ui.file.files[0];
  if (!file) return;
  try {
    activateSession(await file.text());
  } catch (error) {
    window.alert(error.message);
  }
});

async function autoloadSession() {
  const session = new URLSearchParams(window.location.search).get("session");
  if (!session) return;
  try {
    await loadSessionFromUrl(session);
  } catch (error) {
    window.alert(`无法加载调试会话 ${session}：${error.message}`);
  }
}

autoloadSession();

function setStep(next) {
  step = Math.max(0, Math.min(states.length - 1, Number(next)));
  render();
}

ui.timeline.addEventListener("input", () => setStep(ui.timeline.value));
ui.first.addEventListener("click", () => setStep(0));
ui.previous.addEventListener("click", () => setStep(step - 1));
ui.next.addEventListener("click", () => setStep(step + 1));
ui.last.addEventListener("click", () => setStep(states.length - 1));
ui.play.addEventListener("click", togglePlay);

function togglePlay() {
  if (playing) {
    clearInterval(playing); playing = null; ui.play.textContent = "播放"; return;
  }
  if (step === states.length - 1) setStep(0);
  ui.play.textContent = "暂停";
  playing = setInterval(() => {
    if (step >= states.length - 1) { togglePlay(); return; }
    setStep(step + 1);
  }, 450);
}

document.addEventListener("keydown", (event) => {
  if (ui.debugger.hidden || ["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
  if (event.key === "ArrowLeft") setStep(step - 1);
  if (event.key === "ArrowRight") setStep(step + 1);
  if (event.key === " ") { event.preventDefault(); togglePlay(); }
});

function bindingsHtml(title, bindings) {
  if (!bindings?.length) return `<div class="ssa-group"><strong>${title}</strong><span class="muted">无</span></div>`;
  return `<div class="ssa-group"><strong>${title}</strong>${bindings.map((binding) => `
    <div class="ssa-value ${binding.bound ? "" : "unset"}">
      <code>${escapeHtml(binding.ssa)}</code><span>=</span><code>${escapeHtml(binding.value)}</code><small>${escapeHtml(binding.type)}</small>
    </div>`).join("")}</div>`;
}

function opHtml(op, detailed = false) {
  if (!op) return '<span class="muted">—</span>';
  if (!detailed) return `${escapeHtml(op.name)}<span class="location">${escapeHtml(op.location)}</span>`;
  return `<pre class="operation-text">${escapeHtml(op.text ?? op.name)}</pre>
    <div class="ssa-bindings">${bindingsHtml("输入 SSA", op.inputs)}${bindingsHtml("输出 SSA", op.outputs)}</div>
    <span class="location">${escapeHtml(op.location)}</span>`;
}

function syncHtml(sync, prefix = "") {
  if (!sync) return "";
  if (sync.kind === "pipe_barrier") {
    return `<strong>${escapeHtml(prefix + "BARRIER")}</strong> ${escapeHtml(sync.pipe)}`;
  }
  const event = sync.dynamic_event ? "dynamic event" : `event ${sync.event_id}`;
  return `<strong>${escapeHtml(prefix + sync.kind.toUpperCase())}</strong> ${escapeHtml(sync.set_pipe)} → ${escapeHtml(sync.wait_pipe)} · ${escapeHtml(event)}`;
}

function statusLabel(status) {
  return ({
    runnable: "可执行", done: "已完成", inactive: "未实例化", failed: "失败",
    blocked_on_flag: "等待 Flag", blocked_on_barrier: "等待 Barrier", blocked_on_lock: "等待锁"
  })[status] ?? status;
}

function resultLabel(result) {
  return ({
    ready: "就绪", advance: "已推进", handled: "已处理", block: "已阻塞",
    complete: "已完成", done: "已完成", success: "成功", failed: "失败"
  })[result] ?? result;
}

function taskKindLabel(kind) {
  return ({token: "同步令牌", effect: "延迟任务"})[kind] ?? kind;
}

function render() {
  const state = states[step];
  const cores = visibleCores(state);
  if (!cores.some((core) => core.id === selectedCoreId)) selectedCoreId = cores[0]?.id ?? null;
  const selected = cores.find((core) => core.id === selectedCoreId);
  const scheduled = state.cores.find((core) => core.id === state.selected_core) ?? state.selected_core;
  ui.timeline.value = String(step); ui.current.textContent = String(step);
  ui.program.textContent = `${selectedProgramId}（共 ${meta.block_dim}）`;
  ui.selectedCore.textContent = coreLabel(selected);
  ui.result.textContent = resultLabel(state.result);
  ui.sequence.textContent = `seq ${state.sequence}`;
  ui.executedCore.textContent = state.executed ? `执行于 ${coreLabel(scheduled)}` : "尚未执行";
  ui.executed.innerHTML = state.executed ? opHtml(state.executed, true) : '<span class="muted">初始状态</span>';
  renderCores(state); renderPipes(state); renderSync(state); renderIr(state); replayMemory(step); renderMemory(state);
}

function renderIrPane(kind, state, force = false) {
  const source = kind === "AIC" ? ui.aicIr : ui.aivIr;
  const core = visibleCores(state).find((entry) => entry.kind === kind);
  const executedKind = coreCoordinates(state.selected_core)?.kind;
  const previousLine = highlightedIrLines[kind];
  const line = Number(executedKind === kind && state.executed
    ? state.executed.ir_line
    : core?.current?.ir_line ?? lastExecutedIrLines[kind][step] ?? 0);
  if (line !== previousLine) {
    if (previousLine) source.querySelector(`[data-line="${previousLine}"]`)?.classList.remove("current");
    highlightedIrLines[kind] = line;
  } else if (!force) return;
  const current = line ? source.querySelector(`[data-line="${line}"]`) : null;
  current?.classList.add("current");
  if (current) {
    const sourceBox = source.getBoundingClientRect();
    const lineBox = current.getBoundingClientRect();
    source.scrollTop += lineBox.top - sourceBox.top - (source.clientHeight - lineBox.height) / 2;
  }
}

function renderIr(state, force = false) {
  renderIrPane("AIC", state, force);
  renderIrPane("AIV", state, force);
}

ui.followAicIr.addEventListener("click", () => renderIrPane("AIC", states[step], true));
ui.followAivIr.addEventListener("click", () => renderIrPane("AIV", states[step], true));

function renderCores(state) {
  const cores = visibleCores(state);
  ui.cores.innerHTML = cores.map((core) => `
    <article class="core ${core.id === selectedCoreId ? "selected" : ""} ${core.inactive ? "inactive" : ""}" data-core="${escapeHtml(core.id)}" role="button" tabindex="0" aria-pressed="${core.id === selectedCoreId}">
      <div class="core-head"><strong>${escapeHtml(coreLabel(core))}</strong><span class="status ${escapeHtml(core.status)}">${escapeHtml(statusLabel(core.status))}</span></div>
      <div class="core-op">${core.inactive ? '<span class="muted">本 kernel 未实例化</span>' : `<span class="core-caption">下一条</span>${opHtml(core.current)}`}</div>
      <div class="core-meta">已执行 ${core.steps} 步 · 时钟 ${escapeHtml(core.clock)}</div>
    </article>`).join("");
  ui.cores.querySelectorAll(".core").forEach((node) => {
    const select = () => { selectedCoreId = node.dataset.core; render(); };
    node.addEventListener("click", select);
    node.addEventListener("keydown", (event) => {
      if (["Enter", " "].includes(event.key)) { event.preventDefault(); select(); }
    });
  });
}

function rangeSummary(task) {
  const format = (range) => `A${range.arena}[0x${range.begin.toString(16)},0x${range.end.toString(16)})`;
  const reads = task.reads?.map(format) ?? [];
  const writes = task.writes?.map(format) ?? [];
  return [...reads.map((x) => `R ${x}`), ...writes.map((x) => `W ${x}`)].join("<br>");
}

function renderPipes(state) {
  const cores = visibleCores(state);
  const core = cores.find((entry) => entry.id === selectedCoreId) ?? cores[0];
  ui.pipeCore.textContent = core?.inactive ? `${coreLabel(core)} · 未实例化` : `观察 ${coreLabel(core)}`;
  ui.pipes.innerHTML = (core?.pipes ?? []).map((pipe) => `
    <article class="pipe ${pipe.tasks.length ? "active" : ""}">
      <div class="pipe-head"><strong>${escapeHtml(pipe.pipe)}</strong><span>${pipe.tasks.length}</span></div>
      ${pipe.tasks.length ? pipe.tasks.map((task) => `
        <div class="task"><span class="task-kind">${escapeHtml(taskKindLabel(task.kind))}</span><br>
          ${task.kind === "token"
            ? syncHtml({kind: "set_flag", set_pipe: task.set_pipe, wait_pipe: task.wait_pipe, event_id: task.event_id, dynamic_event: false}, "待提交 ")
            : escapeHtml(task.operation?.text ?? task.operation?.name ?? "effect")}<br>
          <span class="muted">${rangeSummary(task)}</span>
        </div>`).join("") : '<div class="empty-pipe">空闲</div>'}
    </article>`).join("");
}

function renderSync(state) {
  const cores = visibleCores(state);
  const blocked = cores.filter((core) => core.status.startsWith("blocked"));
  ui.waits.innerHTML = blocked.length ? blocked.map((core) => `
    <div class="list-item"><strong>${escapeHtml(coreLabel(core))}</strong> · ${escapeHtml(statusLabel(core.status))}
      <small>${escapeHtml(core.blocked.what || "等待条件未满足")}</small>
      <small>${escapeHtml(core.blocked.operation?.location ?? "")}</small>
    </div>`).join("") : '<div class="empty-list">当前没有 core 等待同步。</div>';

  const flags = cores.flatMap((core) => core.flags.map((flag) => ({ core: coreLabel(core), ...flag })));
  const pendingTokens = cores.flatMap((core) => core.pipes.flatMap((pipe) =>
    pipe.tasks.filter((task) => task.kind === "token").map((task) => ({
      core: coreLabel(core), pipe: pipe.pipe, ...task
    }))));
  const crossFlags = state.cross_flags.filter(crossFlagIsVisible);
  const items = [
    ...(state.executed?.sync ? [`<div class="list-item current-sync">${syncHtml(state.executed.sync, "本步 ")}<small>${escapeHtml(state.executed.location)}</small></div>`] : []),
    ...pendingTokens.map((token) => `<div class="list-item"><strong>${escapeHtml(token.core)} 待提交 SET_FLAG</strong> ${escapeHtml(token.set_pipe)} → ${escapeHtml(token.wait_pipe)} · event ${token.event_id}<small>位于 ${escapeHtml(token.pipe)} 队列</small></div>`),
    ...flags.map((flag) => `<div class="list-item"><strong>${escapeHtml(flag.core)}</strong> ${escapeHtml(flag.set_pipe)} → ${escapeHtml(flag.wait_pipe)} · event ${flag.event_id}<small>信号计数 = ${flag.count}</small></div>`),
    ...crossFlags.map((flag) => `<div class="list-item"><strong>${flag.scope < 0 || String(flag.key).includes("inter-block") ? "启动级" : "AIC ↔ AIV"}</strong> ${escapeHtml(flag.key)}<small>AIC 代次 = ${flag.aic_generation}</small></div>`),
    ...state.barriers.map((barrier) => `<div class="list-item"><strong>Barrier</strong> ${escapeHtml(barrier.key)}<small>已到达 ${barrier.arrived.length} / ${barrier.expected} · 代次 ${barrier.generation}</small></div>`)
  ];
  ui.sync.innerHTML = items.length ? items.join("") : '<div class="empty-list">没有已发布的 flag 或活跃 barrier。</div>';
}

function replayMemory(target) {
  if (target < replayedThrough) { memory = new Map(); replayedThrough = -1; }
  for (let i = replayedThrough + 1; i <= target; i++) {
    for (const patch of states[i].memory_patches) {
      const raw = patch.bytes;
      const bytes = new Uint8Array(raw.length / 2);
      for (let p = 0; p < bytes.length; p++) bytes[p] = parseInt(raw.slice(p * 2, p * 2 + 2), 16);
      const required = patch.offset + bytes.length;
      let arena = memory.get(patch.arena) ?? new Uint8Array(0);
      if (arena.length < required) { const grown = new Uint8Array(required); grown.set(arena); arena = grown; memory.set(patch.arena, arena); }
      arena.set(bytes, patch.offset);
    }
  }
  replayedThrough = target;
}

function parseAddress(text) {
  const value = Number(text.trim());
  return Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0;
}

function float16(raw) {
  const sign = raw & 0x8000 ? -1 : 1;
  const exponent = (raw >> 10) & 0x1f;
  const fraction = raw & 0x3ff;
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function decodeElement(bytes, offset, type, width) {
  if (offset + width > bytes.length) return "?";
  const view = new DataView(bytes.buffer, bytes.byteOffset + offset, width);
  if (type === "f16") return String(float16(view.getUint16(0, true)));
  if (type === "bf16") {
    const temp = new ArrayBuffer(4); const words = new DataView(temp);
    words.setUint16(2, view.getUint16(0, true), true);
    return String(words.getFloat32(0, true));
  }
  if (type === "f32") return String(view.getFloat32(0, true));
  if (type === "f64") return String(view.getFloat64(0, true));
  const match = /^i(\d+)$/.exec(type);
  if (match) {
    const bits = Number(match[1]);
    if (bits === 1) return String(view.getUint8(0) & 1);
    if (bits <= 8) return String(view.getInt8(0));
    if (bits <= 16) return String(view.getInt16(0, true));
    if (bits <= 32) return String(view.getInt32(0, true));
    if (bits <= 64) return String(view.getBigInt64(0, true));
  }
  return "0x" + [...bytes.slice(offset, offset + width)].reverse().map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function renderMemory(state) {
  const previousArena = Number(ui.arena.value);
  const arenas = visibleArenas(state);
  ui.arena.innerHTML = arenas.map((arena) => {
    const lane = String(arena.space).toLowerCase() === "ub" && Number.isInteger(arena.sub_core_id)
      ? " · AIV" : "";
    return `<option value="${arena.id}">${escapeHtml(arena.space)}${isCoreLocalArena(arena) ? ` · program ${arena.owner}${lane}` : " · shared"} · A${arena.id}</option>`;
  }).join("");
  if (arenas.some((arena) => arena.id === previousArena)) ui.arena.value = String(previousArena);
  const arenaId = Number(ui.arena.value || arenas[0]?.id || 0);
  const arenaState = state.arenas.find((entry) => entry.id === arenaId);
  const bytes = memory.get(arenaId) ?? new Uint8Array(0);
  const changed = new Set();
  state.memory_patches.filter((patch) => patch.arena === arenaId).forEach((patch) => {
    for (let p = 0; p < patch.bytes.length / 2; p++) changed.add(patch.offset + p);
  });

  ui.allocations.innerHTML = (arenaState?.allocations ?? []).map((allocation) => `
    <button class="allocation ${allocation.live ? "" : "dead"}" data-offset="${allocation.begin}">
      ${escapeHtml(allocation.name)} · 0x${allocation.begin.toString(16)}–0x${allocation.end.toString(16)}
    </button>`).join("") || '<span class="empty-list">该 arena 尚无分配。</span>';
  ui.allocations.querySelectorAll(".allocation").forEach((node) => node.addEventListener("click", () => {
    ui.offset.value = `0x${Number(node.dataset.offset).toString(16)}`; renderMemory(state);
  }));

  const begin = Math.min(parseAddress(ui.offset.value), Math.max(0, bytes.length));
  const requested = Math.max(16, Math.min(1024, Number(ui.length.value) || 256));
  const end = Math.min(bytes.length, begin + requested);
  const allocation = (arenaState?.allocations ?? []).find((item) => begin >= item.begin && begin < item.end);
  if (allocation?.element_bytes) {
    const width = allocation.element_bytes;
    const first = Math.max(allocation.begin, begin - ((begin - allocation.begin) % width));
    const typed = [];
    for (let address = first; address + width <= Math.min(end, allocation.end) && typed.length < 64; address += width) {
      const isChanged = [...Array(width).keys()].some((lane) => changed.has(address + lane));
      const index = Math.floor((address - allocation.begin) / width);
      typed.push(`<span class="typed-value ${isChanged ? "changed" : ""}">[${index}] ${escapeHtml(decodeElement(bytes, address, allocation.element_type, width))}</span>`);
    }
    ui.typedValues.innerHTML = typed.join("");
  } else {
    ui.typedValues.innerHTML = '<span class="empty-list">当前地址没有可识别的元素类型。</span>';
  }
  if (begin >= end) { ui.memory.innerHTML = '<div class="empty-list">该地址尚未分配。</div>'; return; }
  const rows = [];
  for (let row = begin; row < end; row += 16) {
    const rowEnd = Math.min(row + 16, end);
    const cells = []; let ascii = "";
    for (let address = row; address < rowEnd; address++) {
      const byte = bytes[address];
      cells.push(`<span class="${changed.has(address) ? "changed" : ""}">${byte.toString(16).padStart(2, "0")}</span>`);
      ascii += byte >= 32 && byte < 127 ? String.fromCharCode(byte) : ".";
    }
    rows.push(`<div class="hex-row"><span class="hex-address">0x${row.toString(16).padStart(8, "0")}</span><span class="hex-bytes">${cells.join("")}</span><span class="ascii">${escapeHtml(ascii)}</span></div>`);
  }
  ui.memory.innerHTML = rows.join("");
}

ui.arena.addEventListener("change", () => renderMemory(states[step]));
ui.offset.addEventListener("input", () => renderMemory(states[step]));
ui.length.addEventListener("input", () => renderMemory(states[step]));

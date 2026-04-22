/**
 * Shared workflow execution + validation (ported from dna_workflow_ui_demo.html).
 * Expects global SPECS. Callers pass `state` { cards, chimeraManifest }.
 */
(function (global) {
  function getCard(state, cardId) {
    return (state.cards || []).find((c) => c.card_id === cardId) || null;
  }

  function validateAndResolve(card, state) {
    const spec = SPECS[card.job_type];
    const errors = [];
    const ctx = {};

    for (const inputSpec of spec.inputs) {
      const b = card.inputs[inputSpec.id] || { mode: "manual", value: "", source: null, chimeraIndex: null };
      if (b.mode === "manual") {
        const val = (b.value ?? "").toString().trim();
        if (inputSpec.required && !val) {
          errors.push({
            code: "MISSING_REQUIRED_INPUT",
            field: inputSpec.id,
            message: `Required input '${inputSpec.id}' is missing.`
          });
        } else if (val) {
          ctx[inputSpec.id] = val;
        }
      } else if (b.mode === "inherited") {
        const src = b.source || {};
        const srcCard = getCard(state, src.card_id);
        if (!srcCard) {
          errors.push({
            code: "INHERIT_SOURCE_NOT_FOUND",
            field: inputSpec.id,
            message: `Source card not found for '${inputSpec.id}'.`
          });
          continue;
        }
        if (srcCard.run_state !== "success") {
          errors.push({
            code: "INHERIT_SOURCE_NOT_READY",
            field: inputSpec.id,
            message: `Source card is not successful for '${inputSpec.id}'.`
          });
          continue;
        }
        const val = ((srcCard.outputs || {}).resolved || {})[src.output_id];
        if (!val) {
          errors.push({
            code: "INHERIT_SOURCE_NOT_FOUND",
            field: inputSpec.id,
            message: `Source output missing for '${inputSpec.id}'.`
          });
          continue;
        }
        const outType = (((srcCard.outputs || {}).types || {})[src.output_id]) || "";
        if (inputSpec.artifact_type && outType && inputSpec.artifact_type !== outType) {
          errors.push({
            code: "INHERIT_TYPE_MISMATCH",
            field: inputSpec.id,
            message: `Type mismatch for inherited '${inputSpec.id}'.`
          });
          continue;
        }
        ctx[inputSpec.id] = val;
      } else if (b.mode === "chimera") {
        const entries = (state.chimeraManifest && state.chimeraManifest.entries) || [];
        const idx = b.chimeraIndex;
        const ent = typeof idx === "number" ? entries[idx] : null;
        if (!state.chimeraManifest) {
          errors.push({
            code: "CHIMERA_MANIFEST_MISSING",
            field: inputSpec.id,
            message: `Load a ChimeraX manifest first (Advanced → Load manifest).`
          });
          continue;
        }
        if (!ent || !ent.path) {
          errors.push({
            code: "CHIMERA_ENTRY_INVALID",
            field: inputSpec.id,
            message: `Choose a manifest entry with a file path for '${inputSpec.id}'.`
          });
          continue;
        }
        if (inputSpec.artifact_type && ent.artifact_type && inputSpec.artifact_type !== ent.artifact_type) {
          errors.push({
            code: "INHERIT_TYPE_MISMATCH",
            field: inputSpec.id,
            message: `Type mismatch for ChimeraX input '${inputSpec.id}'.`
          });
          continue;
        }
        ctx[inputSpec.id] = ent.path;
      }
    }

    for (const p of spec.params) {
      if (p.ui_only) continue;
      let raw = card.params[p.id];
      if ((raw === null || raw === undefined || raw === "") && p.required && p.default !== null && p.default !== undefined) {
        raw = p.default;
        card.params[p.id] = raw;
      }
      if (p.required && (raw === null || raw === undefined || raw === "")) {
        errors.push({ code: "MISSING_REQUIRED_PARAM", field: p.id, message: `Required parameter '${p.id}' is missing.` });
        continue;
      }
      if (raw === null || raw === undefined || raw === "") continue;
      let v = raw;
      if (p.type === "float") {
        v = Number(raw);
        if (!Number.isFinite(v)) {
          errors.push({ code: "INVALID_PARAM_TYPE", field: p.id, message: `Invalid numeric value for '${p.id}'.` });
          continue;
        }
      }
      if (p.type === "int") {
        v = parseInt(raw, 10);
        if (!Number.isFinite(v)) {
          errors.push({ code: "INVALID_PARAM_TYPE", field: p.id, message: `Invalid integer value for '${p.id}'.` });
          continue;
        }
      }
      if (typeof v === "number") {
        if (p.min !== undefined && v < p.min) errors.push({ code: "PARAM_OUT_OF_RANGE", field: p.id, message: `'${p.id}' below min ${p.min}.` });
        if (p.max !== undefined && v > p.max) errors.push({ code: "PARAM_OUT_OF_RANGE", field: p.id, message: `'${p.id}' above max ${p.max}.` });
      }
      ctx[p.id] = v;
    }

    if (spec.output_passthrough && card.outputs && card.outputs.resolved) {
      for (const [outId, inId] of Object.entries(spec.output_passthrough)) {
        const src = ctx[inId];
        if (src) card.outputs.resolved[outId] = String(src);
      }
    }

    for (const o of spec.outputs) {
      ctx[o.id] = ((card.outputs || {}).resolved || {})[o.id] || o.default || "";
    }

    if (spec.arg_builders) {
      for (const [name, b] of Object.entries(spec.arg_builders)) {
        if (ctx[b.when_input_present]) {
          let t = b.value_template;
          for (const [k, v] of Object.entries(ctx)) t = t.replaceAll(`{${k}}`, String(v));
          ctx[name] = t;
        } else {
          ctx[name] = "";
        }
      }
    }

    if (spec.param_arg_builders) {
      for (const [name, b] of Object.entries(spec.param_arg_builders)) {
        const key = b.when_param_present;
        const pv = key ? ctx[key] : undefined;
        const present = pv !== undefined && pv !== null && String(pv) !== "";
        if (!present) {
          ctx[name] = "";
          continue;
        }
        let t = b.value_template;
        for (const [k, v] of Object.entries(ctx)) t = t.replaceAll(`{${k}}`, String(v));
        ctx[name] = t;
      }
    }

    let cmd = spec.command_template;
    for (const [k, v] of Object.entries(ctx)) cmd = cmd.replaceAll(`{${k}}`, String(v));
    const unresolved = cmd.match(/\{[^{}]+\}/g) || [];
    if (unresolved.length) {
      for (const tok of unresolved) errors.push({ code: "TEMPLATE_TOKEN_UNRESOLVED", field: tok, message: `Unresolved token ${tok}` });
    }
    cmd = cmd.split(/\s+/).filter(Boolean).join(" ");

    const ok = errors.length === 0;
    card.validation = { ok, errors, command: cmd };
    card.validation_state = ok ? "valid" : "invalid";
    if (card.run_state === "draft" && ok) card.run_state = "ready";
    if (card.run_state === "ready" && !ok) card.run_state = "draft";
    return card.validation;
  }

  function versionedPathForRun(pathValue, runNumber, artifactType) {
    const raw = String(pathValue || "").trim();
    if (!raw) return raw;
    if (runNumber <= 0) return raw;
    const noTrail = artifactType === "dir" ? raw.replace(/\/+$/, "") : raw;
    const slash = noTrail.lastIndexOf("/");
    const dir = slash >= 0 ? noTrail.slice(0, slash + 1) : "";
    const base = slash >= 0 ? noTrail.slice(slash + 1) : noTrail;
    if (artifactType === "dir") {
      return `${dir}${base}_${runNumber}`;
    }
    const dot = base.lastIndexOf(".");
    if (dot > 0) {
      return `${dir}${base.slice(0, dot)}_${runNumber}${base.slice(dot)}`;
    }
    return `${dir}${base}_${runNumber}`;
  }

  function stripTrailingRunSuffix(pathValue, artifactType) {
    const raw = String(pathValue || "").trim();
    if (!raw) return raw;
    if (artifactType === "dir") {
      return raw.replace(/(?:_\d+)+\/?$/, "");
    }
    const slash = raw.lastIndexOf("/");
    const dir = slash >= 0 ? raw.slice(0, slash + 1) : "";
    const base = slash >= 0 ? raw.slice(slash + 1) : raw;
    const dot = base.lastIndexOf(".");
    if (dot > 0) {
      const stem = base.slice(0, dot).replace(/(?:_\d+)+$/, "");
      return `${dir}${stem}${base.slice(dot)}`;
    }
    return `${dir}${base.replace(/(?:_\d+)+$/, "")}`;
  }

  function applyRunVersionToOutputs(card) {
    if (!card || !card.outputs || !card.outputs.resolved) return;
    const nextRun = (Number(card.run_counter) || 0) + 1;
    card.run_counter = nextRun;
    const resolved = card.outputs.resolved || {};
    const base = card.outputs.base || {};
    const types = (card.outputs && card.outputs.types) || {};
    const spec = SPECS[card.job_type];
    const skipVersion = new Set((spec && spec.outputs || []).filter((o) => o.skip_run_version).map((o) => o.id));
    if (!card.outputs.base || Object.keys(card.outputs.base).length === 0) {
      card.outputs.base = {};
      for (const [oid, val] of Object.entries(resolved)) {
        card.outputs.base[oid] = stripTrailingRunSuffix(val, types[oid] || "");
      }
    }
    for (const [oid, val] of Object.entries(card.outputs.base || {})) {
      if (!val) continue;
      if (skipVersion.has(oid)) {
        resolved[oid] = val;
        continue;
      }
      resolved[oid] = versionedPathForRun(val, nextRun, types[oid] || "");
    }
  }

  function createCardFromJobType(jobType, nextId) {
    const spec = SPECS[jobType];
    const cardId = `card_${nextId}`;
    const inputs = {};
    for (const i of spec.inputs) {
      inputs[i.id] = { mode: "manual", value: "", source: null, chimeraIndex: null };
    }
    const params = {};
    for (const p of spec.params) params[p.id] = p.default;
    const outputsResolved = {};
    const outputTypes = {};
    for (const o of spec.outputs) {
      outputsResolved[o.id] = o.default;
      outputTypes[o.id] = o.artifact_type || "";
    }
    return {
      card_id: cardId,
      job_type: jobType,
      run_state: "draft",
      validation_state: "invalid",
      inputs,
      params,
      outputs: { resolved: outputsResolved, types: outputTypes, base: { ...outputsResolved } },
      last_run: { status: "", log: "", command: "" },
      validation: { ok: false, errors: [], command: "" },
      run_counter: 0
    };
  }

  function pathMeasurePort(card) {
    const raw = card && card.params ? card.params.port : 8008;
    const port = parseInt(String(raw ?? "8008"), 10);
    return Number.isFinite(port) ? port : 8008;
  }

  async function checkPathMeasureForCard(card, onRender) {
    const api = document.getElementById("apiBase").value.replace(/\/$/, "");
    const port = pathMeasurePort(card);
    try {
      const resp = await fetch(`${api}/ui/pathmeasure-status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ host: "127.0.0.1", port })
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || resp.statusText);
      const status = data.running ? `running at ${data.url}` : "not running";
      card.last_run.log += `[${new Date().toLocaleTimeString()}] Status: ${status}\n`;
      if (!data.running) card.run_state = "draft";
      if (onRender) onRender();
    } catch (err) {
      card.last_run.log += `[${new Date().toLocaleTimeString()}] Status check failed: ${err.message || err}\n`;
      if (onRender) onRender();
    }
  }

  async function stopPathMeasureForCard(card, onRender) {
    const api = document.getElementById("apiBase").value.replace(/\/$/, "");
    const port = pathMeasurePort(card);
    try {
      const resp = await fetch(`${api}/ui/pathmeasure-stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ host: "127.0.0.1", port })
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.ok === false) {
        const msg = data.message || data.detail || resp.statusText;
        throw new Error(msg);
      }
      card.run_state = "draft";
      card.last_run.status = "stopped";
      card.last_run.log += `[${new Date().toLocaleTimeString()}] PathMeasure stopped.\n`;
      if (onRender) onRender();
    } catch (err) {
      card.last_run.log += `[${new Date().toLocaleTimeString()}] Stop failed: ${err.message || err}\n`;
      if (onRender) onRender();
      window.alert(`Could not stop PathMeasure: ${err.message || err}`);
    }
  }

  async function startPathMeasureForCard(card, onRender, openBrowser = true) {
    const api = document.getElementById("apiBase").value.replace(/\/$/, "");
    const cwd = document.getElementById("cwd").value.trim();
    const port = pathMeasurePort(card);
    if (!Number.isFinite(port) || port < 1 || port > 65535) {
      window.alert("PathMeasure port must be 1-65535.");
      return;
    }
    card.run_state = "running";
    card.last_run.status = "running";
    card.last_run.command = `cryomodel pathmeasure serve --host 127.0.0.1 --port ${port}`;
    card.last_run.log = `[${new Date().toLocaleTimeString()}] Starting PathMeasure on port ${port}...\n`;
    if (onRender) onRender();
    try {
      const resp = await fetch(`${api}/ui/pathmeasure-start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ host: "127.0.0.1", port, cwd: cwd || null, open_browser: openBrowser })
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.ok === false) {
        const msg = data.message || data.detail || resp.statusText;
        throw new Error(msg);
      }
      const shownUrl = data.ui_url || data.url;
      card.run_state = "success";
      card.last_run.status = "success";
      card.last_run.log += `PathMeasure running.\n${shownUrl ? `UI: ${shownUrl}\n` : ""}`;
      if (shownUrl) {
        try {
          window.open(shownUrl, "_blank", "noopener");
        } catch (_) {}
      }
    } catch (err) {
      card.run_state = "error";
      card.last_run.status = "error";
      card.last_run.log += `Error: ${err.message || err}\n`;
      window.alert(`Could not launch PathMeasure: ${err.message || err}`);
    } finally {
      if (onRender) onRender();
    }
  }

  async function runCard(cardId, state, onRender) {
    const card = getCard(state, cardId);
    if (!card) return;
    if (card.job_type === "pathmeasure_launcher") {
      await startPathMeasureForCard(card, onRender, true);
      return;
    }
    applyRunVersionToOutputs(card);
    const v = validateAndResolve(card, state);
    if (onRender) onRender();
    if (!v.ok) return;

    card.run_state = "running";
    card.last_run.status = "running";
    card.last_run.command = v.command;
    card.last_run.log = `[${new Date().toLocaleTimeString()}] Running:\n${v.command}\n`;
    if (onRender) onRender();

    const api = document.getElementById("apiBase").value.replace(/\/$/, "");
    const cwd = document.getElementById("cwd").value.trim();
    try {
      const startResp = await fetch(`${api}/ui/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ card_id: card.card_id, command: v.command, cwd })
      });
      const startData = await startResp.json();
      if (!startResp.ok) throw new Error(startData.detail || JSON.stringify(startData));
      const runId = startData.run_id;
      card.last_run.run_id = runId;

      let done = false;
      while (!done) {
        const [sResp, lResp] = await Promise.all([fetch(`${api}/ui/status/${runId}`), fetch(`${api}/ui/log/${runId}`)]);
        const sData = await sResp.json();
        const lData = await lResp.json();
        if (lResp.ok) card.last_run.log = lData.log || card.last_run.log;
        if (sResp.ok) {
          card.last_run.status = sData.status;
          if (sData.status === "success" || sData.status === "error") {
            done = true;
            card.run_state = sData.status;
          } else {
            card.run_state = "running";
          }
        }
        if (onRender) onRender();
        if (!done) await new Promise((r) => setTimeout(r, 1000));
      }
    } catch (err) {
      card.last_run.log += `\n[ui] backend unavailable, falling back to simulation:\n${String(err)}\n`;
      await new Promise((r) => setTimeout(r, 700));
      card.run_state = "success";
      card.last_run.status = "success";
      card.last_run.log += `[${new Date().toLocaleTimeString()}] Simulated success.\n`;
      if (onRender) onRender();
    }
  }

  async function runAllWorkspace(state, onRender) {
    for (const card of state.cards) {
      if (card.job_type === "pathmeasure_launcher") continue;
      await runCard(card.card_id, state, onRender);
      const c = getCard(state, card.card_id);
      if (c && c.run_state === "error") break;
    }
  }

  global.CryoWorkflowEngine = {
    getCard,
    validateAndResolve,
    applyRunVersionToOutputs,
    createCardFromJobType,
    runCard,
    runAllWorkspace,
    startPathMeasureForCard,
    checkPathMeasureForCard,
    stopPathMeasureForCard
  };
})(typeof globalThis !== "undefined" ? globalThis : this);

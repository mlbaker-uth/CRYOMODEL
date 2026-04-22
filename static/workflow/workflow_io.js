/**
 * Workflow JSON/YAML export, JSON import, ChimeraX open — ported from dna_workflow_ui_demo.html.
 * Depends on SPECS, JOB_TOOL_COMMAND, document fields cwd/apiBase/chimeraxApp.
 */
(function (global) {
  const CHIMERAX_VIEW_EXT = /\.(pdb|cif|mmcif|mrc|map|ccp4)$/i;
  const CHIMERAX_SKIP_OUTPUT_IDS = new Set(["out_dir", "scores_csv", "summary_json"]);

  function collectWorkflowMetadata() {
    const defaultName = "ui_exported_workflow";
    const defaultDesc = "Exported from CryoModel workflow UI";
    const name = (window.prompt("Workflow name:", defaultName) || "").trim();
    if (!name) return null;
    const description = (window.prompt("Workflow description:", defaultDesc) || "").trim() || defaultDesc;
    return { name, description };
  }

  function toWorkflowDoc(state, meta) {
    const m = meta || { name: "ui_exported_workflow", description: "Exported from CryoModel workflow UI" };
    const files = {};
    const steps = [];
    for (const card of state.cards) {
      if (card.job_type === "pathmeasure_launcher") continue;
      const spec = SPECS[card.job_type];
      if (!spec) continue;
      const tc = JOB_TOOL_COMMAND[card.job_type] || { tool: card.job_type, command: "" };
      const step = {
        name: card.card_id,
        tool: tc.tool,
        command: tc.command,
        inputs: {},
        outputs: { ...((card.outputs && card.outputs.resolved) || {}) },
        depends_on: []
      };
      const deps = new Set();

      for (const inputSpec of spec.inputs) {
        const b = card.inputs[inputSpec.id];
        if (!b) continue;
        if (b.mode === "manual") {
          const v = (b.value || "").trim();
          if (v) {
            step.inputs[inputSpec.id] = v;
          } else if (inputSpec.required) {
            const key = `${card.card_id}_${inputSpec.id}`;
            files[key] = "";
            step.inputs[inputSpec.id] = `\${${key}}`;
          }
        } else if (b.mode === "inherited" && b.source && b.source.card_id && b.source.output_id) {
          step.inputs[inputSpec.id] = `\${${b.source.card_id}.${b.source.output_id}}`;
          deps.add(b.source.card_id);
        } else if (b.mode === "chimera" && state.chimeraManifest && typeof b.chimeraIndex === "number") {
          const ent = state.chimeraManifest.entries && state.chimeraManifest.entries[b.chimeraIndex];
          if (ent && ent.path) step.inputs[inputSpec.id] = ent.path;
        }
      }

      for (const [k, v] of Object.entries(card.params || {})) {
        const paramSpec = (spec.params || []).find((pp) => pp.id === k);
        if (paramSpec && paramSpec.ui_only) continue;
        if (v !== null && v !== undefined && String(v).trim() !== "") step.inputs[k] = v;
      }
      step.depends_on = Array.from(deps);
      steps.push(step);
    }

    return {
      name: m.name,
      description: m.description,
      version: "1.0",
      variables: {},
      files,
      parameters: {},
      steps,
      ui_workspace: {
        cards: state.cards,
        nextId: state.nextId
      }
    };
  }

  function exportWorkflowJson(state) {
    const meta = collectWorkflowMetadata();
    if (!meta) return;
    const doc = toWorkflowDoc(state, meta);
    const text = JSON.stringify(doc, null, 2);
    const blob = new Blob([text], { type: "application/json" });
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const safeName = meta.name.replace(/[^a-zA-Z0-9._-]+/g, "_");
    a.download = `${safeName || "cryomodel_workflow"}_${ts}.json`;
    document.body.appendChild(a);
    a.click();
    URL.revokeObjectURL(a.href);
    a.remove();
  }

  async function exportWorkflowYaml(state) {
    const meta = collectWorkflowMetadata();
    if (!meta) return;
    const doc = toWorkflowDoc(state, meta);
    const api = document.getElementById("apiBase").value.replace(/\/$/, "");
    try {
      const resp = await fetch(`${api}/ui/workflow-export-yaml`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow: doc })
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const d = data.detail;
        const msg = typeof d === "string" ? d : JSON.stringify(data);
        throw new Error(msg || resp.statusText);
      }
      const text = data.yaml || "";
      const blob = new Blob([text], { type: "text/yaml" });
      const ts = new Date().toISOString().replace(/[:.]/g, "-");
      const safeName = meta.name.replace(/[^a-zA-Z0-9._-]+/g, "_");
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${safeName || "cryomodel_workflow"}_${ts}.yaml`;
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(a.href);
      a.remove();
    } catch (err) {
      window.alert(`Could not export YAML (is workflow API running on ${api}?): ${err.message || err}`);
    }
  }

  async function importWorkflowFileFromInput(ev, state) {
    const file = ev.target.files && ev.target.files[0];
    ev.target.value = "";
    if (!file) return false;
    const text = await file.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (err) {
      window.alert(`Could not parse workflow JSON: ${err.message || err}`);
      return false;
    }
    if (!data || !Array.isArray(data.steps)) {
      window.alert("Invalid workflow JSON: missing steps array.");
      return false;
    }

    if (data.ui_workspace && Array.isArray(data.ui_workspace.cards)) {
      state.cards = data.ui_workspace.cards;
      state.nextId = Number.isFinite(data.ui_workspace.nextId) ? data.ui_workspace.nextId : state.cards.length + 1;
      state.selectedCardId = state.cards.length ? state.cards[0].card_id : null;
      return true;
    }

    const reverseMap = {};
    for (const [jobType, tc] of Object.entries(JOB_TOOL_COMMAND)) reverseMap[`${tc.tool}:${tc.command}`] = jobType;
    const cards = [];
    let nid = 1;
    for (const step of data.steps) {
      const jt = reverseMap[`${step.tool}:${step.command}`];
      if (!jt || !SPECS[jt]) continue;
      const spec = SPECS[jt];
      const cardId = `card_${nid++}`;
      const inputs = {};
      for (const i of spec.inputs) {
        const val = (step.inputs && step.inputs[i.id]) != null ? String(step.inputs[i.id]) : "";
        inputs[i.id] = { mode: "manual", value: val, source: null, chimeraIndex: null };
      }
      const params = {};
      for (const p of spec.params) params[p.id] = (step.inputs && step.inputs[p.id]) ?? p.default;
      const outputsResolved = {};
      const outputTypes = {};
      for (const o of spec.outputs) {
        outputsResolved[o.id] = (step.outputs && step.outputs[o.id]) || o.default;
        outputTypes[o.id] = o.artifact_type || "";
      }
      cards.push({
        card_id: cardId,
        job_type: jt,
        run_state: "draft",
        validation_state: "invalid",
        inputs,
        params,
        outputs: { resolved: outputsResolved, types: outputTypes, base: { ...outputsResolved } },
        last_run: { status: "", log: "", command: "" },
        validation: { ok: false, errors: [], command: "" },
        run_counter: 0
      });
    }
    state.cards = cards;
    state.nextId = nid;
    state.selectedCardId = cards.length ? cards[0].card_id : null;
    return true;
  }

  function collectChimeraXPathStrings(state) {
    const paths = [];
    const seen = new Set();
    for (const c of state.cards) {
      const resolved = (c.outputs && c.outputs.resolved) || {};
      const types = (c.outputs && c.outputs.types) || {};
      for (const [oid, val] of Object.entries(resolved)) {
        if (!val || CHIMERAX_SKIP_OUTPUT_IDS.has(oid)) continue;
        const t = types[oid] || "";
        if (t === "dir" || t === "table.csv" || t === "json") continue;
        const s = String(val).trim();
        if (!s || !CHIMERAX_VIEW_EXT.test(s)) continue;
        if (seen.has(s)) continue;
        seen.add(s);
        paths.push(s);
      }
    }
    return paths;
  }

  async function openOutputsInChimeraX(state) {
    const api = document.getElementById("apiBase").value.replace(/\/$/, "");
    const cwd = document.getElementById("cwd").value.trim();
    const appName = document.getElementById("chimeraxApp").value.trim();
    const paths = collectChimeraXPathStrings(state);
    if (!paths.length) {
      window.alert(
        "No PDB/MRC-class outputs found on workspace cards. Add pipeline cards (outputs use .pdb / .mrc / .map / .ccp4 / .cif), or run the pipeline so files exist under CWD."
      );
      return;
    }
    try {
      const resp = await fetch(`${api}/ui/open-chimerax`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths, cwd: cwd || null, app_name: appName || null })
      });
      let data = {};
      try {
        data = await resp.json();
      } catch (_) {}
      if (!resp.ok) {
        const detail = data.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
              : JSON.stringify(data);
        throw new Error(msg || resp.statusText);
      }
      if (data.opened && data.opened.length) {
        console.info("[ChimeraX] opened:", data.opened);
      }
    } catch (err) {
      window.alert(`Could not open ChimeraX (is the workflow API running on ${api}?): ${err.message || err}`);
    }
  }

  global.CryoWorkflowIO = {
    toWorkflowDoc,
    collectWorkflowMetadata,
    exportWorkflowJson,
    exportWorkflowYaml,
    importWorkflowFileFromInput,
    openOutputsInChimeraX
  };
})(typeof globalThis !== "undefined" ? globalThis : this);

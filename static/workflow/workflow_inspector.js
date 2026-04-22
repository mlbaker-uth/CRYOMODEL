/**
 * Inspector: Parameters | Command | Run log (tab panels).
 * Focus restore on Parameters panel avoids losing caret on each keystroke.
 */
(function (global) {
  function escHtml(s) {
    return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;");
  }

  function escAttr(s) {
    return String(s ?? "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
  }

  function parseFastaRecords(text) {
    const recs = [];
    let curHeader = null;
    let curChunks = [];
    for (const line of String(text || "").split(/\r?\n/)) {
      const t = line.trim();
      if (!t) continue;
      if (t.startsWith(">")) {
        if (curHeader !== null) recs.push({ header: curHeader, raw: curChunks.join("") });
        curHeader = t.slice(1).trim().split(/\s+/)[0] || "?";
        curChunks = [];
      } else {
        curChunks.push(t.replace(/\s/g, ""));
      }
    }
    if (curHeader !== null) recs.push({ header: curHeader, raw: curChunks.join("") });
    return recs;
  }

  function normalizeParamOption(opt) {
    if (opt && typeof opt === "object" && "value" in opt) {
      const value = String(opt.value);
      const label = opt.label != null ? String(opt.label) : value;
      return { value, label };
    }
    const str = String(opt ?? "");
    return { value: str, label: str };
  }

  function buildParamSelectInnerHtml(p, val) {
    const selVal = String(val ?? "");
    if (p.option_groups && Array.isArray(p.option_groups)) {
      return p.option_groups
        .map((g) => {
          const gl = escAttr(g.label || "");
          const inner = (g.options || [])
            .map((opt) => {
              const { value, label } = normalizeParamOption(opt);
              const sel = selVal === value ? "selected" : "";
              return `<option value="${escAttr(value)}" ${sel}>${escHtml(label)}</option>`;
            })
            .join("");
          return `<optgroup label="${gl}">${inner}</optgroup>`;
        })
        .join("");
    }
    if (p.options && Array.isArray(p.options)) {
      return p.options
        .map((opt) => {
          const { value, label } = normalizeParamOption(opt);
          const sel = selVal === value ? "selected" : "";
          return `<option value="${escAttr(value)}" ${sel}>${escHtml(label)}</option>`;
        })
        .join("");
    }
    return "";
  }

  function paramUsesSelect(p) {
    return (p.options && Array.isArray(p.options)) || (p.option_groups && Array.isArray(p.option_groups));
  }

  function devStatusBadge(jobType) {
    const spec = SPECS[jobType];
    const raw = (spec && spec.dev_status) || "testing";
    const label = (typeof DEV_STATUS_LABELS !== "undefined" && DEV_STATUS_LABELS[raw]) || raw;
    return `<span class="badge dev-${raw}" title="Development status">${escHtml(label)}</span>`;
  }

  function availableInheritedSources(state, card, inputSpec) {
    const idx = state.cards.findIndex((c) => c.card_id === card.card_id);
    if (idx < 0) return [];
    const prior = state.cards.slice(0, idx);
    const out = [];
    for (const c of prior) {
      for (const [oid, otype] of Object.entries((c.outputs && c.outputs.types) || {})) {
        if (inputSpec.artifact_type && otype && inputSpec.artifact_type !== otype) continue;
        out.push({
          label: `${SPECS[c.job_type].display_name} (${c.card_id}) :: ${oid}`,
          source: { card_id: c.card_id, output_id: oid }
        });
      }
    }
    return out;
  }

  function availableChimeraSources(state, inputSpec) {
    const entries = (state.chimeraManifest && state.chimeraManifest.entries) || [];
    const out = [];
    entries.forEach((ent, index) => {
      if (!ent || !ent.path) return;
      if (inputSpec.artifact_type && ent.artifact_type && inputSpec.artifact_type !== ent.artifact_type) {
        return;
      }
      out.push({
        index,
        label: `${ent.label} [${ent.kind}]`,
        path: ent.path
      });
    });
    return out;
  }

  function fillPickSequenceSelect(state, cardId, records) {
    const sel = document.getElementById(`pick_seq_${cardId}`);
    if (!sel) return;
    const card = CryoWorkflowEngine.getCard(state, cardId);
    const cur = card && card.params ? Number(card.params.selected_row) : 0;
    if (!records.length) {
      sel.innerHTML = '<option value="">(no sequences loaded)</option>';
      return;
    }
    sel.innerHTML = records
      .map((r, i) => {
        const lab = `${i}: ${r.header}`.slice(0, 220);
        const selected = cur === i ? " selected" : "";
        return `<option value="${i}"${selected}>${escHtml(lab)}</option>`;
      })
      .join("");
    const pick = Number.isFinite(cur) && cur >= 0 && cur < records.length ? cur : 0;
    sel.value = String(pick);
  }

  function wireAlignmentSequencePick(state, cardId, api) {
    const card = CryoWorkflowEngine.getCard(state, cardId);
    if (card && card.pick_records && Array.isArray(card.pick_records)) {
      fillPickSequenceSelect(state, cardId, card.pick_records);
    }
    const fileEl = document.getElementById(`pick_file_${cardId}`);
    const pasteEl = document.getElementById(`pick_paste_${cardId}`);
    const seqSel = document.getElementById(`pick_seq_${cardId}`);
    const loadPaste = document.getElementById(`pick_load_paste_${cardId}`);
    if (fileEl) {
      fileEl.addEventListener("change", () => {
        const f = fileEl.files && fileEl.files[0];
        if (!f) return;
        const r = new FileReader();
        r.onload = () => {
          const recs = parseFastaRecords(r.result);
          const c = CryoWorkflowEngine.getCard(state, cardId);
          if (c) c.pick_records = recs;
          fillPickSequenceSelect(state, cardId, recs);
          if (recs.length) api.setParamValue(cardId, "selected_row", "0");
          CryoWorkflowEngine.validateAndResolve(CryoWorkflowEngine.getCard(state, cardId), state);
          api.renderAll();
        };
        r.readAsText(f);
      });
    }
    if (loadPaste && pasteEl) {
      loadPaste.addEventListener("click", () => {
        const recs = parseFastaRecords(pasteEl.value);
        const c = CryoWorkflowEngine.getCard(state, cardId);
        if (c) c.pick_records = recs;
        fillPickSequenceSelect(state, cardId, recs);
        if (recs.length) api.setParamValue(cardId, "selected_row", "0");
        CryoWorkflowEngine.validateAndResolve(CryoWorkflowEngine.getCard(state, cardId), state);
        api.renderAll();
      });
    }
    if (seqSel) {
      seqSel.addEventListener("change", () => {
        if (seqSel.value === "") return;
        api.setParamValue(cardId, "selected_row", seqSel.value);
        CryoWorkflowEngine.validateAndResolve(CryoWorkflowEngine.getCard(state, cardId), state);
        api.renderAll();
      });
    }
  }

  function captureFocus(container) {
    const ae = document.activeElement;
    if (!ae || !container || !container.contains(ae) || !ae.id) return null;
    const tag = ae.tagName;
    if (tag !== "INPUT" && tag !== "TEXTAREA" && tag !== "SELECT") return null;
    return {
      id: ae.id,
      start: typeof ae.selectionStart === "number" ? ae.selectionStart : null,
      end: typeof ae.selectionEnd === "number" ? ae.selectionEnd : null
    };
  }

  function restoreFocus(fr) {
    if (!fr) return;
    const el = document.getElementById(fr.id);
    if (!el || !el.matches("input, textarea, select")) return;
    el.focus();
    if (fr.start !== null && typeof el.setSelectionRange === "function") {
      const n = el.value.length;
      const a = Math.min(fr.start, n);
      const b = Math.min(fr.end ?? fr.start, n);
      el.setSelectionRange(a, b);
    }
  }

  async function copyToClipboard(text, btn) {
    const t = String(text ?? "");
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(t);
      } else {
        const ta = document.createElement("textarea");
        ta.value = t;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      if (btn) {
        const prev = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => {
          btn.textContent = prev;
        }, 1600);
      }
    } catch (_) {
      window.alert("Could not copy to clipboard.");
    }
  }

  function wireInspectorCopyButtons(commandText, logText) {
    const copyCmd = document.getElementById("inspectorCopyCmdBtn");
    if (copyCmd) copyCmd.addEventListener("click", () => copyToClipboard(commandText, copyCmd));
    const copyLog = document.getElementById("inspectorCopyLogBtn");
    if (copyLog) copyLog.addEventListener("click", () => copyToClipboard(logText, copyLog));
  }

  function scrollInspectorLogToEnd() {
    const el = document.getElementById("inspectorLogBlock");
    if (el) el.scrollTop = el.scrollHeight;
  }

  function renderInspectorPanels(paramsEl, commandEl, logEl, state, api) {
    const card = CryoWorkflowEngine.getCard(state, state.selectedCardId);
    if (!paramsEl || !commandEl || !logEl) return;
    if (!card) {
      paramsEl.innerHTML = "";
      commandEl.innerHTML = "";
      logEl.innerHTML = "";
      return;
    }

    const fr = captureFocus(paramsEl);
    const spec = SPECS[card.job_type];

    if (card.job_type === "pathmeasure_launcher") {
      const portVal = card.params.port ?? 8008;
      const logText = (card.last_run && card.last_run.log) || "";
      paramsEl.innerHTML = `
        <div class="inspector-title">${escHtml(spec.display_name)} <span class="helper-text">(${escHtml(card.card_id)})</span> ${devStatusBadge(card.job_type)}</div>
        <p class="inspector-subtitle">${escHtml(spec.description)}</p>
        <div class="field">
          <label>Port *</label>
          <input id="pm_port" class="input" value="${escAttr(String(portVal))}" />
        </div>
        <div class="inspector-actions" style="margin-top:10px">
          <button type="button" id="pm_start_btn" class="btn btn-primary">Start + Open PathMeasure</button>
          <button type="button" id="pm_stop_btn" class="btn">Stop PathMeasure</button>
          <button type="button" id="pm_status_btn" class="btn">Check Status</button>
        </div>`;
      commandEl.innerHTML = `<p class="helper-text">PathMeasure is launched from the Parameters tab; there is no separate shell command preview.</p>`;
      const pmStatus = (card.last_run && card.last_run.status) || "—";
      logEl.innerHTML = `<div class="inspector-tab-toolbar">
          <span class="helper-text">Status: <strong>${escHtml(pmStatus)}</strong></span>
          <button type="button" class="btn btn-small" id="inspectorCopyLogBtn">Copy log</button>
        </div>
        <div id="inspectorLogBlock" class="log-block inspector-log-scroll">${escHtml(logText)}</div>`;
      wireInspectorCopyButtons("", logText);

      document.getElementById("pm_port").addEventListener("input", (e) => api.setParamValue(card.card_id, "port", e.target.value));
      document.getElementById("pm_start_btn").addEventListener("click", () => api.startPathMeasure(card));
      document.getElementById("pm_stop_btn").addEventListener("click", () => api.stopPathMeasure(card));
      document.getElementById("pm_status_btn").addEventListener("click", () => api.checkPathMeasure(card));
      restoreFocus(fr);
      return;
    }

    const v = CryoWorkflowEngine.validateAndResolve(card, state);
    const errByField = {};
    for (const e of v.errors) {
      if (!errByField[e.field]) errByField[e.field] = [];
      errByField[e.field].push(e.message);
    }

    let htmlParams = `<div class="inspector-title">${escHtml(spec.display_name)} <span class="helper-text">(${escHtml(card.card_id)})</span> ${devStatusBadge(card.job_type)}</div>`;
    htmlParams += `<p class="inspector-subtitle">${escHtml(spec.description)}</p>`;

    htmlParams += `<div class="section-title">Inputs</div>`;
    for (const inp of spec.inputs) {
      const b = card.inputs[inp.id];
      const fieldErr = errByField[inp.id] || [];
      const sources = availableInheritedSources(state, card, inp);
      const chims = availableChimeraSources(state, inp);
      const hasManifest = !!state.chimeraManifest;
      const escOpt = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
      let body;
      if (b.mode === "manual") {
        body = `<input id="input_${inp.id}" class="input ${fieldErr.length ? "error" : ""}" value="${escAttr((b.value ?? "").toString())}" />`;
      } else if (b.mode === "inherited") {
        body = `<select id="src_${inp.id}" class="input ${fieldErr.length ? "error" : ""}">
                <option value="">Select source...</option>
                ${sources
                  .map((s) => {
                    const val = `${s.source.card_id}::${s.source.output_id}`;
                    const sel =
                      b.source && b.source.card_id === s.source.card_id && b.source.output_id === s.source.output_id
                        ? "selected"
                        : "";
                    return `<option value="${escAttr(val)}" ${sel}>${escOpt(s.label)}</option>`;
                  })
                  .join("")}
              </select>`;
      } else {
        body = `<select id="chim_${inp.id}" class="input ${fieldErr.length ? "error" : ""}">
                <option value="">Select from manifest...</option>
                ${chims
                  .map((s) => {
                    const sel = b.chimeraIndex === s.index ? "selected" : "";
                    return `<option value="${s.index}" ${sel}>${escOpt(s.label)} — ${escOpt(s.path)}</option>`;
                  })
                  .join("")}
              </select>`;
      }
      htmlParams += `<div class="field">
        <label>${escHtml(inp.label)}${inp.required ? " *" : ""}</label>
        <div class="field-row">
          <select id="mode_${inp.id}" class="input">
            <option value="manual" ${b.mode === "manual" ? "selected" : ""}>Manual</option>
            <option value="inherited" ${b.mode === "inherited" ? "selected" : ""}>Inherited</option>
            <option value="chimera" ${b.mode === "chimera" ? "selected" : ""} ${hasManifest ? "" : "disabled"} title="${hasManifest ? "" : "Load manifest first"}">ChimeraX manifest</option>
          </select>
          ${body}
        </div>
        ${fieldErr.map((m) => `<div class="errtext">${escHtml(m)}</div>`).join("")}
      </div>`;
    }

    if (card.job_type === "alignment_sequence_pick_run") {
      htmlParams += `<div class="section-title">List sequences (UI only)</div>
        <p class="helper-text" style="margin-bottom:8px"><strong>Run</strong> uses the MSA path from Inputs. Load the same file here (or paste FASTA) so names match row indices.</p>
        <div class="field">
          <label>Load from file</label>
          <input type="file" id="pick_file_${card.card_id}" accept=".fa,.fas,.fasta,.faa,.txt,.aln" />
        </div>
        <div class="field">
          <label>Paste FASTA</label>
          <textarea id="pick_paste_${card.card_id}" class="input" rows="5" placeholder=">header&#10;SEQUENCE"></textarea>
        </div>
        <div class="inspector-actions" style="margin-bottom:8px">
          <button type="button" id="pick_load_paste_${card.card_id}" class="btn">Load from paste</button>
        </div>
        <div class="field">
          <label>Pick sequence</label>
          <select id="pick_seq_${card.card_id}" class="input"><option value="">(Load file or paste, then Load from paste)</option></select>
        </div>`;
    }

    htmlParams += `<div class="section-title">Parameters</div>`;
    for (const p of spec.params) {
      const val = card.params[p.id] ?? "";
      const fieldErr = errByField[p.id] || [];
      let control;
      if (paramUsesSelect(p)) {
        const opts = buildParamSelectInnerHtml(p, val);
        control = `<select id="param_${p.id}" class="input ${fieldErr.length ? "error" : ""}">${opts}</select>`;
      } else if (p.type === "textarea") {
        control = `<textarea id="param_${p.id}" class="input" rows="5">${escHtml(String(val ?? ""))}</textarea>`;
      } else {
        control = `<input id="param_${p.id}" class="input ${fieldErr.length ? "error" : ""}" value="${escAttr(String(val))}" />`;
      }
      htmlParams += `<div class="field">
        <label>${escHtml(p.label)}${p.required ? " *" : ""}</label>
        ${control}
        ${fieldErr.map((m) => `<div class="errtext">${escHtml(m)}</div>`).join("")}
      </div>`;
    }

    htmlParams += `<div class="inspector-actions" style="margin-top:14px">
      <button type="button" id="validateBtn" class="btn">Validate</button>
      <button type="button" id="runBtn" class="btn btn-primary" ${v.ok ? "" : "disabled"}>${card.run_state === "running" ? "Running…" : "Run card"}</button>
    </div>`;

    const commandText = v.command || "(invalid)";
    const logText = card.last_run.log || "";
    const htmlCommand = `<div class="inspector-tab-toolbar">
        <span class="helper-text">Command preview</span>
        <button type="button" class="btn btn-small" id="inspectorCopyCmdBtn">Copy command</button>
      </div>
      <div class="code-block inspector-command-scroll" style="min-height:120px; white-space:pre-wrap">${escHtml(commandText)}</div>`;
    const htmlLog = `<div class="inspector-tab-toolbar">
        <span class="helper-text">Run log</span>
        <button type="button" class="btn btn-small" id="inspectorCopyLogBtn">Copy log</button>
      </div>
      <div id="inspectorLogBlock" class="log-block inspector-log-scroll" style="min-height:220px">${escHtml(logText)}</div>`;

    paramsEl.innerHTML = htmlParams;
    commandEl.innerHTML = htmlCommand;
    logEl.innerHTML = htmlLog;
    wireInspectorCopyButtons(commandText, logText);

    for (const inp of spec.inputs) {
      const modeSel = document.getElementById(`mode_${inp.id}`);
      modeSel.addEventListener("change", (e) => api.setInputMode(card.card_id, inp.id, e.target.value));
      const im = card.inputs[inp.id].mode;
      if (im === "manual") {
        document.getElementById(`input_${inp.id}`).addEventListener("input", (e) => api.setInputValue(card.card_id, inp.id, e.target.value));
      } else if (im === "inherited") {
        document.getElementById(`src_${inp.id}`).addEventListener("change", (e) => {
          const val = e.target.value;
          if (!val) return api.setInheritedSource(card.card_id, inp.id, "", "");
          const [cid, oid] = val.split("::");
          api.setInheritedSource(card.card_id, inp.id, cid, oid);
        });
      } else {
        document.getElementById(`chim_${inp.id}`).addEventListener("change", (e) => api.setChimeraEntry(card.card_id, inp.id, e.target.value));
      }
    }
    for (const p of spec.params) {
      const pel = document.getElementById(`param_${p.id}`);
      const ev = paramUsesSelect(p) ? "change" : "input";
      pel.addEventListener(ev, (e) => api.setParamValue(card.card_id, p.id, e.target.value));
    }
    if (card.job_type === "alignment_sequence_pick_run") {
      wireAlignmentSequencePick(state, card.card_id, api);
    }
    document.getElementById("validateBtn").addEventListener("click", () => {
      CryoWorkflowEngine.validateAndResolve(CryoWorkflowEngine.getCard(state, card.card_id), state);
      api.renderAll();
    });
    document.getElementById("runBtn").addEventListener("click", () => api.runCard(card.card_id));

    restoreFocus(fr);
  }

  global.CryoWorkflowInspector = { renderInspectorPanels, scrollInspectorLogToEnd };
})(typeof globalThis !== "undefined" ? globalThis : this);

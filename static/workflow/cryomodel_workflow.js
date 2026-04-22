/**
 * V2.2 workspace + V2.3 inspector: Parameters | Command | Run log, copy actions, tab keyboard nav.
 */
(function () {
  const TAG_CHIPS = ["All", "Production", "DNA", "Map", "Model", "Symmetry", "Experimental", "Utility", "Bridge", "Testing"];

  const state = {
    cards: [],
    nextId: 1,
    selectedCardId: null,
    chimeraManifest: null,
    search: "",
    activeTags: new Set(),
    runAllBusy: false,
    dragWorkspaceId: null,
    /** @type {"params"|"command"|"log"} */
    activeInspectorTab: "params",
    assistantLog: "",
    assistantBusy: false
  };

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function shortBlurb(desc) {
    if (!desc) return "";
    const plain = desc.replace(/\*\*/g, "").replace(/\s+/g, " ").trim();
    const words = plain.split(" ");
    if (words.length <= 5) return plain;
    return words.slice(0, 5).join(" ") + "…";
  }

  /** Stable hue 0–359 from job type (V2.4 placeholder icon color). */
  function cardIconHue(jobType) {
    let h = 216;
    const s = String(jobType || "");
    for (let i = 0; i < s.length; i++) h = (h * 33 + s.charCodeAt(i)) >>> 0;
    return h % 360;
  }

  /** One–two character label for placeholder tile (letters / digits only). */
  function cardIconInitials(displayName, jobType) {
    const raw = String(displayName || jobType || "?").trim();
    const words = raw.split(/\s+/).filter((w) => w.length > 0);
    const alnum = (x) => String(x).replace(/[^a-zA-Z0-9]/g, "");
    let letters = "";
    if (words.length >= 2) {
      const a = alnum(words[0]);
      const b = alnum(words[1]);
      letters = (a[0] || "") + (b[0] || "");
    } else if (words.length === 1) {
      letters = alnum(words[0]).slice(0, 2);
    }
    if (!letters) letters = alnum(jobType).slice(0, 2) || "?";
    return letters.toUpperCase().slice(0, 2);
  }

  /** 50×50 placeholder tile; replace with real assets later via SPECS if desired. */
  function cardTypeIconHtml(jobType, spec, extraClass) {
    const initials = cardIconInitials(spec.display_name, jobType);
    const h = cardIconHue(jobType);
    const h2 = (h + 24) % 360;
    const tip = esc(spec.display_name || jobType);
    const cls = "card-type-icon" + (extraClass ? ` ${extraClass}` : "");
    return `<div class="${cls}" style="background:linear-gradient(145deg,hsl(${h},58%,46%),hsl(${h2},52%,34%))" title="${tip}" aria-hidden="true"><span class="card-type-icon-inner">${esc(initials)}</span></div>`;
  }

  function inferTags(jobType, spec) {
    const tags = [];
    const ds = spec.dev_status || "testing";
    if (ds === "production") tags.push("Production");
    if (ds === "experimental") tags.push("Experimental");
    if (ds === "testing" || ds === "untested") tags.push("Testing");
    const blob = (JSON.stringify(spec.inputs || []) + JSON.stringify(spec.outputs || [])).toLowerCase();
    if (blob.includes("map.mrc")) tags.push("Map");
    if (blob.includes("model.structure")) tags.push("Model");
    if (/dna|dnabuild|basehunter|seqconservation/i.test(jobType)) tags.push("DNA");
    if (/symmetry_find_run|helical_find_run|helical_segment_run/i.test(jobType)) tags.push("Symmetry");
    if (/pathmeasure|alignment_sequence|fasta|bridge/i.test(jobType)) tags.push("Bridge");
    if (/train_ml|train_ensemble|extract_features/i.test(jobType)) tags.push("Experimental");
    if (tags.length === 0) tags.push("Utility");
    return [...new Set(tags)];
  }

  function tagClass(tag) {
    if (tag === "DNA") return "tag-dna";
    if (tag === "Production") return "tag-prod";
    if (tag === "Experimental") return "tag-exp";
    return "tag-util";
  }

  function matchesFilters(jobType, spec) {
    const q = state.search.trim().toLowerCase();
    if (q) {
      const hay = `${spec.display_name || ""} ${jobType} ${spec.description || ""}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    if (state.activeTags.size === 0) return true;
    const cardTags = inferTags(jobType, spec);
    for (const t of state.activeTags) {
      if (cardTags.includes(t)) return true;
    }
    return false;
  }

  function sortedJobTypes() {
    return Object.keys(SPECS).sort((a, b) => {
      const na = SPECS[a].display_name || a;
      const nb = SPECS[b].display_name || b;
      return na.localeCompare(nb);
    });
  }

  function stepClassForRun(rs) {
    if (rs === "running") return "step-number step-warning";
    if (rs === "success") return "step-number step-success";
    if (rs === "error") return "step-number step-error";
    if (rs === "ready") return "step-number step-ready";
    return "step-number step-pending";
  }

  function ioLine(card) {
    CryoWorkflowEngine.validateAndResolve(card, state);
    const spec = SPECS[card.job_type];
    const bits = [];
    for (const inp of spec.inputs.slice(0, 2)) {
      const b = card.inputs[inp.id];
      let s = "";
      if (!b || b.mode === "manual") s = ((b && b.value) || "").toString().trim().slice(0, 28) || "—";
      else if (b.mode === "inherited") s = b.source ? `${b.source.card_id}:${b.source.output_id}` : "—";
      else s = "ChimeraX";
      bits.push(`${inp.id}: ${s}`);
    }
    return bits.join(" · ") || "—";
  }

  function renderTagChips() {
    const row = document.getElementById("libraryTagRow");
    if (!row) return;
    row.innerHTML = "";
    for (const tag of TAG_CHIPS) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      if (tag === "All") b.classList.toggle("chip-active", state.activeTags.size === 0);
      else b.classList.toggle("chip-active", state.activeTags.has(tag));
      b.textContent = tag;
      b.addEventListener("click", () => {
        if (tag === "All") state.activeTags.clear();
        else if (state.activeTags.has(tag)) state.activeTags.delete(tag);
        else state.activeTags.add(tag);
        renderTagChips();
        renderLibrary();
        updateLibraryMeta();
      });
      row.appendChild(b);
    }
  }

  function renderLibrary() {
    const list = document.getElementById("libraryList");
    if (!list) return;
    list.innerHTML = "";
    let shown = 0;
    for (const jobType of sortedJobTypes()) {
      const spec = SPECS[jobType];
      if (!matchesFilters(jobType, spec)) continue;
      shown++;
      const tags = inferTags(jobType, spec);
      const item = document.createElement("div");
      item.className = "library-item";
      item.draggable = true;
      item.dataset.jobType = jobType;
      item.innerHTML = `
        ${cardTypeIconHtml(jobType, spec, "card-type-icon-library")}
        <div class="library-item-main">
          <div class="library-item-title">${esc(spec.display_name || jobType)}</div>
          <div class="library-item-desc">${esc(shortBlurb(spec.description || ""))}</div>
          <div class="tag-row" aria-label="Tags">${tags.map((t) => `<span class="tag ${tagClass(t)}">${esc(t)}</span>`).join("")}</div>
        </div>
        <button type="button" class="btn btn-small library-add-btn" data-job-type="${esc(jobType)}">Add</button>
      `;
      item.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("application/x-cryomodel-job", jobType);
        e.dataTransfer.effectAllowed = "copy";
      });
      item.querySelector(".library-add-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        addCard(jobType);
      });
      list.appendChild(item);
    }
    if (shown === 0) {
      list.innerHTML =
        '<div class="shell-empty" style="margin:12px"><strong>No matching cards</strong>Try another search or tag filter.</div>';
    }
  }

  function updateLibraryMeta() {
    const el = document.getElementById("libraryPanelMeta");
    if (!el) return;
    const total = Object.keys(SPECS).length;
    let shown = 0;
    for (const jobType of sortedJobTypes()) {
      if (matchesFilters(jobType, SPECS[jobType])) shown++;
    }
    el.textContent = shown === total ? `${total} tools` : `${shown} / ${total} tools`;
  }

  function addCard(jobType) {
    if (!SPECS[jobType]) return;
    const card = CryoWorkflowEngine.createCardFromJobType(jobType, state.nextId++);
    state.cards.push(card);
    state.selectedCardId = card.card_id;
    CryoWorkflowEngine.validateAndResolve(card, state);
    renderAll();
  }

  function removeCard(cardId) {
    state.cards = state.cards.filter((c) => c.card_id !== cardId);
    if (state.selectedCardId === cardId) {
      state.selectedCardId = state.cards.length ? state.cards[state.cards.length - 1].card_id : null;
    }
    renderAll();
  }

  function duplicateCard(cardId) {
    const c = CryoWorkflowEngine.getCard(state, cardId);
    if (!c) return;
    const fresh = CryoWorkflowEngine.createCardFromJobType(c.job_type, state.nextId++);
    fresh.inputs = JSON.parse(JSON.stringify(c.inputs));
    fresh.params = JSON.parse(JSON.stringify(c.params));
    Object.assign(fresh.outputs.resolved, JSON.parse(JSON.stringify(c.outputs.resolved)));
    fresh.outputs.types = JSON.parse(JSON.stringify(c.outputs.types || {}));
    if (c.outputs.base) fresh.outputs.base = JSON.parse(JSON.stringify(c.outputs.base));
    CryoWorkflowEngine.validateAndResolve(fresh, state);
    const idx = state.cards.findIndex((x) => x.card_id === cardId);
    state.cards.splice(idx + 1, 0, fresh);
    state.selectedCardId = fresh.card_id;
    renderAll();
  }

  function moveCard(cardId, delta) {
    const i = state.cards.findIndex((c) => c.card_id === cardId);
    if (i < 0) return;
    const j = i + delta;
    if (j < 0 || j >= state.cards.length) return;
    const t = state.cards[i];
    state.cards[i] = state.cards[j];
    state.cards[j] = t;
    renderAll();
  }

  function reorderCards(fromId, toId) {
    if (fromId === toId) return;
    const fromIdx = state.cards.findIndex((c) => c.card_id === fromId);
    const toIdx = state.cards.findIndex((c) => c.card_id === toId);
    if (fromIdx < 0 || toIdx < 0) return;
    const [moved] = state.cards.splice(fromIdx, 1);
    state.cards.splice(toIdx, 0, moved);
    renderAll();
  }

  function selectCard(cardId) {
    state.selectedCardId = cardId;
    renderAll();
  }

  function setInputMode(cardId, inputId, mode) {
    const card = CryoWorkflowEngine.getCard(state, cardId);
    if (!card) return;
    card.inputs[inputId].mode = mode;
    if (mode === "manual") {
      card.inputs[inputId].source = null;
      card.inputs[inputId].chimeraIndex = null;
    }
    if (mode === "inherited") card.inputs[inputId].chimeraIndex = null;
    if (mode === "chimera") card.inputs[inputId].source = null;
    CryoWorkflowEngine.validateAndResolve(card, state);
    renderAll();
  }

  function setChimeraEntry(cardId, inputId, indexStr) {
    const card = CryoWorkflowEngine.getCard(state, cardId);
    if (!card) return;
    const idx = parseInt(indexStr, 10);
    card.inputs[inputId].mode = "chimera";
    card.inputs[inputId].chimeraIndex = Number.isFinite(idx) ? idx : null;
    card.inputs[inputId].source = null;
    CryoWorkflowEngine.validateAndResolve(card, state);
    renderAll();
  }

  function setInheritedSource(cardId, inputId, cardSrcId, outputId) {
    const card = CryoWorkflowEngine.getCard(state, cardId);
    if (!card) return;
    card.inputs[inputId].mode = "inherited";
    if (!cardSrcId || !outputId) {
      card.inputs[inputId].source = null;
    } else {
      card.inputs[inputId].source = { card_id: cardSrcId, output_id: outputId };
    }
    card.inputs[inputId].chimeraIndex = null;
    CryoWorkflowEngine.validateAndResolve(card, state);
    renderAll();
  }

  function setInputValue(cardId, inputId, value) {
    const card = CryoWorkflowEngine.getCard(state, cardId);
    if (!card) return;
    card.inputs[inputId].value = value;
    CryoWorkflowEngine.validateAndResolve(card, state);
    renderAll();
  }

  function setParamValue(cardId, pid, value) {
    const card = CryoWorkflowEngine.getCard(state, cardId);
    if (!card) return;
    card.params[pid] = value;
    CryoWorkflowEngine.validateAndResolve(card, state);
    renderAll();
  }

  async function loadChimeraManifest() {
    const api = document.getElementById("apiBase").value.replace(/\/$/, "");
    let path = document.getElementById("chimeraManifestPath").value.trim();
    if (!path) path = "~/cryomodel_chimerax_manifest.json";
    const statusEl = document.getElementById("manifestStatus");
    if (statusEl) statusEl.textContent = "Loading…";
    try {
      const resp = await fetch(`${api}/ui/chimerax-manifest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path })
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const d = data.detail;
        const msg =
          typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg || JSON.stringify(x)).join("; ") : JSON.stringify(data);
        throw new Error(msg || resp.statusText);
      }
      state.chimeraManifest = data;
      const n = (data.entries && data.entries.length) || 0;
      if (statusEl) statusEl.textContent = `${n} entr${n === 1 ? "y" : "ies"}`;
    } catch (err) {
      state.chimeraManifest = null;
      if (statusEl) statusEl.textContent = "";
      window.alert(`Could not load manifest: ${err.message || err}`);
    }
    renderAll();
  }

  const api = {
    renderAll,
    setInputMode,
    setChimeraEntry,
    setInheritedSource,
    setInputValue,
    setParamValue,
    async runCard(cardId) {
      await CryoWorkflowEngine.runCard(cardId, state, renderAll);
      renderAll();
    },
    async startPathMeasure(card) {
      await CryoWorkflowEngine.startPathMeasureForCard(card, renderAll, true);
    },
    async stopPathMeasure(card) {
      await CryoWorkflowEngine.stopPathMeasureForCard(card, renderAll);
    },
    async checkPathMeasure(card) {
      await CryoWorkflowEngine.checkPathMeasureForCard(card, renderAll);
    }
  };

  function renderAll() {
    renderWorkspace();
    renderInspectorColumn();
    refreshAssistantPanel();
  }

  function refreshAssistantPanel() {
    const log = document.getElementById("assistantLog");
    const status = document.getElementById("assistantStatus");
    const btn = document.getElementById("assistantSendBtn");
    const modeEl = document.getElementById("assistantMode");
    if (log) log.textContent = state.assistantLog || "";
    if (status) status.textContent = state.assistantBusy ? "Running assistant…" : "";
    if (btn) btn.disabled = !!state.assistantBusy;
    if (modeEl) modeEl.disabled = !!state.assistantBusy;
  }

  async function runAssistant() {
    const apiEl = document.getElementById("apiBase");
    const promptEl = document.getElementById("assistantPrompt");
    const modeEl = document.getElementById("assistantMode");
    if (!apiEl || !promptEl || !modeEl) return;
    const api = apiEl.value.replace(/\/$/, "");
    const cwdEl = document.getElementById("cwd");
    const cwd = cwdEl ? cwdEl.value.trim() : "";
    const mode = modeEl.value;
    const prompt = promptEl.value.trim();
    if (!prompt) {
      window.alert("Enter an assistant request.");
      return;
    }
    state.assistantBusy = true;
    const ts = new Date().toLocaleTimeString();
    state.assistantLog += `[${ts}] mode=${mode}\n> ${prompt}\n`;
    refreshAssistantPanel();
    try {
      const resp = await fetch(`${api}/ui/assistant`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, prompt, cwd: cwd || null })
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const d = data.detail;
        const msg = typeof d === "string" ? d : JSON.stringify(data);
        throw new Error(msg || resp.statusText);
      }
      const out = (data.stdout || "").trim();
      const err = (data.stderr || "").trim();
      if (out) state.assistantLog += `${out}\n`;
      if (err) state.assistantLog += `[stderr]\n${err}\n`;
      state.assistantLog += `\n`;
    } catch (err) {
      state.assistantLog += `[assistant error] ${err.message || err}\n\n`;
    } finally {
      state.assistantBusy = false;
      refreshAssistantPanel();
      const logEl = document.getElementById("assistantLog");
      if (logEl) logEl.scrollTop = logEl.scrollHeight;
    }
  }

  function wireAssistant() {
    const send = document.getElementById("assistantSendBtn");
    const prompt = document.getElementById("assistantPrompt");
    if (send) send.addEventListener("click", () => runAssistant());
    if (prompt) {
      prompt.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          runAssistant();
        }
      });
    }
  }

  function renderWorkspace() {
    const list = document.getElementById("workspaceList");
    const empty = document.getElementById("workspaceEmpty");
    const meta = document.getElementById("workspacePanelMeta");
    if (!list || !empty) return;

    if (state.cards.length === 0) {
      list.innerHTML = "";
      empty.style.display = "block";
      if (meta) meta.textContent = "No steps yet";
      return;
    }
    empty.style.display = "none";
    if (meta) meta.textContent = `${state.cards.length} step${state.cards.length === 1 ? "" : "s"}`;

    list.innerHTML = "";
    state.cards.forEach((c, idx) => {
      CryoWorkflowEngine.validateAndResolve(c, state);
      const spec = SPECS[c.job_type];
      const row = document.createElement("div");
      row.className = "workspace-card v2-workspace-row" + (state.selectedCardId === c.card_id ? " selected" : "");
      row.dataset.cardId = c.card_id;
      row.draggable = true;
      row.innerHTML = `
        <div class="workspace-card-head">
          <div class="workspace-card-head-left">
            <div class="${stepClassForRun(c.run_state)}" aria-hidden="true">${idx + 1}</div>
            ${cardTypeIconHtml(c.job_type, spec, "card-type-icon-workspace")}
            <div>
              <div class="workspace-card-title">${esc(spec.display_name || c.job_type)}</div>
              <div class="workspace-card-subtitle">${esc(shortBlurb(spec.description || ""))}</div>
              <div class="helper-text workspace-io-line">${esc(ioLine(c))}</div>
            </div>
          </div>
          <div class="workspace-actions-column">
            <span class="status ${statusClass(c)}" title="Run state">${esc(shortRunLabel(c))}</span>
            <div class="workspace-actions-row">
              <button type="button" class="btn btn-small workspace-run-btn" data-card-id="${esc(c.card_id)}" ${
                c.run_state === "running" || state.runAllBusy ? "disabled" : ""
              }>Run</button>
              <button type="button" class="btn btn-small" data-act="up" data-card-id="${esc(c.card_id)}" ${idx === 0 ? "disabled" : ""}>↑</button>
              <button type="button" class="btn btn-small" data-act="down" data-card-id="${esc(c.card_id)}" ${
                idx >= state.cards.length - 1 ? "disabled" : ""
              }>↓</button>
              <button type="button" class="btn btn-small" data-act="copy" data-card-id="${esc(c.card_id)}" title="Copy this step">Copy</button>
              <button type="button" class="btn btn-small btn-danger-soft workspace-remove-btn" data-card-id="${esc(c.card_id)}">Remove</button>
            </div>
          </div>
        </div>
      `;

      row.addEventListener("click", (e) => {
        if (e.target.closest("button")) return;
        selectCard(c.card_id);
      });

      row.addEventListener("dragstart", (e) => {
        state.dragWorkspaceId = c.card_id;
        e.dataTransfer.setData("text/plain", c.card_id);
        e.dataTransfer.effectAllowed = "move";
      });
      row.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
      });
      row.addEventListener("drop", (e) => {
        e.preventDefault();
        const from = state.dragWorkspaceId;
        if (from && from !== c.card_id) reorderCards(from, c.card_id);
        state.dragWorkspaceId = null;
      });

      row.querySelector(".workspace-run-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        api.runCard(c.card_id);
      });
      row.querySelector('[data-act="up"]').addEventListener("click", (e) => {
        e.stopPropagation();
        moveCard(c.card_id, -1);
      });
      row.querySelector('[data-act="down"]').addEventListener("click", (e) => {
        e.stopPropagation();
        moveCard(c.card_id, 1);
      });
      row.querySelector('[data-act="copy"]').addEventListener("click", (e) => {
        e.stopPropagation();
        duplicateCard(c.card_id);
      });
      row.querySelector(".workspace-remove-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        removeCard(c.card_id);
      });

      list.appendChild(row);
    });
    const rab = document.getElementById("runAllBtn");
    if (rab) rab.disabled = state.runAllBusy || state.cards.length === 0;
    const cwb = document.getElementById("clearWorkspaceBtn");
    if (cwb) cwb.disabled = state.cards.length === 0;
  }

  function shortRunLabel(c) {
    const rs = c.run_state;
    if (rs === "ready") return "Ready";
    if (rs === "draft") return c.validation_state === "valid" ? "Valid" : "Needs input";
    if (rs === "running") return "Running";
    if (rs === "success") return "OK";
    if (rs === "error") return "Error";
    return rs || "—";
  }

  function statusClass(c) {
    const rs = c.run_state;
    if (rs === "success") return "status-success";
    if (rs === "running") return "status-warning";
    if (rs === "error") return "status-error";
    if (rs === "ready") return "status-ready";
    return "status-draft";
  }

  function syncInspectorTabs() {
    const name = state.activeInspectorTab || "params";
    document.querySelectorAll("#inspectorTabs .tab").forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
    });
    document.querySelectorAll("#inspectorShell .inspector-tab-panel").forEach((p) => {
      p.classList.toggle("is-active", p.dataset.tab === name);
    });
    if (name === "log" && typeof CryoWorkflowInspector !== "undefined" && CryoWorkflowInspector.scrollInspectorLogToEnd) {
      requestAnimationFrame(() => CryoWorkflowInspector.scrollInspectorLogToEnd());
    }
  }

  function wireInspectorTabs() {
    const tablist = document.getElementById("inspectorTabs");
    const tabs = document.querySelectorAll("#inspectorTabs .tab");
    const order = ["params", "command", "log"];
    const btnByTab = {
      params: "inspectorTabBtnParams",
      command: "inspectorTabBtnCommand",
      log: "inspectorTabBtnLog"
    };
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        state.activeInspectorTab = tab.dataset.tab || "params";
        syncInspectorTabs();
      });
    });
    if (tablist) {
      tablist.addEventListener("keydown", (e) => {
        const target = e.target;
        if (!target || target.getAttribute("role") !== "tab") return;
        const cur = order.indexOf(target.dataset.tab || "");
        if (cur < 0) return;
        let next = cur;
        if (e.key === "ArrowRight" || e.key === "ArrowDown") {
          e.preventDefault();
          next = (cur + 1) % order.length;
        } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
          e.preventDefault();
          next = (cur - 1 + order.length) % order.length;
        } else if (e.key === "Home") {
          e.preventDefault();
          next = 0;
        } else if (e.key === "End") {
          e.preventDefault();
          next = order.length - 1;
        } else {
          return;
        }
        state.activeInspectorTab = order[next];
        syncInspectorTabs();
        const id = btnByTab[order[next]];
        const btn = id ? document.getElementById(id) : null;
        if (btn) btn.focus();
      });
    }
  }

  function renderInspectorColumn() {
    const empty = document.getElementById("inspectorEmpty");
    const shell = document.getElementById("inspectorShell");
    const paramsEl = document.getElementById("inspectorPanelParams");
    const commandEl = document.getElementById("inspectorPanelCommand");
    const logEl = document.getElementById("inspectorPanelLog");
    if (!empty || !shell || !paramsEl || !commandEl || !logEl) return;
    const card = CryoWorkflowEngine.getCard(state, state.selectedCardId);
    if (!card) {
      empty.style.display = "block";
      shell.style.display = "none";
      paramsEl.innerHTML = "";
      commandEl.innerHTML = "";
      logEl.innerHTML = "";
      return;
    }
    empty.style.display = "none";
    shell.style.display = "block";
    CryoWorkflowInspector.renderInspectorPanels(paramsEl, commandEl, logEl, state, api);
    syncInspectorTabs();
  }

  function clearWorkspace() {
    if (!state.cards.length) return;
    if (!window.confirm("Remove all steps from the workspace?")) return;
    state.cards = [];
    state.selectedCardId = null;
    renderAll();
  }

  function wireSearch() {
    const inp = document.getElementById("librarySearch");
    if (!inp) return;
    inp.addEventListener("input", () => {
      state.search = inp.value;
      renderLibrary();
      updateLibraryMeta();
    });
  }

  function wireWorkspaceDrop() {
    const zone = document.getElementById("workspaceDropZone");
    if (!zone) return;
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      if (e.dataTransfer.types.includes("application/x-cryomodel-job")) e.dataTransfer.dropEffect = "copy";
    });
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      const jt = e.dataTransfer.getData("application/x-cryomodel-job");
      if (jt && SPECS[jt]) addCard(jt);
    });
  }

  function wireRunAll() {
    const btn = document.getElementById("runAllBtn");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      if (state.runAllBusy) return;
      state.runAllBusy = true;
      btn.disabled = true;
      try {
        await CryoWorkflowEngine.runAllWorkspace(state, renderAll);
      } finally {
        state.runAllBusy = false;
        btn.disabled = false;
        renderAll();
      }
    });
  }

  function wireManifest() {
    const b = document.getElementById("loadManifestBtn");
    if (b) b.addEventListener("click", () => loadChimeraManifest());
  }

  function wireClearWorkspace() {
    const b = document.getElementById("clearWorkspaceBtn");
    if (b) b.addEventListener("click", () => clearWorkspace());
  }

  function wireWorkflowIo() {
    if (typeof CryoWorkflowIO === "undefined") {
      console.warn("cryomodel_workflow: workflow_io.js not loaded; import/export/ChimeraX disabled");
      return;
    }
    const impBtn = document.getElementById("importWorkflowBtn");
    const impFile = document.getElementById("importWorkflowFile");
    const expBtn = document.getElementById("exportWorkflowBtn");
    const chimeraBtn = document.getElementById("chimeraxBtn");
    if (impBtn && impFile) {
      impBtn.addEventListener("click", () => impFile.click());
      impFile.addEventListener("change", async (e) => {
        const ok = await CryoWorkflowIO.importWorkflowFileFromInput(e, state);
        if (ok) renderAll();
      });
    }
    if (expBtn) expBtn.addEventListener("click", () => CryoWorkflowIO.exportWorkflowJson(state));
    if (chimeraBtn) chimeraBtn.addEventListener("click", () => CryoWorkflowIO.openOutputsInChimeraX(state));
  }

  function init() {
    if (typeof SPECS === "undefined" || typeof CryoWorkflowEngine === "undefined") {
      console.error("cryomodel_workflow: missing workflow_specs.js or workflow_engine.js");
      return;
    }
    if (typeof CryoWorkflowInspector === "undefined") {
      console.error("cryomodel_workflow: missing workflow_inspector.js");
      return;
    }
    if (typeof JOB_TOOL_COMMAND === "undefined") {
      console.error("cryomodel_workflow: missing workflow_job_command.js");
      return;
    }
    wireSearch();
    wireWorkspaceDrop();
    wireRunAll();
    wireManifest();
    wireClearWorkspace();
    wireWorkflowIo();
    wireAssistant();
    wireInspectorTabs();
    renderTagChips();
    renderLibrary();
    updateLibraryMeta();
    renderAll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

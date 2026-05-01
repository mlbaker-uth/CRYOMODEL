(function () {
  const q = new URLSearchParams(window.location.search);
  const API = q.get("api") || "http://127.0.0.1:8011";
  const DEFAULT_PROJECT_ROOT = q.get("default_project_root") || "";
  const DEFAULT_API_HOST = q.get("default_api_host") || "127.0.0.1";
  const DEFAULT_API_PORT = Number(q.get("default_api_port") || 8010);
  const DEFAULT_API_PORT_TEXT = String(Number.isFinite(DEFAULT_API_PORT) ? DEFAULT_API_PORT : 8010);
  const HOME_DIR = q.get("home_dir") || "~";
  const THEME_KEY = "cryomodel.ui.theme";
  let projects = [];
  let selected = null;
  let lastKnownLastProject = null;

  const el = (id) => document.getElementById(id);
  const statusEl = el("status");

  function setStatus(msg, isErr) {
    statusEl.textContent = msg || "";
    statusEl.style.color = isErr ? "#8a1f1f" : "#1f315d";
  }

  function sortProjects(list) {
    const copy = (list || []).slice();
    copy.sort((a, b) => {
      const ta = [String(a.last_opened || ""), String(a.updated_at || "")].join("\0");
      const tb = [String(b.last_opened || ""), String(b.updated_at || "")].join("\0");
      return tb.localeCompare(ta);
    });
    return copy;
  }

  function firstNonBlank(...values) {
    for (const v of values) {
      if (v === null || v === undefined) continue;
      const s = String(v).trim();
      if (s) return s;
    }
    return "";
  }

  function parsePortOrDefault(raw) {
    const n = Number.parseInt(String(raw || "").trim(), 10);
    return Number.isFinite(n) && n > 0 ? n : DEFAULT_API_PORT;
  }

  function preferredTheme() {
    const saved = (localStorage.getItem(THEME_KEY) || "").trim();
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function applyTheme(theme) {
    const t = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem(THEME_KEY, t);
    const b = el("themeToggleBtn");
    if (b) b.textContent = t === "dark" ? "☀ Light mode" : "🌙 Dark mode";
  }

  function applyForm(p) {
    selected = p || null;
    el("name").value = p?.name || "";
    el("projectRoot").value = p?.project_root || DEFAULT_PROJECT_ROOT;
    el("apiHost").value = firstNonBlank(p?.api_host, DEFAULT_API_HOST);
    el("apiPort").value = firstNonBlank(p?.api_port, DEFAULT_API_PORT_TEXT);
    el("chimeraxApp").value = p?.chimerax_app || "ChimeraX";
    el("manifestPath").value = p?.manifest_path || "";
    el("description").value = p?.description || "";
    el("autoLoadLast").checked = p?.auto_load_last ?? true;
    el("startServer").checked = p?.start_server_on_launch ?? true;
    el("manifestSameAsProject").checked = false;
    renderProjectRootHint(Boolean(p));
    syncManifestFromProjectIfNeeded();
    updateManifestControls();
  }

  function renderProjectRootHint(hasSelectedProject) {
    const hint = el("projectRootHint");
    if (hasSelectedProject) {
      hint.innerHTML = '<span class="hint-chip">Loaded from saved project entry</span>';
      return;
    }
    hint.innerHTML = `<span class="hint-chip">Default from launch context: ${DEFAULT_PROJECT_ROOT || HOME_DIR}</span>`;
  }

  function renderProjects(lastProject) {
    const root = el("projectsList");
    root.innerHTML = "";
    if (!projects.length) {
      const empty = document.createElement("div");
      empty.className = "empty-list";
      empty.textContent =
        "No projects in the list yet. Pick a project directory (Browse or Use Home), name it, and click Save to add an entry. Each different directory becomes another row here.";
      root.appendChild(empty);
      return;
    }
    const selectedKey = selected?.project_root || null;
    projects.forEach((p) => {
      const item = document.createElement("div");
      item.className = "item" + (selectedKey && selectedKey === p.project_root ? " active" : "");

      const left = document.createElement("div");
      const title = document.createElement("div");
      const strong = document.createElement("strong");
      strong.textContent = p.name || "(unnamed)";
      title.appendChild(strong);
      const pathLine = document.createElement("div");
      pathLine.className = "muted";
      pathLine.textContent = p.project_root;
      const metaLine = document.createElement("div");
      metaLine.className = "muted";
      metaLine.textContent =
        (p.api_base || "") +
        (lastProject === p.project_root ? " · last used" : "") +
        (p.last_opened ? ` · opened ${p.last_opened}` : "");
      left.appendChild(title);
      left.appendChild(pathLine);
      left.appendChild(metaLine);

      const right = document.createElement("div");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn";
      btn.textContent = "Select";
      btn.onclick = () => {
        applyForm(p);
        renderProjects(lastProject);
      };
      right.appendChild(btn);

      item.appendChild(left);
      item.appendChild(right);
      root.appendChild(item);
    });
  }

  async function refresh() {
    const r = await fetch(`${API}/manager/projects`);
    if (!r.ok) throw new Error(await r.text());
    const payload = await r.json();
    lastKnownLastProject = payload.last_project || null;
    projects = sortProjects(payload.projects || []);

    if (selected) {
      const fresh = projects.find((x) => x.project_root === selected.project_root);
      if (fresh) {
        selected = fresh;
      }
    }

    if (!selected && projects.length) {
      const p =
        projects.find((x) => x.project_root === payload.last_project) || projects[0];
      applyForm(p);
    }

    if (!projects.length) {
      applyForm(null);
    }

    renderProjects(payload.last_project);
  }

  function formPayload() {
    const portText = String(el("apiPort").value || "").trim();
    return {
      project_root: el("projectRoot").value.trim(),
      name: el("name").value.trim() || null,
      description: el("description").value.trim() || "",
      api_host: firstNonBlank(el("apiHost").value, DEFAULT_API_HOST),
      api_port: parsePortOrDefault(portText),
      chimerax_app: el("chimeraxApp").value.trim() || "ChimeraX",
      manifest_path: el("manifestPath").value.trim() || "",
      auto_load_last: el("autoLoadLast").checked,
      start_server_on_launch: el("startServer").checked,
    };
  }

  async function browseDirectory(initialDir, title) {
    const r = await fetch(`${API}/manager/browse/directory`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ initial_dir: initialDir || null, title: title || null }),
    });
    if (!r.ok) throw new Error(await r.text());
    return (await r.json()).path || "";
  }

  async function browseFile(initialDir, title) {
    const r = await fetch(`${API}/manager/browse/file`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ initial_dir: initialDir || null, title: title || null }),
    });
    if (!r.ok) throw new Error(await r.text());
    return (await r.json()).path || "";
  }

  function syncManifestFromProjectIfNeeded() {
    if (!el("manifestSameAsProject").checked) return;
    const root = el("projectRoot").value.trim();
    if (!root) return;
    el("manifestPath").value = `${root}/cryomodel_chimerax_manifest.json`;
  }

  function updateManifestControls() {
    const locked = el("manifestSameAsProject").checked;
    el("manifestPath").disabled = locked;
    el("browseManifestBtn").disabled = locked;
  }

  async function saveCurrent() {
    const payload = formPayload();
    if (!payload.project_root) throw new Error("Project directory is required.");
    const r = await fetch(`${API}/manager/projects/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(await r.text());
    const out = await r.json();
    selected = out.project;
    await refresh();
    setStatus(`Saved: ${selected.name}`);
  }

  el("newBtn").onclick = () => {
    applyForm(null);
    renderProjects(lastKnownLastProject);
  };
  el("themeToggleBtn").onclick = () => {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    applyTheme(cur === "dark" ? "light" : "dark");
  };
  el("openBtn").onclick = () => {
    const root = prompt("Enter existing project directory path:");
    if (root) {
      el("projectRoot").value = root;
      selected = null;
      renderProjectRootHint(false);
      syncManifestFromProjectIfNeeded();
      renderProjects(lastKnownLastProject);
    }
  };
  el("saveBtn").onclick = async () => {
    try {
      await saveCurrent();
    } catch (e) {
      setStatus(String(e), true);
    }
  };
  el("useHomeBtn").onclick = () => {
    el("projectRoot").value = HOME_DIR;
    selected = null;
    renderProjectRootHint(false);
    syncManifestFromProjectIfNeeded();
    renderProjects(lastKnownLastProject);
  };
  el("browseProjectBtn").onclick = async () => {
    try {
      const picked = await browseDirectory(
        el("projectRoot").value.trim() || DEFAULT_PROJECT_ROOT,
        "Select project directory"
      );
      if (picked) {
        el("projectRoot").value = picked;
        selected = null;
        renderProjectRootHint(false);
        syncManifestFromProjectIfNeeded();
        renderProjects(lastKnownLastProject);
      }
    } catch (e) {
      setStatus(String(e), true);
    }
  };
  el("browseChimeraBtn").onclick = async () => {
    try {
      const picked = await browseFile("/Applications", "Select ChimeraX app or executable");
      if (picked) el("chimeraxApp").value = picked;
    } catch (e) {
      setStatus(String(e), true);
    }
  };
  el("browseManifestBtn").onclick = async () => {
    try {
      const base = el("projectRoot").value.trim() || DEFAULT_PROJECT_ROOT;
      const picked = await browseFile(base, "Select manifest file");
      if (picked) el("manifestPath").value = picked;
    } catch (e) {
      setStatus(String(e), true);
    }
  };
  el("manifestSameAsProject").onchange = () => {
    syncManifestFromProjectIfNeeded();
    updateManifestControls();
  };
  el("projectRoot").addEventListener("input", () => {
    syncManifestFromProjectIfNeeded();
    renderProjectRootHint(Boolean(selected));
  });
  el("deleteBtn").onclick = async () => {
    try {
      const root = el("projectRoot").value.trim();
      if (!root) throw new Error("Project directory is required.");
      if (!confirm("Delete registry entry only? Project data will not be deleted.")) return;
      const r = await fetch(`${API}/manager/projects/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_root: root, yes: true }),
      });
      if (!r.ok) throw new Error(await r.text());
      selected = null;
      await refresh();
      setStatus("Deleted project entry.");
    } catch (e) {
      setStatus(String(e), true);
    }
  };
  el("launchBtn").onclick = async () => {
    try {
      await saveCurrent();
      const payload = formPayload();
      if (!payload.project_root) throw new Error("Project directory is required.");
      const r = await fetch(`${API}/manager/projects/launch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_root: payload.project_root,
          host: payload.api_host,
          port: payload.api_port,
          start_server: payload.start_server_on_launch,
          open_ui: true,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const out = await r.json();
      await refresh();
      const wf = out.workflow_url ? `\nWorkflow UI: ${out.workflow_url}` : "";
      setStatus(`Launched ${out.project_root} → ${out.api_url}${wf}`);
    } catch (e) {
      setStatus(String(e), true);
    }
  };

  applyTheme(preferredTheme());
  refresh().catch((e) => setStatus(String(e), true));
  el("apiHost").placeholder = DEFAULT_API_HOST;
  el("apiPort").placeholder = DEFAULT_API_PORT_TEXT;
  setStatus(`Manager API: ${API} · default workflow API ${DEFAULT_API_HOST}:${DEFAULT_API_PORT_TEXT}`);
  if (!selected) {
    renderProjectRootHint(false);
    updateManifestControls();
  }
})();

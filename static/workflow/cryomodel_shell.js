/**
 * CryoModel workflow UI shell — persisted env (CWD, APIs, manifest), project manager window, registry sync.
 */
(function () {
  const WORKFLOW_ENV_KEY = "cryomodel.workflow.env.v1";
  const MANAGER_API_KEY = "cryomodel.managerApiBase";

  const state = { homeDir: "" };

  function loadWorkflowEnv() {
    try {
      const raw = localStorage.getItem(WORKFLOW_ENV_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (e) {
      return {};
    }
  }

  function saveWorkflowEnv(partial) {
    const cur = loadWorkflowEnv();
    Object.assign(cur, partial);
    try {
      localStorage.setItem(WORKFLOW_ENV_KEY, JSON.stringify(cur));
    } catch (e) {}
  }

  function getManagerApiBase() {
    try {
      return localStorage.getItem(MANAGER_API_KEY) || "http://127.0.0.1:8011";
    } catch (e) {
      return "http://127.0.0.1:8011";
    }
  }

  function setManagerApiBase(s) {
    try {
      localStorage.setItem(MANAGER_API_KEY, String(s || "").replace(/\/$/, ""));
    } catch (e) {}
  }

  function persistTopbarFields() {
    const titleEl = document.getElementById("workflowProjectTitle");
    saveWorkflowEnv({
      cwd: document.getElementById("cwd").value.trim(),
      apiBase: document.getElementById("apiBase").value.replace(/\/$/, ""),
      chimeraManifestPath: document.getElementById("chimeraManifestPath").value.trim(),
      chimeraApp: document.getElementById("chimeraxApp").value.trim(),
      projectName: titleEl ? titleEl.textContent.trim() : ""
    });
    const mab = document.getElementById("managerApiBase");
    if (mab) setManagerApiBase(mab.value);
  }

  function initWorkflowEnvFromStorageAndUrl() {
    const q = new URLSearchParams(window.location.search);
    const env = loadWorkflowEnv();
    const cwdEl = document.getElementById("cwd");
    const apiEl = document.getElementById("apiBase");
    if (q.has("cwd")) cwdEl.value = q.get("cwd");
    else if (env.cwd) cwdEl.value = env.cwd;
    if (q.has("api")) apiEl.value = q.get("api");
    else if (env.apiBase) apiEl.value = env.apiBase;
    const m = document.getElementById("chimeraManifestPath");
    const cx = document.getElementById("chimeraxApp");
    if (q.has("manifest")) m.value = q.get("manifest");
    else if (env.chimeraManifestPath && m) m.value = env.chimeraManifestPath;
    if (q.has("chimerax")) cx.value = q.get("chimerax");
    else if (env.chimeraxApp && cx) cx.value = env.chimeraxApp;
    const mab = document.getElementById("managerApiBase");
    if (mab) mab.value = getManagerApiBase();
    const titleEl = document.getElementById("workflowProjectTitle");
    if (titleEl) {
      const emptyLabel = "No project selected";
      if (q.has("project")) titleEl.textContent = q.get("project") || emptyLabel;
      else if (env.projectName) titleEl.textContent = env.projectName;
      else titleEl.textContent = emptyLabel;
    }
    if (q.has("cwd") || q.has("api") || q.has("manifest") || q.has("chimerax") || q.has("project")) {
      persistTopbarFields();
    }
  }

  function hasLaunchQueryParams() {
    const q = new URLSearchParams(window.location.search);
    return q.has("cwd") || q.has("api") || q.has("manifest") || q.has("chimerax") || q.has("project");
  }

  function applyProjectToTopbar(project) {
    if (!project || typeof project !== "object") return false;
    const cwd = String(project.project_root || "").trim();
    if (!cwd) return false;
    const apiBase = String(project.api_base || "").trim();
    const chimeraxApp = String(project.chimerax_app || "").trim();
    const manifestPath = String(project.manifest_path || "").trim();
    const projectName = String(project.name || "").trim();

    document.getElementById("cwd").value = cwd;
    if (apiBase) document.getElementById("apiBase").value = apiBase;
    if (chimeraxApp) document.getElementById("chimeraxApp").value = chimeraxApp;
    document.getElementById("chimeraManifestPath").value = manifestPath;
    const titleEl = document.getElementById("workflowProjectTitle");
    if (titleEl) titleEl.textContent = projectName || "No project selected";
    persistTopbarFields();
    return true;
  }

  async function hydrateFromManagerLastProjectIfNeeded() {
    if (hasLaunchQueryParams()) return;
    const titleEl = document.getElementById("workflowProjectTitle");
    const hasTitle = titleEl && titleEl.textContent.trim() !== "" && titleEl.textContent.trim() !== "No project selected";
    if (hasTitle) return;
    try {
      const managerApi = (document.getElementById("managerApiBase").value || "").trim() || getManagerApiBase();
      const base = managerApi.replace(/\/$/, "");
      const r = await fetch(`${base}/manager/projects`);
      if (!r.ok) return;
      const data = await r.json();
      const items = Array.isArray(data.projects) ? data.projects : [];
      if (!items.length) return;
      const byRoot = new Map(items.map((p) => [String(p.project_root || ""), p]));
      const picked = byRoot.get(String(data.last_project || "")) || items[0];
      applyProjectToTopbar(picked);
      const label = document.getElementById("registryProjectLabel");
      if (label) label.textContent = `Registry: ${picked.name || "(unnamed)"}`;
    } catch (e) {
      // Non-fatal fallback only.
    }
  }

  async function fetchHomeDirForManager() {
    try {
      const api = document.getElementById("apiBase").value.replace(/\/$/, "");
      const r = await fetch(`${api}/ui/home-dir`);
      if (!r.ok) return;
      const data = await r.json();
      state.homeDir = data.home_dir || "";
    } catch (e) {
      state.homeDir = "";
    }
  }

  function openProjectManagerWindow() {
    const mab = document.getElementById("managerApiBase");
    const managerApi = (mab && mab.value.trim()) || getManagerApiBase();
    const ma = managerApi.replace(/\/$/, "");
    const cwd = document.getElementById("cwd").value.trim();
    const apiStr = document.getElementById("apiBase").value.replace(/\/$/, "");
    let host = "127.0.0.1";
    let port = "8010";
    try {
      const u = new URL(apiStr.includes("://") ? apiStr : `http://${apiStr}`);
      host = u.hostname || host;
      port = u.port || (u.protocol === "https:" ? "443" : "80");
    } catch (e) {}
    const base = new URL("cryomodel_manager.html", window.location.href);
    const params = new URLSearchParams({
      api: ma,
      default_project_root: cwd,
      default_api_host: host,
      default_api_port: port
    });
    if (state.homeDir) params.set("home_dir", state.homeDir);
    window.open(`${base.href}?${params.toString()}`, "_blank", "noopener,noreferrer");
  }

  async function syncProjectFromRegistry() {
    const label = document.getElementById("registryProjectLabel");
    const managerApi = (document.getElementById("managerApiBase").value || "").trim() || getManagerApiBase();
    const cwd = document.getElementById("cwd").value.trim();
    if (!cwd) {
      alert("Set CWD in Advanced first.");
      return;
    }
    try {
      const base = managerApi.replace(/\/$/, "");
      const r = await fetch(`${base}/manager/projects/match?path=${encodeURIComponent(cwd)}`);
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      if (!data.project) {
        label.textContent = "No registry entry for this CWD.";
        return;
      }
      const p = data.project;
      document.getElementById("apiBase").value = p.api_base || document.getElementById("apiBase").value;
      document.getElementById("chimeraxApp").value = p.chimerax_app || "ChimeraX";
      document.getElementById("chimeraManifestPath").value = p.manifest_path || "";
      label.textContent = `Registry: ${p.name || "(unnamed)"}`;
      const wt = document.getElementById("workflowProjectTitle");
      if (wt) wt.textContent = p.name || "";
      persistTopbarFields();
    } catch (e) {
      label.textContent = "";
      alert(`Sync failed (is manager API running at ${managerApi}?): ${e.message || e}`);
    }
  }

  function legacyWorkflowUrl() {
    const cwd = document.getElementById("cwd").value.trim();
    const api = document.getElementById("apiBase").value.replace(/\/$/, "");
    const manifest = document.getElementById("chimeraManifestPath").value.trim();
    const chimerax = document.getElementById("chimeraxApp").value.trim();
    let project = document.getElementById("workflowProjectTitle").textContent.trim();
    if (project === "No project selected") project = "";
    const base = new URL("dna_workflow_ui_demo.html", window.location.href);
    const q = new URLSearchParams();
    if (cwd) q.set("cwd", cwd);
    if (api) q.set("api", api);
    q.set("manifest", manifest);
    q.set("chimerax", chimerax);
    if (project) q.set("project", project);
    return `${base.href}?${q.toString()}`;
  }

  function toggleAdvanced() {
    const p = document.getElementById("advancedPanel");
    const open = !p.classList.contains("is-open");
    p.classList.toggle("is-open", open);
    document.getElementById("advancedToggle").setAttribute("aria-expanded", open ? "true" : "false");
  }

  function wire() {
    initWorkflowEnvFromStorageAndUrl();
    hydrateFromManagerLastProjectIfNeeded().catch(() => {});
    fetchHomeDirForManager().catch(() => {});

    const projectTitleBtn = document.getElementById("projectTitleBtn");
    if (projectTitleBtn) projectTitleBtn.addEventListener("click", openProjectManagerWindow);
    const openPmBtn = document.getElementById("openProjectManagerBtn");
    if (openPmBtn) openPmBtn.addEventListener("click", openProjectManagerWindow);
    document.getElementById("advancedToggle").addEventListener("click", toggleAdvanced);
    document.getElementById("syncRegistryBtn").addEventListener("click", () => syncProjectFromRegistry());

    ["cwd", "apiBase", "chimeraManifestPath", "chimeraxApp", "managerApiBase"].forEach((id) => {
      const n = document.getElementById(id);
      if (n) n.addEventListener("change", persistTopbarFields);
    });

    document.getElementById("legacyWorkflowBtn").addEventListener("click", () => {
      window.location.href = legacyWorkflowUrl();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})();

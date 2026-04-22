# CryoModel Development Phases

This file defines execution scope for post-MVP development, with clear non-goals and exit criteria.

## Current status

- Phase 1 (MVP bridge) is functionally complete.
- Remaining work is mostly cleanup/hardening and polish.
- Phase 2: several workflow cards have passed manual operator verification (DNA Axis, Map Filter, Affilter, PathMeasure, **`model2map`**); foldhunter remains unverified pending re-test. End-of–Phase 2 UX (session launcher, card reorder/clone) is captured in `PHASE2_CARD_BACKLOG.md`. See that file for the running sign-off table.

## Phase 1 cleanup checklist (closeout)

- [ ] Confirm ChimeraX manifest includes disk paths for both structures and maps in common open-from-file cases.
- [ ] Verify `POST /ui/chimerax-manifest` error handling for invalid JSON/schema/pathless entries.
- [ ] Validate UI import behavior for artifact-type filtering (`map.mrc` vs `model.structure`).
- [ ] Verify workflow export/import round-trip (JSON + YAML export; import JSON restore path).
- [ ] Write short operator runbook for end-to-end flow (ChimeraX -> UI import -> run).

Exit criteria (Phase 1 closeout):
- 3 repeatable demo runs on different datasets with no manual code edits.
- No blocking errors in manifest import + card execution path.
- Documentation exists for setup, expected files, and recovery from common errors.

---

## Phase 2: Capability expansion + hardening

### Goals

1. Bring the rest of the CryoModel apps into the workflow UI and API surface.
2. Add test coverage and stabilization for command execution and workflow orchestration.
3. Improve docs/assistant quality with tool-specific operational guidance.
4. Clean up ChimeraX integration reliability and user feedback.

### In scope

- UI card spec coverage for remaining tools and key subcommands.
- Standardized artifact contracts (input/output typing, file naming, output capture).
- Workflow engine and UI integration tests:
  - command rendering/validation tests,
  - workflow import/export tests,
  - manifest ingestion and compatibility filtering tests,
  - smoke tests for end-to-end runs.
- Documentation refresh:
  - tool quick references,
  - workflow cookbook examples,
  - troubleshooting guide,
  - assistant prompt/response best practices.
- Assistant updates:
  - expand knowledge base for newly integrated tools,
  - improve parameter guidance and mode-specific examples.
- ChimeraX integration cleanup:
  - robust path capture for maps/structures,
  - clearer command/status messaging,
  - reliability checks around bundle install/update flow.

### Non-goals

- Full live bidirectional ChimeraX session API.
- Final UI redesign and complete UX overhaul.
- Large-scale external user rollout.

### Deliverables

- Expanded workflow UI with additional production-ready cards.
- Test suite and CI checks for core workflow paths.
- Updated docs set (user + operator + troubleshooting).
- Assistant content refresh aligned with actual tool behavior.
- ChimeraX bridge reliability improvements and known-limitations doc.

### Exit criteria

- >=80% of planned Phase 2 tool surface integrated into UI specs.
- Automated test pass for critical paths (workflow run/import/export/manifest ingest).
- No unresolved P1/P2 defects in workflow execution or ChimeraX import path.
- New user can complete guided setup and run at least two reference workflows from docs.

---

## Phase 3: Limited rollout + UX maturation + advanced bridge

### Goals

1. Run a controlled pilot with external users and collect actionable feedback.
2. Evolve the UI and workflow UX based on observed usage and bottlenecks.
3. Implement advanced ChimeraX API integration.
4. Upgrade assistant to be workflow-aware and context-richer.

### In scope

- Limited rollout program:
  - defined user cohort,
  - feedback cadence,
  - issue triage protocol.
- UX/design iteration:
  - card ergonomics,
  - parameter discoverability,
  - run history and observability,
  - error explainability and recovery UX.
- Telemetry/instrumentation (privacy-conscious, opt-in where needed):
  - workflow usage patterns,
  - failure hotspots,
  - time-to-success metrics.
- Full ChimeraX API bridge:
  - richer data exchange than file-only manifest flow,
  - explicit session-aware operations where useful.
- Assistant v2:
  - better contextual grounding from workflow state,
  - stronger troubleshooting suggestions,
  - tighter integration with docs/examples.

### Non-goals

- Broad public launch at phase start.
- Major backend rearchitecture unrelated to pilot learnings.

### Deliverables

- Pilot release package and onboarding assets.
- Updated UI informed by pilot feedback.
- Advanced ChimeraX integration prototype (then hardened).
- Assistant v2 feature set with measurable usefulness improvements.
- Prioritized post-pilot roadmap with cost/impact estimates.

### Exit criteria

- Pilot cohort completes predefined tasks with acceptable success rate.
- Measurable reduction in common user errors vs pre-pilot baseline.
- Positive usability signal on workflow setup/run experience.
- ChimeraX advanced bridge validated on representative datasets.
- Clear go/no-go decision document for broader release.

---

## Future / advanced tooling (not phase-gated yet)

- **Zonal + meta refinement (cryo-EM):** **Local A0–A2** are in-tree (`cryomodel zonal-refine run` — χ1 hard/soft, optional `--rama-backbone` for small φ/ψ moves vs Ramachandran outliers). **Next:** A3 polish; **global** Part B after local A is mature. See `docs/ZONAL_REFINEMENT_PLAN.md`.

---

## Cross-phase operating rules

- Keep one stabilization lane active in every sprint (bug fixes + regressions).
- Prefer incremental releases with demoable checkpoints.
- Track each feature by: user value, implementation risk, test status, docs status.
- Do not promote phase completion until exit criteria are objectively met.

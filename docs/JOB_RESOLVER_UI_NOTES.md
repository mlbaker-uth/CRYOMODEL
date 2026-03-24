# Job Resolver UI Integration Notes

This note describes how to wire `crymodel.workflow.resolver.resolve_command()` into a card-based UI and map resolver errors to field highlights.

## 1) Resolver wiring flow

For every selected card in the workspace:

1. Build `card` payload from UI state:
   - `card_id`
   - `run_state`
   - `inputs` (with `mode=manual|inherited`)
   - `params`
   - `outputs` (resolved/types for previously run cards)
2. Load corresponding `spec` from job-spec registry (`dnaaxis_extract`, `dnabuild_build`, `basehunter_run`).
3. Call:

```python
from crymodel.workflow.resolver import resolve_command
result = resolve_command(card, spec, workspace)
```

4. Use `result` to drive UI:
   - `result.ok == True`: enable Run button and show rendered command preview
   - `result.ok == False`: disable Run and show field-level errors

## 2) Field-level error mapping

`ResolveResult.errors` entries are shaped like:

```json
{
  "code": "MISSING_REQUIRED_INPUT",
  "field": "centerline_pdb",
  "message": "Required input 'centerline_pdb' is missing."
}
```

Map `error.field` to the right-panel widget by `id`.

Recommended UI behavior:

- **Inline message** under input/param field
- **Red border** on invalid field
- **Top summary** with total error count

Suggested code -> UX mapping:

- `MISSING_REQUIRED_INPUT` -> "Required input missing"
- `MISSING_REQUIRED_PARAM` -> "Required parameter missing"
- `INVALID_PARAM_TYPE` -> "Wrong value type"
- `PARAM_OUT_OF_RANGE` -> "Value out of allowed range"
- `INHERIT_SOURCE_NOT_FOUND` -> "Inherited source missing"
- `INHERIT_SOURCE_NOT_READY` -> "Run upstream job first"
- `INHERIT_TYPE_MISMATCH` -> "Incompatible link type"
- `TEMPLATE_TOKEN_UNRESOLVED` -> "Internal template issue"

## 3) Minimal UI state contract

Per card, keep:

```json
{
  "card_id": "card_b",
  "job_type": "dnabuild_build",
  "run_state": "draft",
  "inputs": {
    "centerline_pdb": {
      "mode": "inherited",
      "source": { "card_id": "card_a", "output_id": "out_pdb" }
    },
    "map": { "mode": "manual", "value": "/data/map.mrc" }
  },
  "params": {
    "resolution": 3.0
  }
}
```

Workspace object must include all cards so inherited links can be resolved.

## 4) Validation timing

Use two validation moments:

- **On every input/param change** (fast feedback)
- **On Run click** (final gate before execution)

Do not run a card unless `result.ok` is true.

## 5) Run API contract (prototype)

If `result.ok`, send `result.command` to your runner API:

### `POST /jobs/run`

Request:

```json
{
  "card_id": "card_b",
  "command": "crymodel dnabuild build --centerline-pdb ...",
  "cwd": "/path/to/project"
}
```

Response:

```json
{
  "run_id": "run_0021",
  "status": "started"
}
```

Then poll:

- `GET /jobs/status/{run_id}`
- `GET /jobs/log/{run_id}`

On success, register declared outputs into:

```json
"outputs": {
  "resolved": { "out_pdb": "outputs/dnabuild/dna_initial.pdb" },
  "types": { "out_pdb": "model.structure" }
}
```

This enables downstream `mode=inherited` fields to auto-resolve.

## 6) Practical UX tips

- Show a small badge for inherited fields:
  - `Inherited from DNA Axis (card_a): out_pdb`
- Add one-click toggle:
  - **Use inherited**
  - **Convert to manual**
- Keep inherited values editable by allowing manual override mode.

---

This is enough to support the first DNA-only workflow prototype with robust validation and predictable run behavior.

## 7) React-style starter pseudocode

Below is lightweight pseudocode showing how to validate and run a selected card.

```tsx
// types are illustrative
type WorkspaceState = {
  cards: any[];
  selectedCardId: string | null;
};

function useCardRunner(specRegistry: Record<string, any>) {
  const [workspace, setWorkspace] = useState<WorkspaceState>({ cards: [], selectedCardId: null });
  const [validationByCard, setValidationByCard] = useState<Record<string, any>>({});
  const [runByCard, setRunByCard] = useState<Record<string, { runId?: string; status?: string }>>({});

  const selectedCard = useMemo(
    () => workspace.cards.find(c => c.card_id === workspace.selectedCardId) ?? null,
    [workspace]
  );

  function validateCard(cardId: string) {
    setWorkspace(prev => {
      const card = prev.cards.find(c => c.card_id === cardId);
      if (!card) return prev;
      const spec = specRegistry[card.job_type];
      const result = resolve_command(card, spec, prev); // Python backend call or local bridge
      setValidationByCard(v => ({ ...v, [cardId]: result }));
      return prev;
    });
  }

  function onFieldChange(cardId: string, patch: any) {
    setWorkspace(prev => {
      const cards = prev.cards.map(c => (c.card_id === cardId ? { ...c, ...patch } : c));
      return { ...prev, cards };
    });
    // fast feedback
    queueMicrotask(() => validateCard(cardId));
  }

  async function runCard(cardId: string) {
    const card = workspace.cards.find(c => c.card_id === cardId);
    if (!card) return;
    const spec = specRegistry[card.job_type];
    const validation = resolve_command(card, spec, workspace); // backend resolver call
    setValidationByCard(v => ({ ...v, [cardId]: validation }));
    if (!validation.ok) return;

    // mark running in UI
    setWorkspace(prev => ({
      ...prev,
      cards: prev.cards.map(c => (c.card_id === cardId ? { ...c, run_state: "running" } : c)),
    }));

    const start = await fetch("/jobs/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        card_id: cardId,
        command: validation.command,
        cwd: "/path/to/project",
      }),
    }).then(r => r.json());

    setRunByCard(r => ({ ...r, [cardId]: { runId: start.run_id, status: "running" } }));

    // simple polling loop
    let done = false;
    while (!done) {
      const status = await fetch(`/jobs/status/${start.run_id}`).then(r => r.json());
      setRunByCard(r => ({ ...r, [cardId]: { ...r[cardId], status: status.status } }));

      if (status.status === "success" || status.status === "error") {
        done = true;

        // update card run_state + outputs on success
        setWorkspace(prev => {
          const cards = prev.cards.map(c => {
            if (c.card_id !== cardId) return c;
            const next = { ...c, run_state: status.status };
            if (status.status === "success" && status.outputs_resolved) {
              next.outputs = {
                ...(next.outputs ?? {}),
                resolved: status.outputs_resolved,
                types: status.outputs_types ?? (next.outputs?.types ?? {}),
              };
            }
            return next;
          });
          return { ...prev, cards };
        });
      } else {
        await new Promise(res => setTimeout(res, 1000));
      }
    }
  }

  return {
    workspace,
    selectedCard,
    validationByCard,
    runByCard,
    onFieldChange,
    validateCard,
    runCard,
  };
}
```

### Field highlight mapping snippet

```tsx
function fieldError(cardId: string, fieldId: string, validationByCard: Record<string, any>) {
  const v = validationByCard[cardId];
  if (!v || !v.errors) return null;
  return v.errors.find((e: any) => e.field === fieldId) ?? null;
}
```

Use `fieldError(...)` to:
- set red border class on invalid input widgets,
- render inline helper text under the field,
- disable Run button until `validation.ok === true`.

# Contributing

## Branching
- `main`: stable releases
- `dev`: integration branch
- feature branches: `feat/<short-name>`

## PR checklist
- [ ] Small (< 400 lines changed)
- [ ] Tests added/updated
- [ ] `pre-commit run -a` passes
- [ ] CI green for 3.9–3.12
- [ ] Docs/CLI examples updated if behavior changed

## Dev setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"   # or just -e .
pre-commit install
pytest

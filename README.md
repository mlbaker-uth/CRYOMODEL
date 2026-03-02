CRYOMODEL
=========

CryoModel is a unified cryo-EM modeling toolkit with a command-line interface
(`crymodel`) and multiple modules for modeling, fitting, validation, and
workflow automation.

If you are new to Python or the terminal, start with `INSTALL.md`.

Quick Start
-----------

Install from source (recommended for development):

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Run a quick check:

```
crymodel --help
```

Optional features:

```
pip install -e ".[ml]"
pip install -e ".[pathwalk]"
```

Docs
----

- `INSTALL.md` - step-by-step install for new users
- `CRYOMODEL_USER_GUIDE.md` - end-to-end usage
- `WORKFLOW_GUIDE.md` - workflow system
- `ASSISTANT_GUIDE.md` - assistant usage
- `BASEHUNTER_TEMPLATES.md` - BaseHunter templates



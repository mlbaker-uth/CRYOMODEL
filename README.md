Install and Setup Guide (Linux, macOS, Windows)
================================================

This is a complete setup guide for CLI and ChimeraX users.

Repository:

- https://github.com/mlbaker-uth/CRYOMODEL

Contents
--------

1. What you need
2. Download the code
3. Install CRYOMODEL CLI
4. Verify CLI install
5. Install the ChimeraX bundle
6. Update workflow
7. Troubleshooting


1) What you need
----------------

Required:

- Python 3.9 or newer
- pip
- Git (recommended)

Optional but common:

- UCSF ChimeraX (for interactive tools and bundle commands)

Version checks (run in a terminal):

```bash
python --version
pip --version
git --version
```

If `python` is not recognized, try `python3`.


2) Download the code
--------------------

### Option A (recommended): Git clone

```bash
git clone https://github.com/mlbaker-uth/CRYOMODEL.git
cd CRYOMODEL
```

### Option B: Download ZIP

1. Open https://github.com/mlbaker-uth/CRYOMODEL
2. Click Code -> Download ZIP
3. Unzip
4. Open a terminal in the unzipped `CRYOMODEL` folder


3) Install CRYOMODEL CLI
------------------------

Use a virtual environment so dependencies stay isolated.

### Linux

```bash
cd /path/to/CRYOMODEL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

`cryomodel manager` and `cryomodel manager serve` need the API stack (`fastapi`, `uvicorn`), which is included in the default install above.

### macOS

```bash
cd /path/to/CRYOMODEL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

### Windows (PowerShell)

```powershell
cd C:\path\to\CRYOMODEL
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

If activation is blocked by execution policy, run once in PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then open a new PowerShell window and activate again.

### Optional extras

Install optional dependency groups as needed:

```bash
pip install -e ".[ml]"
pip install -e ".[pathwalk]"
```


4) Verify CLI install
---------------------

In the same activated environment:

```bash
cryomodel --help
```

You should see command help text.

Quick sanity checks:

```bash
cryomodel mapfilter list
cryomodel --version
```


5) Install the ChimeraX bundle
------------------------------

If you only use CLI tools, you can skip this section.

Bundle location in this repo:

- `chimerax-bundles/cryomodel` (main CryoModel bundle)
- `chimerax-bundles/pyhole_chimerax_fix4`, `chimerax-bundles/ChimeraX-pathwalker-fixed4`, `chimerax-bundles/cryomodel_domain_com` (satellite tools; same `devel install` pattern)

See `chimerax-bundles/cryomodel/README.md` for bundle names used by `toolshed uninstall`.

In ChimeraX command line (not shell terminal), run:

```text
devel clean /absolute/path/to/CRYOMODEL/chimerax-bundles/cryomodel
devel install /absolute/path/to/CRYOMODEL/chimerax-bundles/cryomodel
```

Then verify ChimeraX command registration:

```text
cryomodel_manifest
```

Expected result:

- A manifest file is written (default `~/cryomodel_chimerax_manifest.json`)

Notes:

- Use absolute paths.
- Re-run `devel install` after pulling new changes to the bundle.


6) Update workflow
------------------

After first install, future updates are usually:

```bash
cd /path/to/CRYOMODEL
git pull
source .venv/bin/activate   # Linux/macOS
# OR .\.venv\Scripts\Activate.ps1  # Windows
pip install -e .
```

For ChimeraX users, also run in ChimeraX:

```text
devel install /absolute/path/to/CRYOMODEL/chimerax-bundles/cryomodel
```


7) Troubleshooting
------------------

### `python` / `python3` / `py` not found

- Linux/macOS: install Python 3 from your package manager or python.org
- Windows: install Python from python.org and enable PATH option

### `cryomodel: command not found`

- Virtual environment is not active
- Reactivate environment and run:
  ```bash
  pip install -e .
  ```

### `pip` installs to wrong Python

Use:

```bash
python -m pip install -U pip
python -m pip install -e .
```

### `cryomodel manager serve` says `pip install uvicorn fastapi`

Install (or repair) dependencies in the active environment:

```bash
python -m pip install -U pip
python -m pip install -e .
```

### Windows activation script blocked

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then open a new PowerShell window.

### ChimeraX says command not found (`devel` or `cryomodel_manifest`)

- Ensure command is run in ChimeraX command line, not OS terminal
- Confirm bundle path is correct and absolute
- Re-run:
  ```text
  devel clean /absolute/path/to/CRYOMODEL/chimerax-bundles/cryomodel
  devel install /absolute/path/to/CRYOMODEL/chimerax-bundles/cryomodel
  ```

### Clean reinstall

From repo root:

```bash
rm -rf .venv  # Linux/macOS
```

Windows PowerShell:

```powershell
Remove-Item -Recurse -Force .venv
```

Then repeat sections 3-5.

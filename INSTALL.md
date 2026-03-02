Install CryoModel (Beginner Friendly)
=====================================

This guide assumes no prior Python experience. It walks you through installing
CryoModel from GitHub using a virtual environment so your system stays clean.

1) Install Python
-----------------

- Download Python 3.9+ from https://www.python.org/downloads/
- During installation, enable "Add Python to PATH" if you are on Windows.

2) Install Git (optional but recommended)
-----------------------------------------

- Download Git: https://git-scm.com/downloads

If you do not want to use Git, you can download the repository ZIP from GitHub
and skip to step 4.

3) Download CryoModel
---------------------

With Git:

```
git clone https://github.com/<your-org-or-username>/CRYOMODEL.git
```

Without Git:

1. Go to the CryoModel GitHub page.
2. Click "Code" -> "Download ZIP".
3. Unzip it.

4) Open a terminal and go to the folder
---------------------------------------

On macOS/Linux:

```
cd /path/to/CRYOMODEL
```

On Windows (PowerShell):

```
cd C:\path\to\CRYOMODEL
```

5) Create and activate a virtual environment
--------------------------------------------

macOS/Linux:

```
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

6) Install CryoModel
--------------------

```
pip install -U pip
pip install -e .
```

7) Verify the install
---------------------

```
crymodel --help
```

If you see the help output, you are ready to go.

Optional features
-----------------

- Machine learning tools:
  ```
  pip install -e ".[ml]"
  ```

- PathWalker optimization:
  ```
  pip install -e ".[pathwalk]"
  ```

Common issues
-------------

- "python: command not found"
  - Reinstall Python and ensure it is on PATH.

- "crymodel: command not found"
  - Make sure the virtual environment is activated.
  - Re-run `pip install -e .`



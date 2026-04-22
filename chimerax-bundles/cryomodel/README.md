# CryoModel ChimeraX bundle

ChimeraX tools and commands for CryoModel (domain tool, workflow manifest, etc.).

## Build / install (in ChimeraX)

```text
devel clean /path/to/chimerax-bundles/cryomodel
devel install /path/to/chimerax-bundles/cryomodel
```

If `toolshed install` needs metadata fixes, see `INSTALL_WHEEL.txt` and `fix_wheel_metadata.py`.

## Manifest for workflow UI

In ChimeraX’s **command line** (not the terminal):

```text
cryomodel_manifest
```

Writes `~/cryomodel_chimerax_manifest.json` by default.

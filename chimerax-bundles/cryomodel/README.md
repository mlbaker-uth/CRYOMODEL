# CryoModel ChimeraX bundle

ChimeraX tools and commands for CryoModel (BaseHunter Interactive, workflow manifest, legacy domain tool, and related commands).

## What ships in this wheel vs other wheels

This bundle is the **main CryoModel** wheel: **BaseHunter Interactive**, **CryoModel Manifest**, **CryoModel Domain Tool** (legacy UI kept for debugging until the Domain/COM bundle is verified), plus `**cryomodel_findligands`**, `**cryomodel_pdbdomain**`, and `**cryomodel_manifest**`.

**PyHole**, **PathWalker**, and **CryoModel Domain COM** ship as **separate ChimeraX bundles** next to this one (same `chimerax-bundles/` parent). Their `**bundle_info.xml`** uses `**CryoModel**` for `<Categories>` and for the tool classifier’s fourth field (`**ChimeraX :: Tool :: … :: CryoModel :: …**`), so they appear under **Tools → CryoModel**.


| Directory (under `chimerax-bundles/`) | `BundleInfo` name (`toolshed list` / `toolshed uninstall`) |
| ------------------------------------- | ---------------------------------------------------------- |
| `pyhole_chimerax_fix4`                | `ChimeraX-pyHole`                                          |
| `ChimeraX-pathwalker-fixed4`          | `ChimeraX-pathwalker`                                      |
| `cryomodel_domain_com`                | `ChimeraX-CryoModel-Domain-COM`                            |


Those trees use `**bundle_info.xml` + `src/`** (legacy layout). If you later add a `**pyproject.toml**` to a satellite bundle for the same style as `cryomodel/`, mirror `**[tool.chimerax.tool."…"]` `category = "CryoModel"**` there as well.

When Domain/COM is stable, **remove the legacy CryoModel Domain Tool** from this bundle and let the Domain/COM wheel own that menu entry (same display name if you want a drop-in replacement). A batch or script installer for multiple wheels can come later.

## ChimeraX: clean, build, install, uninstall

Run these in ChimeraX’s **command line** (not the host terminal), unless noted.

### See what is installed

```text
toolshed list installed
```

Use the **bundle name** column (for this repo’s main bundle it is `**cryomodel`**, matching `name` in `bundle_info.xml`). Satellites use the names in the table above (e.g. `**toolshed uninstall ChimeraX-pyHole**`).

### Uninstall a bundle (wheel or toolshed install)

```text
toolshed uninstall cryomodel
```

If ChimeraX refuses because another bundle declares a dependency, either uninstall the dependent bundle first or force removal (use sparingly):

```text
toolshed uninstall cryomodel forceRemove true
```

Repeat with each satellite bundle’s name when you want to remove PyHole, Pathwalker, or Domain/COM.

### Developer install from source (editable)

After code changes, a full reinstall picks up metadata and non-Python assets; Python-only edits may be enough with `**editable true**` (see ChimeraX `devel install` docs).

```text
devel clean /path/to/CRYOMODEL/chimerax-bundles/cryomodel
devel install /path/to/CRYOMODEL/chimerax-bundles/cryomodel
```

Use the **same two commands** on each satellite directory, for example:

```text
devel clean /path/to/CRYOMODEL/chimerax-bundles/pyhole_chimerax_fix4
devel install /path/to/CRYOMODEL/chimerax-bundles/pyhole_chimerax_fix4
```

(and similarly for `ChimeraX-pathwalker-fixed4` and `cryomodel_domain_com`).

### Wheel build + `toolshed install` (distribution-style)

1. Optional: `devel clean /path/to/<bundle>`
2. `devel build /path/to/<bundle>`
3. Outside ChimeraX, if the toolshed rejects the wheel metadata, run `python3 fix_wheel_metadata.py dist/<wheel>.whl` from this bundle (see `INSTALL_WHEEL.txt`).
4. In ChimeraX: `toolshed install /full/path/to/dist/<wheel>.whl`
5. Restart ChimeraX.

Wheel file names follow `**[project] name` and version** in each bundle’s `pyproject.toml` (for example `cryomodel-0.1.6-py3-none-any.whl` for the main bundle).

More detail for the main bundle’s wheel path is in `**INSTALL_WHEEL.txt`**.

### Before building the main `cryomodel` bundle (library sync)

If BaseHunter or other code expects the vendored `**cryomodel**` package under `src/cryomodel/`, run `**scripts/sync_cryomodel_into_bundle.sh**` from the bundle root (or your usual sync) before `**devel build**` / `**devel install**`.

## Manifest for workflow UI

In ChimeraX’s **command line** (not the terminal):

```text
cryomodel_manifest
```

Writes `~/cryomodel_chimerax_manifest.json` by default.
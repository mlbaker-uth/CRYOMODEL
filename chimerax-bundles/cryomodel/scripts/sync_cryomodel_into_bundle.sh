#!/usr/bin/env bash
# Copy the CRYOMODEL Python package into this bundle's src/ so ``devel build`` can ship it.
# ChimeraX bundle ExtraDir does not join paths to the bundle root (copy fails); run this instead.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BUNDLE_ROOT}/../.." && pwd)"
SRC_LIB="${REPO_ROOT}/cryomodel"
DEST="${BUNDLE_ROOT}/src/cryomodel"
if [[ ! -d "${SRC_LIB}" ]]; then
  echo "error: expected library at ${SRC_LIB}" >&2
  exit 1
fi
rm -rf "${DEST}"
cp -R "${SRC_LIB}" "${DEST}"
echo "Synced ${SRC_LIB} -> ${DEST}"

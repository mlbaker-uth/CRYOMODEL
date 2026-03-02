from typing import Optional, Tuple
import numpy as np
try: import gemmi
except Exception: gemmi = None
try: import mrcfile
except Exception: mrcfile = None

from ..core.types import MapVolume, ModelAtoms

class _Structure:
    def read_model(self, path: str) -> ModelAtoms:
        if gemmi is None: raise ImportError("gemmi is required for CIF/PDB I/O")
        s = gemmi.read_structure(path)
        xyz=[]; name=[]; resname=[]; chain=[]; resi=[]; element=[]
        for mdl in s:
            for ch in mdl:
                for res in ch:
                    for at in res:
                        xyz.append([at.pos.x, at.pos.y, at.pos.z])
                        name.append(at.name); resname.append(res.name)
                        chain.append(ch.name)
                        resi.append(res.seqid.num if res.seqid.num is not None else 0)
                        element.append(at.element.name if at.element else at.name.strip()[0])
        return ModelAtoms(
            xyz=np.array(xyz, np.float32),
            name=np.array(name, object),
            resname=np.array(resname, object),
            chain=np.array(chain, object),
            resi=np.array(resi, int),
            element=np.array(element, object),
        )

    def write_model(self, path: str, model: ModelAtoms) -> None:
        if gemmi is None:
            raise ImportError("gemmi is required for CIF/PDB I/O")

        st = gemmi.Structure()
        try:
            st.spacegroup_hm = "P 1"
        except Exception:
            pass

        mdl = gemmi.Model(1)
        st.add_model(mdl)

        chains = {}
        residues = {}

        n = int(len(model.xyz))
        print(f"[debug] writing {len(model.xyz)} atoms -> {path}")
        
        for i in range(n):
            # --- coerce all fields to plain Python scalars/str ---
            x, y, z = map(float, model.xyz[i].tolist())
            chn = str(model.chain[i]) if i < len(model.chain) else "A"
            resn = int(model.resi[i]) if i < len(model.resi) else (i + 1)
            rnm = str(model.resname[i]) if i < len(model.resname) else "LIG"
            anm = str(model.name[i]) if i < len(model.name) else "C1"
            elm = str(model.element[i]).upper() if i < len(model.element) else "C"

            # chain
            if chn not in chains:
                chains[chn] = gemmi.Chain(chn)
                mdl.add_chain(chains[chn])

            # residue
            key = (chn, resn, rnm)
            if key not in residues:
                res = gemmi.Residue()
                res.name = str(rnm)
                try:
                    res.seqid = gemmi.SeqId(int(resn), "")
                except Exception:
                    res.seqid = gemmi.SeqId(str(resn))
                chains[chn].add_residue(res)
                residues[key] = res

            # atom
            at = gemmi.Atom()

            # name (handle both attribute and setter; force plain str)
            name_s = str(anm)
            if hasattr(at, "set_name"):
                at.set_name(name_s)           # some Gemmi builds expose set_name()
            else:
                at.name = name_s              # others expose .name as a property

            # element (force plain str; fall back to C on failure)
            elm_s = str(elm).upper()
            try:
                at.element = gemmi.Element(elm_s)
            except Exception:
                at.element = gemmi.Element("C")

            # coordinates (force plain floats)
            at.pos = gemmi.Position(float(x), float(y), float(z))

            # make PDB path happy (even if you stick to mmCIF, harmless)
            at.occ = 1.0
            at.b_iso = 20.0
            #at.altloc = ''
            at.serial = i + 1

            residues[key].add_atom(at)
            
        # write PDB or mmCIF
        path_lower = path.lower()
        if path_lower.endswith(".cif") or path_lower.endswith(".mmcif"):
            doc = st.make_mmcif_document()
            doc.write_file(path)
        else:
            st.write_pdb(path)   # not write_minimal_pdb()
class _Maps:
    def read_map(self, path: str) -> MapVolume:
        if mrcfile is None: raise ImportError("mrcfile is required for MRC/CCP4 I/O")
        with mrcfile.open(path, permissive=True) as m:
            data = np.array(m.data, np.float32)
            apix = float(getattr(m.voxel_size, 'x', 1.0) or 1.0)
            origin = (float(getattr(m.header, 'nxstart', 0)),
                      float(getattr(m.header, 'nystart', 0)),
                      float(getattr(m.header, 'nzstart', 0)))
        return MapVolume(data=data, apix=apix, origin=origin)

    def write_map(self, path: str, vol: MapVolume) -> None:
        if mrcfile is None: raise ImportError("mrcfile is required for MRC/CCP4 I/O")
        with mrcfile.new(path, overwrite=True) as m:
            m.set_data(vol.data.astype('float32'))
            try: m.voxel_size = (vol.apix, vol.apix, vol.apix)
            except Exception: pass

class Backend:
    structure = _Structure()
    maps = _Maps()

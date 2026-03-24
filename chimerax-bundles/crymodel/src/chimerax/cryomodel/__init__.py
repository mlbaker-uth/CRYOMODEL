from chimerax.core.toolshed import BundleAPI


class _CryoModelAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        """Start the CryoModel PDB domain / COM tool."""
        from .pdbdomain_tool import PDBDomainTool
        return PDBDomainTool(session, ti.name)

    @staticmethod
    def get_class(class_name):
        """Allow session save/restore of the tool."""
        if class_name == "PDBDomainTool":
            from .pdbdomain_tool import PDBDomainTool
            return PDBDomainTool
        raise ValueError(class_name)


bundle_api = _CryoModelAPI()


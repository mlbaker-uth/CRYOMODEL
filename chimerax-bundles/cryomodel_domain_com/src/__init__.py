from chimerax.core.toolshed import BundleAPI


class _API(BundleAPI):
    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        from .pdbcom_tool import CryoModelDomainCOMTool
        return CryoModelDomainCOMTool(session, ti.name)

    @staticmethod
    def get_class(class_name):
        if class_name == "CryoModelDomainCOMTool":
            from .pdbcom_tool import CryoModelDomainCOMTool
            return CryoModelDomainCOMTool
        raise ValueError(class_name)


bundle_api = _API()

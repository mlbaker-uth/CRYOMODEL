from chimerax.core.toolshed import BundleAPI


class _API(BundleAPI):
    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        from .pathwalker_tool import PathWalkerTool
        return PathWalkerTool(session, ti.name)

    @staticmethod
    def get_class(class_name):
        if class_name == "PathWalkerTool":
            from .pathwalker_tool import PathWalkerTool
            return PathWalkerTool
        raise ValueError(class_name)


bundle_api = _API()

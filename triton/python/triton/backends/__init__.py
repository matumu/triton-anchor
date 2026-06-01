import sys
import importlib.metadata
from dataclasses import dataclass
from .driver import DriverBase
from .compiler import BaseBackend


@dataclass(frozen=True)
class Backend:
    compiler: BaseBackend = None
    driver: DriverBase = None


def _discover_backends():
    """通过 entry_points 发现已安装的 out-of-tree 后端插件。

    后端插件（如 triton-sophgo-backend）在 pyproject.toml 中声明:
        [project.entry-points."triton.backends"]
        sophgo = "triton_sophgo"
    这样 `import triton` 时就能自动发现并注册该后端。
    """
    backends = dict()

    try:
        eps = importlib.metadata.entry_points(group="triton.backends")
    except TypeError:
        # Python 3.9 兼容: entry_points() 不支持 group 参数
        eps = importlib.metadata.entry_points().get("triton.backends", [])

    for ep in eps:
        try:
            plugin_obj = ep.load()

            # 支持两种 entry_point 指向格式:
            #   1. 类:   实例化后读取 compiler_cls / driver_cls
            #   2. 模块: 直接读取模块级 compiler_cls / driver_cls
            plugin = plugin_obj() if isinstance(plugin_obj, type) else plugin_obj

            compiler_cls = getattr(plugin, 'compiler_cls', None)
            driver_cls = getattr(plugin, 'driver_cls', None)

            if compiler_cls and driver_cls:
                backends[ep.name] = Backend(compiler=compiler_cls, driver=driver_cls)
        except Exception as e:
            print(f"Warning: Failed to load out-of-tree backend '{ep.name}': {e}",
                  file=sys.stderr)

    return backends


backends = _discover_backends()

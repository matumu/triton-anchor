import importlib.metadata
try:
    eps = importlib.metadata.entry_points(group="triton.backends")
except TypeError:
    eps = importlib.metadata.entry_points().get("triton.backends", [])
for ep in eps:
    print(ep.name, ep.module)

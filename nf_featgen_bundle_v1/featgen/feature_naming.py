FUTR_PREFIX = "futr_"
HIST_PREFIX = "hist_"
STAT_PREFIX = "stat_"

def prefixed(name: str, kind: str) -> str:
    if kind == "futr":
        return f"{FUTR_PREFIX}{name}"
    if kind == "hist":
        return f"{HIST_PREFIX}{name}"
    if kind == "stat":
        return f"{STAT_PREFIX}{name}"
    return name

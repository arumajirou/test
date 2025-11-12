def has_gpu() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False

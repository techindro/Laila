from src.config import LailaConfig, require_hf_token, resolve_device


def __getattr__(name: str):
    if name == "LailaASR":
        from src.inference import LailaASR

        return LailaASR
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = ["LailaConfig", "LailaASR", "require_hf_token", "resolve_device"]

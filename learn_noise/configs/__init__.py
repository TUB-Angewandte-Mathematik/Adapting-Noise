from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping

import yaml

_CONFIG_ROOT = Path(__file__).resolve().parent
_DOMAIN_DIRS: Dict[str, Path] = {
    "2d": _CONFIG_ROOT / "2d",
    "images": _CONFIG_ROOT / "images",
}


def list_configs(domain: str) -> Iterable[str]:
    """Return available configuration names for a given domain."""
    domain = domain.lower()
    if domain not in _DOMAIN_DIRS:
        raise ValueError(f"Unknown domain '{domain}'. Expected one of: {sorted(_DOMAIN_DIRS)}")
    return sorted(p.stem for p in _DOMAIN_DIRS[domain].glob("*.yaml"))


def load_config(domain: str, name: str) -> Dict[str, object]:
    """Load a configuration YAML file as a flat dictionary."""
    domain = domain.lower()
    if domain not in _DOMAIN_DIRS:
        raise ValueError(f"Unknown domain '{domain}'. Expected one of: {sorted(_DOMAIN_DIRS)}")
    config_path = _DOMAIN_DIRS[domain] / f"{name}.yaml"
    if not config_path.exists():
        available = ", ".join(list_configs(domain))
        raise FileNotFoundError(f"Config '{name}' not found for domain '{domain}'. Available: {available}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Config file {config_path} must define a mapping at the top level")
    return dict(data)


def apply_overrides(cfg: MutableMapping[str, object], overrides: Mapping[str, object]) -> MutableMapping[str, object]:
    """Apply string overrides (top-level only) to a config dictionary."""
    for key, value in overrides.items():
        if key not in cfg:
            raise KeyError(f"Unknown config key '{key}'")
        cfg[key] = value
    return cfg


def parse_override_strings(pairs: Iterable[str]) -> Dict[str, object]:
    """Parse key=value strings into a dictionary, attempting numeric/boolean conversion."""
    result: Dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must be in key=value format")
        key, raw = pair.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if raw.lower() in {"true", "false"}:
            value: object = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        result[key] = value
    return result

__all__ = [
    "list_configs",
    "load_config",
    "apply_overrides",
    "parse_override_strings",
]

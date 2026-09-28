"""Decision backend factory."""
from functools import lru_cache
from .base import backend_mode
from .laya_backend import SafeLayaDecisionBackend
from .rule_backend import RuleDecisionBackend

@lru_cache(maxsize=1)
def get_decision_backend():
    return SafeLayaDecisionBackend() if backend_mode() in {"laya", "laya_shadow"} else RuleDecisionBackend()

"""Config-driven ASR backends.

Importing this package triggers the ``@register_asr`` decorators, which is
how backends become visible to ``psdn_sonar.registry.get_asr_backend``.
"""

from . import huggingface_backend

__all__ = ["huggingface_backend"]

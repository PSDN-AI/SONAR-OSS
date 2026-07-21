"""Per-language text processors.

Importing this package triggers the ``@register_language`` decorators, which
is how processors become visible to ``psdn_sonar.registry``.
"""

from . import bengali, english, hindi, korean

__all__ = ["bengali", "english", "hindi", "korean"]

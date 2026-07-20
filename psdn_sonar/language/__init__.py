"""Per-language text processors.

Importing this package triggers the ``@register_language`` decorators, which
is how processors become visible to ``psdn_sonar.registry``. Hindi and Korean
processors land in a follow-up import PR.
"""

from . import bengali, english

__all__ = ["bengali", "english"]

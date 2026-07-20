"""Per-language text processors.

Importing this package triggers the ``@register_language`` decorators, which
is how processors become visible to ``psdn_sonar.registry``. Bengali, Hindi,
and Korean processors land in follow-up import PRs.
"""

from . import english

__all__ = ["english"]

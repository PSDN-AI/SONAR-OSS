"""Guarded import of the optional ``bnlp`` package (issue #204).

``bnlp``'s tokenizer module runs an NLTK resource check in its module body
that names one resource and downloads another::

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("punkt not found. downloading...")
        nltk.download("punkt_tab")

On an environment holding ``punkt_tab`` (what NLTK's tokenizer actually uses,
and what the branch downloads) but not the legacy ``punkt``, the check fails
forever and ``nltk.download`` runs on *every* import — a network round trip
to the NLTK data host with no timeout, observed blocking a run indefinitely
when the remote closed the connection mid-transfer. Because
``psdn_sonar.utils.text_processing`` sits on every evaluation path, any
install with the ``[bengali]`` extra paid that round trip for English, Hindi
and Korean runs too.

Every ``bnlp`` import in this package must go through :func:`import_bnlp`,
which imports the module with two temporary patches in place:

- When ``punkt_tab`` is already local, ``nltk.data.find`` answers the
  ``tokenizers/punkt`` probe with the ``punkt_tab`` lookup, so the check
  succeeds, nothing is printed, and no download runs — the import is fully
  offline.
- When ``punkt_tab`` is genuinely missing, the download must run for the
  tokenizer to work, but a socket default timeout bounds it so a dead
  connection fails the import instead of hanging the run.

Both patches are restored before the function returns.
"""

import importlib.util
import logging
import socket
import sys

logger = logging.getLogger(__name__)

# Bounds every socket operation of a genuinely required first-time download.
# Generous: the punkt_tab archive is small (~4 MB), so a healthy connection
# finishes far sooner; a peer that stopped responding fails in one minute
# instead of never (issue #204: 22 minutes at zero progress, ended by hand).
NLTK_DOWNLOAD_TIMEOUT_S = 60.0

_PUNKT_QUERY = "tokenizers/punkt"
_PUNKT_TAB_QUERY = "tokenizers/punkt_tab"


def bnlp_installed() -> bool:
    """True when the ``bnlp`` package is importable, checked without
    importing it (and therefore without its import-time side effects)."""
    return importlib.util.find_spec("bnlp") is not None


def _punkt_tab_is_local() -> bool:
    try:
        import nltk

        nltk.data.find(_PUNKT_TAB_QUERY)
        return True
    except Exception:
        return False


def import_bnlp():
    """Import and return the ``bnlp`` module, or ``None`` when not installed.

    Exceptions raised by ``bnlp`` itself (beyond it not being installed)
    propagate to the caller, which owns the fallback decision.
    """
    if "bnlp" in sys.modules:
        return sys.modules["bnlp"]
    if not bnlp_installed():
        return None

    import nltk

    original_find = nltk.data.find
    original_download = nltk.download
    original_timeout = socket.getdefaulttimeout()

    if _punkt_tab_is_local():
        # The resource the tokenizer actually uses (and that the download
        # branch would fetch) is already here; answer the stale 'punkt'
        # probe with it so the check passes offline, and neuter download
        # in case some other probe still reaches it.
        def _find(resource_name, *args, **kwargs):
            if resource_name == _PUNKT_QUERY:
                resource_name = _PUNKT_TAB_QUERY
            return original_find(resource_name, *args, **kwargs)

        def _download(*args, **kwargs):
            logger.debug("Suppressed nltk.download(%s) during bnlp import: punkt_tab is already local", args)
            return True

        # setattr keeps ty happy: the wrappers deliberately have looser
        # signatures than the attributes they temporarily replace.
        setattr(nltk.data, "find", _find)  # noqa: B010
        setattr(nltk, "download", _download)  # noqa: B010
    elif original_timeout is None:
        socket.setdefaulttimeout(NLTK_DOWNLOAD_TIMEOUT_S)

    try:
        import bnlp

        return bnlp
    finally:
        nltk.data.find = original_find
        nltk.download = original_download
        socket.setdefaulttimeout(original_timeout)

"""Patch triton's gdc_launch_dependents for compatibility.

Newer triton versions (≥3.2) removed ``gdc_launch_dependents`` from
``triton.language.extra.cuda``.  This patch adds it as a no-op when
missing, preventing ImportError in code paths that reference it.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import triton.language.extra.cuda as _cuda

    if not hasattr(_cuda, "gdc_launch_dependents"):

        def _noop():
            pass

        _cuda.gdc_launch_dependents = _noop
        logger.debug("Patched triton: added gdc_launch_dependents as no-op")
except Exception:
    logger.debug("triton not available, skipping gdc_launch_dependents patch")

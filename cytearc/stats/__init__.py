"""Replicate-level statistics: designs, empirical Bayes moderation, and FDR control.

The package has no storage dependency. :mod:`cytearc.abundance` uses its
empirical Bayes linear models for propeller and its designs and FDR
corrections for Milo, whose edgeR steps run in the optional edgePython
package.
"""

from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .design import (
        DesignMatrix as DesignMatrix,
        contrast_vector as contrast_vector,
        treatment_design as treatment_design,
    )
    from .limma import (
        LinearModelFit as LinearModelFit,
        ModeratedFit as ModeratedFit,
        contrasts_fit as contrasts_fit,
        e_bayes as e_bayes,
        lm_fit as lm_fit,
        squeeze_var as squeeze_var,
    )
    from .multiple_testing import (
        p_adjust_bh as p_adjust_bh,
        weighted_bh as weighted_bh,
    )

__all__ = [
    "DesignMatrix",
    "LinearModelFit",
    "ModeratedFit",
    "contrast_vector",
    "contrasts_fit",
    "e_bayes",
    "lm_fit",
    "p_adjust_bh",
    "squeeze_var",
    "treatment_design",
    "weighted_bh",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "DesignMatrix": ".design",
        "LinearModelFit": ".limma",
        "ModeratedFit": ".limma",
        "contrast_vector": ".design",
        "contrasts_fit": ".limma",
        "e_bayes": ".limma",
        "lm_fit": ".limma",
        "p_adjust_bh": ".multiple_testing",
        "squeeze_var": ".limma",
        "treatment_design": ".design",
        "weighted_bh": ".multiple_testing",
    },
)

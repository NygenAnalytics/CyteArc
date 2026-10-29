from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .chains import (
        CHAIN_PAIRINGS as CHAIN_PAIRINGS,
        LOCI as LOCI,
        NO_RECEPTOR as NO_RECEPTOR,
        RECEPTOR_SUBTYPES as RECEPTOR_SUBTYPES,
        RECEPTOR_TYPES as RECEPTOR_TYPES,
        SLOTS as SLOTS,
        VDJ_LOCI as VDJ_LOCI,
        VJ_LOCI as VJ_LOCI,
    )
    from .clonotypes import (
        DUAL_IR as DUAL_IR,
        RECEPTOR_ARMS as RECEPTOR_ARMS,
        Clonotypes as Clonotypes,
    )
    from .contigs import FIELDS as FIELDS
    from .diversity import (
        ORDERS as ORDERS,
        ClonalDiversity as ClonalDiversity,
        Rarefaction as Rarefaction,
    )
    from .expansion import ClonalExpansion as ClonalExpansion
    from .overlap import (
        OVERLAP_INDICES as OVERLAP_INDICES,
        ClonotypeOverlap as ClonotypeOverlap,
        ClonotypeTracking as ClonotypeTracking,
    )
    from .receptors import ImmuneReceptors as ImmuneReceptors
    from .usage import (
        GENE_FIELDS as GENE_FIELDS,
        GENE_LEVELS as GENE_LEVELS,
        GeneUsage as GeneUsage,
        Spectratype as Spectratype,
    )

__all__ = [
    "CHAIN_PAIRINGS",
    "DUAL_IR",
    "FIELDS",
    "GENE_FIELDS",
    "GENE_LEVELS",
    "LOCI",
    "NO_RECEPTOR",
    "ORDERS",
    "OVERLAP_INDICES",
    "RECEPTOR_ARMS",
    "RECEPTOR_SUBTYPES",
    "RECEPTOR_TYPES",
    "SLOTS",
    "VDJ_LOCI",
    "VJ_LOCI",
    "ClonalDiversity",
    "ClonalExpansion",
    "ClonotypeOverlap",
    "ClonotypeTracking",
    "Clonotypes",
    "GeneUsage",
    "ImmuneReceptors",
    "Rarefaction",
    "Spectratype",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "CHAIN_PAIRINGS": ".chains",
        "DUAL_IR": ".clonotypes",
        "FIELDS": ".contigs",
        "GENE_FIELDS": ".usage",
        "GENE_LEVELS": ".usage",
        "LOCI": ".chains",
        "NO_RECEPTOR": ".chains",
        "ORDERS": ".diversity",
        "OVERLAP_INDICES": ".overlap",
        "RECEPTOR_ARMS": ".clonotypes",
        "RECEPTOR_SUBTYPES": ".chains",
        "RECEPTOR_TYPES": ".chains",
        "SLOTS": ".chains",
        "VDJ_LOCI": ".chains",
        "VJ_LOCI": ".chains",
        "ClonalDiversity": ".diversity",
        "ClonalExpansion": ".expansion",
        "ClonotypeOverlap": ".overlap",
        "ClonotypeTracking": ".overlap",
        "Clonotypes": ".clonotypes",
        "GeneUsage": ".usage",
        "ImmuneReceptors": ".receptors",
        "Rarefaction": ".diversity",
        "Spectratype": ".usage",
    },
)

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
    "RECEPTOR_ARMS",
    "RECEPTOR_SUBTYPES",
    "RECEPTOR_TYPES",
    "SLOTS",
    "VDJ_LOCI",
    "VJ_LOCI",
    "Clonotypes",
    "GeneUsage",
    "ImmuneReceptors",
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
        "RECEPTOR_ARMS": ".clonotypes",
        "RECEPTOR_SUBTYPES": ".chains",
        "RECEPTOR_TYPES": ".chains",
        "SLOTS": ".chains",
        "VDJ_LOCI": ".chains",
        "VJ_LOCI": ".chains",
        "Clonotypes": ".clonotypes",
        "GeneUsage": ".usage",
        "ImmuneReceptors": ".receptors",
        "Spectratype": ".usage",
    },
)

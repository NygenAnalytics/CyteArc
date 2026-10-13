from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from ..features.genomic.annotation import (
        link_regions_to_genes as link_regions_to_genes,
        read_gene_annotation as read_gene_annotation,
    )
    from ..features.genomic.motifs import (
        Motif as Motif,
        match_motifs as match_motifs,
        read_jaspar_motifs as read_jaspar_motifs,
        read_meme_motifs as read_meme_motifs,
    )
    from ..features.genomic.sequence import (
        IndexedFasta as IndexedFasta,
        gc_content as gc_content,
        parse_coordinate_ids as parse_coordinate_ids,
    )
    from .cistarget import (
        MotifEnrichment as MotifEnrichment,
        enrich_signature as enrich_signature,
    )
    from .cistromes import (
        binding_cistromes as binding_cistromes,
        cistromes_from_motifs as cistromes_from_motifs,
        project_to_genes as project_to_genes,
        read_peak_file as read_peak_file,
    )
    from .databases import (
        RankingDatabase as RankingDatabase,
        read_motif_annotations as read_motif_annotations,
    )
    from .eregulons import (
        EregulonParameters as EregulonParameters,
        GseaResult as GseaResult,
        build_eregulons as build_eregulons,
        gsea_prerank as gsea_prerank,
    )
    from .grn import (
        GRNBOOST2_PARAMETERS as GRNBOOST2_PARAMETERS,
        GrnParameters as GrnParameters,
        infer_grn as infer_grn,
        pearson_correlations as pearson_correlations,
    )
    from .links import region_gene_links as region_gene_links
    from .modules import (
        ModuleParameters as ModuleParameters,
        modules_from_adjacencies as modules_from_adjacencies,
    )
    from .programs import (
        GeneProgramParameters as GeneProgramParameters,
        gene_programs as gene_programs,
        program_regulators as program_regulators,
        program_scores as program_scores,
        set_overlaps as set_overlaps,
    )
    from .pruning import (
        PruningParameters as PruningParameters,
        prune_modules as prune_modules,
        regulons_from_motifs as regulons_from_motifs,
    )
    from .results import GrnResult as GrnResult
    from .specificity import (
        connection_specificity as connection_specificity,
        regulon_modules as regulon_modules,
        regulon_specificity as regulon_specificity,
    )

__all__ = [
    "EregulonParameters",
    "GRNBOOST2_PARAMETERS",
    "GeneProgramParameters",
    "GrnParameters",
    "GrnResult",
    "GseaResult",
    "IndexedFasta",
    "ModuleParameters",
    "Motif",
    "MotifEnrichment",
    "PruningParameters",
    "RankingDatabase",
    "binding_cistromes",
    "build_eregulons",
    "cistromes_from_motifs",
    "connection_specificity",
    "enrich_signature",
    "gc_content",
    "gene_programs",
    "gsea_prerank",
    "infer_grn",
    "link_regions_to_genes",
    "match_motifs",
    "modules_from_adjacencies",
    "parse_coordinate_ids",
    "pearson_correlations",
    "program_regulators",
    "program_scores",
    "project_to_genes",
    "prune_modules",
    "read_gene_annotation",
    "read_jaspar_motifs",
    "read_meme_motifs",
    "read_motif_annotations",
    "read_peak_file",
    "region_gene_links",
    "regulon_modules",
    "regulon_specificity",
    "regulons_from_motifs",
    "set_overlaps",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "EregulonParameters": ".eregulons",
        "GRNBOOST2_PARAMETERS": ".grn",
        "GeneProgramParameters": ".programs",
        "GrnParameters": ".grn",
        "GrnResult": ".results",
        "GseaResult": ".eregulons",
        "IndexedFasta": "..features.genomic.sequence",
        "ModuleParameters": ".modules",
        "Motif": "..features.genomic.motifs",
        "MotifEnrichment": ".cistarget",
        "PruningParameters": ".pruning",
        "RankingDatabase": ".databases",
        "binding_cistromes": ".cistromes",
        "build_eregulons": ".eregulons",
        "cistromes_from_motifs": ".cistromes",
        "connection_specificity": ".specificity",
        "enrich_signature": ".cistarget",
        "gc_content": "..features.genomic.sequence",
        "gene_programs": ".programs",
        "gsea_prerank": ".eregulons",
        "infer_grn": ".grn",
        "link_regions_to_genes": "..features.genomic.annotation",
        "match_motifs": "..features.genomic.motifs",
        "modules_from_adjacencies": ".modules",
        "parse_coordinate_ids": "..features.genomic.sequence",
        "pearson_correlations": ".grn",
        "program_regulators": ".programs",
        "program_scores": ".programs",
        "project_to_genes": ".cistromes",
        "prune_modules": ".pruning",
        "read_gene_annotation": "..features.genomic.annotation",
        "read_jaspar_motifs": "..features.genomic.motifs",
        "read_meme_motifs": "..features.genomic.motifs",
        "read_motif_annotations": ".databases",
        "read_peak_file": ".cistromes",
        "region_gene_links": ".links",
        "regulon_modules": ".specificity",
        "regulon_specificity": ".specificity",
        "regulons_from_motifs": ".pruning",
        "set_overlaps": ".programs",
    },
)

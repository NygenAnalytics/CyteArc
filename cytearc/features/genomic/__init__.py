from typing import TYPE_CHECKING

from ..._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .annotation import (
        link_regions_to_genes as link_regions_to_genes,
        overlapping_intervals as overlapping_intervals,
        read_gene_annotation as read_gene_annotation,
        transcription_start_sites as transcription_start_sites,
    )
    from .intervals import (
        binary_search as binary_search,
        create_bed_from_coord_ids as create_bed_from_coord_ids,
        get_feature_mappings as get_feature_mappings,
        get_ranges as get_ranges,
    )
    from .melding import create_counts_mat as create_counts_mat
    from .motifs import (
        Motif as Motif,
        log_odds_matrix as log_odds_matrix,
        match_motifs as match_motifs,
        nucleotide_frequencies as nucleotide_frequencies,
        read_jaspar_motifs as read_jaspar_motifs,
        read_meme_motifs as read_meme_motifs,
        reverse_complement_matrix as reverse_complement_matrix,
        scan_motif_matches as scan_motif_matches,
        score_threshold as score_threshold,
    )
    from .sequence import (
        IndexedFasta as IndexedFasta,
        encode_sequence as encode_sequence,
        gc_content as gc_content,
        parse_coordinate_ids as parse_coordinate_ids,
        read_fasta_index as read_fasta_index,
    )

__all__ = [
    "IndexedFasta",
    "Motif",
    "binary_search",
    "create_bed_from_coord_ids",
    "create_counts_mat",
    "encode_sequence",
    "gc_content",
    "get_feature_mappings",
    "get_ranges",
    "link_regions_to_genes",
    "log_odds_matrix",
    "match_motifs",
    "nucleotide_frequencies",
    "overlapping_intervals",
    "parse_coordinate_ids",
    "read_fasta_index",
    "read_gene_annotation",
    "read_jaspar_motifs",
    "read_meme_motifs",
    "reverse_complement_matrix",
    "scan_motif_matches",
    "score_threshold",
    "transcription_start_sites",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "IndexedFasta": ".sequence",
        "Motif": ".motifs",
        "binary_search": ".intervals",
        "create_bed_from_coord_ids": ".intervals",
        "create_counts_mat": ".melding",
        "encode_sequence": ".sequence",
        "gc_content": ".sequence",
        "get_feature_mappings": ".intervals",
        "get_ranges": ".intervals",
        "link_regions_to_genes": ".annotation",
        "log_odds_matrix": ".motifs",
        "match_motifs": ".motifs",
        "nucleotide_frequencies": ".motifs",
        "overlapping_intervals": ".annotation",
        "parse_coordinate_ids": ".sequence",
        "read_fasta_index": ".sequence",
        "read_gene_annotation": ".annotation",
        "read_jaspar_motifs": ".motifs",
        "read_meme_motifs": ".motifs",
        "reverse_complement_matrix": ".motifs",
        "scan_motif_matches": ".motifs",
        "score_threshold": ".motifs",
        "transcription_start_sites": ".annotation",
    },
)

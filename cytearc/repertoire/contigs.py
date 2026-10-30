"""Contigs of single-cell immune receptor libraries, by their AIRR fields.

A V(D)J library of T or B cells, such as 10x Genomics Chromium Single Cell
Immune Profiling, assembles each cell's receptor transcripts into contigs. A
contig is one chain of a T cell receptor (TCR) or B cell receptor (BCR): its
locus, its V, D, J and constant genes, and its junction, the CDR3 with the
conserved residues on either side. Fields follow the AIRR Community's
Rearrangement schema (Vander Heiden et al. 2018), whose names CyteArc keeps,
so a contig's ``junction_aa`` is the AIRR junction whatever file it came from.

Contigs are encoded as a long table of records
(:mod:`cytearc.metadata.records`), one per contig, identified by its
``sequence_id``. Gene calls and junctions are text, stored once per distinct
value; a contig's ID, its full sequence and its alignments, which nearly every
contig has its own of, are ragged text.
"""

from collections.abc import Mapping
from types import MappingProxyType

from ..metadata.records import (
    FLAG,
    INTEGER,
    NUMBER,
    RAGGED,
    TEXT,
    FieldKind,
    RecordEncoder,
)

__all__ = ["FIELDS", "REQUIRED_FIELDS", "contig_encoder"]

#: Every field that an import keeps, by AIRR name, with its kind, in storage
#: order. ``cell_id`` is not kept: it joins each contig to its cell.
#: ``raw_clonotype_id``, ``raw_consensus_id`` and ``exact_subclonotype_id``
#: are Cell Ranger's own fields.
FIELDS: Mapping[str, FieldKind] = MappingProxyType(
    {
        "sequence_id": RAGGED,
        "locus": TEXT,
        "v_call": TEXT,
        "d_call": TEXT,
        "j_call": TEXT,
        "c_call": TEXT,
        "junction": TEXT,
        "junction_aa": TEXT,
        "productive": FLAG,
        "umi_count": INTEGER,
        "duplicate_count": INTEGER,
        "consensus_count": INTEGER,
        "is_cell": FLAG,
        "high_confidence": FLAG,
        "full_length": FLAG,
        "rev_comp": FLAG,
        "vj_in_frame": FLAG,
        "stop_codon": FLAG,
        "complete_vdj": FLAG,
        "fwr1": TEXT,
        "fwr1_aa": TEXT,
        "cdr1": TEXT,
        "cdr1_aa": TEXT,
        "fwr2": TEXT,
        "fwr2_aa": TEXT,
        "cdr2": TEXT,
        "cdr2_aa": TEXT,
        "fwr3": TEXT,
        "fwr3_aa": TEXT,
        "cdr3": TEXT,
        "cdr3_aa": TEXT,
        "fwr4": TEXT,
        "fwr4_aa": TEXT,
        "np1_length": INTEGER,
        "np2_length": INTEGER,
        "v_sequence_start": INTEGER,
        "v_sequence_end": INTEGER,
        "d_sequence_start": INTEGER,
        "d_sequence_end": INTEGER,
        "j_sequence_start": INTEGER,
        "j_sequence_end": INTEGER,
        "c_sequence_start": INTEGER,
        "c_sequence_end": INTEGER,
        "v_identity": NUMBER,
        "d_identity": NUMBER,
        "j_identity": NUMBER,
        "v_cigar": RAGGED,
        "d_cigar": RAGGED,
        "j_cigar": RAGGED,
        "c_cigar": RAGGED,
        "sequence": RAGGED,
        "sequence_aa": RAGGED,
        "sequence_alignment": RAGGED,
        "germline_alignment": RAGGED,
        "clone_id": TEXT,
        "raw_clonotype_id": TEXT,
        "raw_consensus_id": TEXT,
        "exact_subclonotype_id": TEXT,
    }
)

#: Fields that every contig source must provide. A source without ``locus``
#: has it inferred from its gene calls before encoding.
REQUIRED_FIELDS = ("sequence_id", "locus", "junction_aa", "productive")


def contig_encoder() -> RecordEncoder:
    """Return an encoder of contigs: the fields of ``FIELDS``, by ``sequence_id``."""
    return RecordEncoder(FIELDS, id_field="sequence_id")

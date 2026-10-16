import numpy as np
import pytest

from cytearc.features.genomic.sequence import (
    IndexedFasta,
    encode_sequence,
    gc_content,
    parse_coordinate_ids,
    read_fasta_index,
)

_CHR1 = "ACGTNacgtnGGCC" * 3
_CHR2 = "TTTTAAAACCCCGG"


def _write_fasta(path, *, width: int, newline: str = "\n") -> None:
    lines = []
    for name, sequence in (("chr1", _CHR1), ("chr2 description text", _CHR2)):
        lines.append(f">{name}")
        lines.extend(
            sequence[start : start + width] for start in range(0, len(sequence), width)
        )
    path.write_bytes((newline.join(lines) + newline).encode("ascii"))


@pytest.mark.parametrize("width", [1, 5, 13, 60])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_indexed_fasta_reads_every_interval(tmp_path, width, newline):
    fasta = tmp_path / "genome.fa"
    _write_fasta(fasta, width=width, newline=newline)

    with IndexedFasta(fasta) as genome:
        assert genome.names == ("chr1", "chr2")
        assert genome.length("chr1") == len(_CHR1)
        for name, sequence in (("chr1", _CHR1), ("chr2", _CHR2)):
            for start in range(len(sequence)):
                for end in range(start, len(sequence) + 1):
                    assert genome.fetch(name, start, end) == (
                        sequence[start:end].upper().encode()
                    )


def test_indexed_fasta_uses_samtools_index_when_present(tmp_path, monkeypatch):
    import cytearc.features.genomic.sequence as sequence

    def scan(path):
        raise AssertionError("the FASTA file was scanned despite its index")

    fasta = tmp_path / "genome.fa"
    _write_fasta(fasta, width=5)
    # samtools faidx: name, length, offset, line bases, line width.
    chr2_offset = len(">chr1\n") + len(_CHR1) + len(_CHR1) // 5 + 1
    chr2_offset += len(">chr2 description text\n")
    (tmp_path / "genome.fa.fai").write_text(
        f"chr1\t{len(_CHR1)}\t6\t5\t6\nchr2\t{len(_CHR2)}\t{chr2_offset}\t5\t6\n"
    )

    records = read_fasta_index(tmp_path / "genome.fa.fai")
    assert records["chr2"].offset == chr2_offset
    monkeypatch.setattr(sequence, "_build_fasta_index", scan)
    with IndexedFasta(fasta) as genome:
        assert genome.records == records
        assert genome.fetch("chr2", 3, 9) == _CHR2[3:9].encode()


def test_indexed_fasta_rejects_index_of_another_file(tmp_path):
    fasta = tmp_path / "genome.fa"
    _write_fasta(fasta, width=5)
    index = tmp_path / "other.fai"
    index.write_text("chr1\t100000\t6\t5\t6\n")

    with pytest.raises(ValueError, match="points past the end"):
        IndexedFasta(fasta, index_path=index)
    # An index of 6-base lines fits inside the file, so only a read across a
    # line break finds that the breaks lie elsewhere.
    index.write_text(f"chr1\t{len(_CHR1)}\t6\t6\t7\n")
    with IndexedFasta(fasta, index_path=index) as genome:
        with pytest.raises(ValueError, match="does not match the file"):
            genome.fetch("chr1", 0, 6)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("chr1\t42\t6\t5\n", "line 1: expected 5 fields"),
        ("chr1\t42\t6\tfive\t6\n", "line 1: non-integer field"),
        ("chr1\t42\t6\t5\t4\n", "Invalid FASTA index line 1"),
        ("chr1\t42\t6\t5\t6\nchr1\t42\t6\t5\t6\n", "repeats sequence 'chr1'"),
        # Blank lines are skipped, so this index lists nothing.
        ("\n", "lists no sequences"),
    ],
)
def test_read_fasta_index_rejects_malformed_files(tmp_path, content, match):
    index = tmp_path / "genome.fa.fai"
    index.write_text(content)

    with pytest.raises(ValueError, match=match):
        read_fasta_index(index)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("ACGT\n>chr1\nACGT\n", "does not start with a header"),
        (">\nACGT\n", "has no name"),
        (">chr1\nACGT\n>chr1\nACGT\n", "repeats sequence 'chr1'"),
        (">chr1\nACGT\nAC\nACGT\n", "different widths"),
        # A blank line inside a sequence is a short line too.
        (">chr1\nACGT\n\nACGT\n", "different widths"),
        ("", "contains no sequences"),
    ],
)
def test_indexed_fasta_rejects_malformed_files(tmp_path, content, match):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(content)

    with pytest.raises(ValueError, match=match):
        IndexedFasta(fasta)


def test_indexed_fasta_reads_empty_sequences_but_not_empty_files(tmp_path):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">empty\n>chr1\nACGT\n")

    with IndexedFasta(fasta) as genome:
        assert genome.length("empty") == 0
        assert genome.fetch("empty", 0, 0) == b""
        assert genome.fetch("chr1", 1, 3) == b"CG"
    # An index can describe an empty file, which cannot be memory mapped.
    fasta.write_bytes(b"")
    (tmp_path / "genome.fa.fai").write_text("empty\t0\t7\t1\t1\n")
    with pytest.raises(ValueError, match="empty file"):
        IndexedFasta(fasta)


def test_indexed_fasta_rejects_compressed_input(tmp_path):
    with pytest.raises(ValueError, match="uncompressed"):
        IndexedFasta(tmp_path / "genome.fa.gz")


def test_indexed_fasta_names_missing_sequences_and_bad_intervals(tmp_path):
    fasta = tmp_path / "genome.fa"
    _write_fasta(fasta, width=7)

    with IndexedFasta(fasta) as genome:
        with pytest.raises(KeyError, match="names it 'chr1'"):
            genome.fetch("1", 0, 2)
        with pytest.raises(ValueError, match="past the sequence end"):
            genome.fetch("chr2", 10, len(_CHR2) + 1)
        with pytest.raises(ValueError, match="Invalid interval"):
            genome.fetch("chr2", 5, 4)


def test_encode_sequence_maps_ambiguity_codes_to_unknown():
    np.testing.assert_array_equal(
        encode_sequence("ACGTacgtNRYn-"),
        [0, 1, 2, 3, 0, 1, 2, 3, 4, 4, 4, 4, 4],
    )


def test_gc_content_ignores_ambiguous_bases_like_chromvar(tmp_path):
    fasta = tmp_path / "genome.fa"
    _write_fasta(fasta, width=6)

    with IndexedFasta(fasta) as genome:
        gc = gc_content(genome, ["chr1:0-5", "chr2:8-14", "chr1:4-5"])

    # ACGTN: C and G of the four known bases; CCCCGG: all six; a lone N has
    # no known base.
    np.testing.assert_allclose(gc[:2], [2 / 4, 1.0])
    assert np.isnan(gc[2])


def test_parse_coordinate_ids_keeps_input_order():
    chroms, starts, ends = parse_coordinate_ids(
        ["chr2:30-40", "chr1:5-9", "HLA-A*01:01:0-10"]
    )

    assert chroms.tolist() == ["chr2", "chr1", "HLA-A*01:01"]
    assert starts.tolist() == [30, 5, 0]
    assert ends.tolist() == [40, 9, 10]


@pytest.mark.parametrize("value", ["chr1-5-9", "chr1:9-5", "chr1:5", ":1-2", "c:a-b"])
def test_parse_coordinate_ids_rejects_malformed_ids(value):
    with pytest.raises(ValueError, match="chrom:start-end"):
        parse_coordinate_ids([value])

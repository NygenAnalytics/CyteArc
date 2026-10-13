import itertools

import numpy as np
import pytest

from cytearc.features.genomic.motifs import (
    Motif,
    log_odds_matrix,
    read_jaspar_motifs,
    read_meme_motifs,
    reverse_complement_matrix,
    scan_motif_matches,
    score_threshold,
)
from cytearc.features.genomic.sequence import encode_sequence

_JASPAR = """>MA0004.1\tArnt
A  [     4     19      0      0      0      0 ]
C  [    16      0     20      0      0      0 ]
G  [     0      1      0     20      0     20 ]
T  [     0      0      0      0     20      0 ]
>MA0006.1 Ahr::Arnt
A [ 3 0 0 0 0 0 ]
C [ 8 0 23 0 0 0 ]
G [ 2 23 0 23 0 24 ]
T [ 11 1 1 1 24 0 ]
"""

_MEME = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF MA0004.1 Arnt
letter-probability matrix: alength= 4 w= 3 nsites= 20 E= 0
  0.200000  0.800000  0.000000  0.000000
  0.950000  0.000000  0.050000  0.000000
  0.000000  1.000000  0.000000  0.000000
URL http://jaspar.genereg.net/matrix/MA0004.1

MOTIF MA0006.1
letter-probability matrix: alength= 4 w= 2 nsites= 24 E= 0
 0.125 0.333 0.083 0.458
 0.000 0.000 0.958 0.042
"""


def test_read_jaspar_motifs(tmp_path):
    path = tmp_path / "motifs.jaspar"
    path.write_text(_JASPAR)

    motifs = read_jaspar_motifs(path)

    assert [(motif.motifId, motif.name, motif.width) for motif in motifs] == [
        ("MA0004.1", "Arnt", 6),
        ("MA0006.1", "Ahr::Arnt", 6),
    ]
    np.testing.assert_array_equal(motifs[0].counts[0], [4, 16, 0, 0])
    np.testing.assert_array_equal(motifs[1].counts[:, 3], [11, 1, 1, 1, 24, 0])


def test_read_jaspar_rejects_missing_rows(tmp_path):
    path = tmp_path / "bad.jaspar"
    path.write_text(">M1 X\nA [1 2]\nC [1 2]\nG [1 2]\n")

    with pytest.raises(ValueError, match="needs A, C, G, and T rows"):
        read_jaspar_motifs(path)


def test_read_meme_motifs(tmp_path):
    path = tmp_path / "motifs.meme"
    path.write_text(_MEME)

    motifs = read_meme_motifs(path)

    assert [(motif.motifId, motif.name, motif.width) for motif in motifs] == [
        ("MA0004.1", "Arnt", 3),
        ("MA0006.1", "MA0006.1", 2),
    ]
    np.testing.assert_allclose(motifs[1].counts[1], [0.0, 0.0, 0.958, 0.042])


def test_motif_rejects_empty_positions():
    with pytest.raises(ValueError, match="empty position"):
        Motif("M", "M", np.array([[1, 0, 0, 0], [0, 0, 0, 0]]))


def test_log_odds_matrix_matches_moods_formula():
    counts = np.array([[4.0, 16.0, 0.0, 0.0]])
    bg = np.array([0.3, 0.2, 0.2, 0.3])

    natural = log_odds_matrix(counts, bg, pseudocount=0.8, log_base=np.e)
    bits = log_odds_matrix(counts, bg, pseudocount=0.8)

    expected = np.log((counts + 0.8 * bg) / (20.0 + 0.8)) - np.log(bg)
    np.testing.assert_allclose(natural, expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(bits, expected / np.log(2.0))
    with pytest.raises(ValueError, match="infinite"):
        log_odds_matrix(counts, bg, pseudocount=0.0)
    with pytest.raises(ValueError, match="sum to 1"):
        log_odds_matrix(counts, [1, 1, 1, 1], pseudocount=0.8)


def test_log_odds_matrix_with_even_background_is_tfbstools_to_pwm():
    counts = np.array([[4.0, 16.0, 0.0, 0.0], [19.0, 0.0, 1.0, 0.0]])

    scores = log_odds_matrix(counts, np.full(4, 0.25), pseudocount=0.8)

    # TFBSTools toPWM: log2(((n + 0.8 * 0.25) / (N + 0.8)) / 0.25).
    expected = np.log2(
        ((counts + 0.2) / (counts.sum(axis=1, keepdims=True) + 0.8)) / 0.25
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-15, atol=1e-15)


def _brute_force_tail(matrix, bg, threshold):
    total = 0.0
    for bases in itertools.product(range(4), repeat=matrix.shape[0]):
        score = sum(matrix[row, base] for row, base in enumerate(bases))
        if score >= threshold - 1e-9:
            total += np.prod([bg[base] for base in bases])
    return total


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_score_threshold_matches_exhaustive_enumeration(seed):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 20, size=(5, 4)).astype(float)
    bg = np.array([0.3, 0.2, 0.2, 0.3])
    matrix = log_odds_matrix(counts, bg, pseudocount=1.0, log_base=np.e)
    scores = sorted(
        {
            sum(matrix[row, base] for row, base in enumerate(bases))
            for bases in itertools.product(range(4), repeat=5)
        }
    )

    precision = 1e6
    # Rounding each position to the grid moves a score by at most this much.
    band = matrix.shape[0] / precision
    for p_value in (0.5, 0.05, 1e-2, 1e-3):
        threshold = score_threshold(matrix, bg, p_value, precision=precision)
        # The tail just above the rounding band respects the p-value, and the
        # next attainable score below the band would not.
        assert _brute_force_tail(matrix, bg, threshold + band) <= p_value + 1e-12
        lower = [score for score in scores if score < threshold - band]
        if lower:
            assert _brute_force_tail(matrix, bg, lower[-1]) > p_value


def test_score_threshold_sits_below_the_best_score_when_it_is_too_common():
    matrix = np.array([[1.0, 0.5, 0.0, 0.0], [2.0, 2.0, -1.0, 0.0]])

    # The best score 3.0 has probability 0.125, above the p-value, so MOODS
    # places the threshold half its smallest column gap below it. MOODS skips
    # a value equal to the column maximum, so the second column's gap is 2.0
    # and the first column's 0.5 is the smallest.
    assert score_threshold(matrix, [0.25] * 4, 0.1) == 2.75
    flat = np.zeros((1, 4))
    assert score_threshold(flat, [0.25] * 4, 0.1) == -np.inf


def test_score_threshold_follows_moods_discretization():
    # Two columns whose scores sit on a coarse grid make the DP exact.
    matrix = np.array([[1.0, 0.0, 0.0, -1.0], [1.0, 0.5, 0.0, -1.0]])
    bg = [0.25] * 4

    # P(score >= 2.0) = 1/16, P(score >= 1.5) = 2/16, P(score >= 1.0) = 5/16.
    # MOODS returns one grid unit (1/2000) above the highest level whose tail
    # exceeds the p-value, which admits the same sites as the next level up.
    assert score_threshold(matrix, bg, 0.07) == 1.5005
    assert score_threshold(matrix, bg, 0.13) == 1.0005
    assert score_threshold(matrix, bg, 0.2) == 1.0005


def test_reverse_complement_matrix_scores_the_other_strand():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(4, 4))
    sequence = encode_sequence("ACGG")
    reverse = encode_sequence("CCGT")

    forward_score = sum(matrix[row, code] for row, code in enumerate(reverse))
    rc = reverse_complement_matrix(matrix)
    rc_score = sum(rc[row, code] for row, code in enumerate(sequence))

    assert rc_score == pytest.approx(forward_score)


def _brute_force_matches(sequences, matrices, thresholds):
    hits = np.zeros((len(sequences), len(matrices)), dtype=bool)
    for row, sequence in enumerate(sequences):
        for column, (matrix, threshold) in enumerate(
            zip(matrices, thresholds, strict=True)
        ):
            rc = reverse_complement_matrix(matrix)
            width = matrix.shape[0]
            for start in range(len(sequence) - width + 1):
                window = sequence[start : start + width]
                if np.any(window > 3):
                    continue
                for scorer in (matrix, rc):
                    score = scorer[np.arange(width), window].sum()
                    if score >= threshold:
                        hits[row, column] = True
    return hits


@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_scan_motif_matches_agrees_with_brute_force(batch_size):
    rng = np.random.default_rng(7)
    alphabet = np.array(list("ACGTN"))
    sequences = []
    for length in rng.integers(0, 40, size=25):
        letters = rng.choice(alphabet, size=length, p=[0.24, 0.24, 0.24, 0.24, 0.04])
        sequences.append(encode_sequence("".join(letters)))
    bg = np.full(4, 0.25)
    matrices = [
        log_odds_matrix(rng.integers(0, 10, size=(width, 4)), bg, pseudocount=0.5)
        for width in (3, 5, 6, 8)
    ]
    assert all(np.isfinite(score_threshold(matrix, bg, 0.02)) for matrix in matrices)
    thresholds = [score_threshold(matrix, bg, 0.02) for matrix in matrices]

    hits = scan_motif_matches(sequences, matrices, thresholds, batch_size=batch_size)

    assert hits.shape == (25, 4)
    assert hits.dtype == bool
    np.testing.assert_array_equal(
        hits.toarray(), _brute_force_matches(sequences, matrices, thresholds)
    )


def test_scan_motif_matches_ignores_windows_with_unknown_bases():
    matrix = np.log(np.array([[0.97, 0.01, 0.01, 0.01]] * 3) / 0.25)

    hits = scan_motif_matches(
        [encode_sequence("AAN"), encode_sequence("TTT"), encode_sequence("NAAA")],
        [matrix],
        [matrix.max(axis=1).sum() - 1e-9],
    )

    # TTT is the reverse complement of AAA.
    np.testing.assert_array_equal(hits.toarray()[:, 0], [False, True, True])


def test_match_motifs_finds_planted_sites_on_both_strands(tmp_path):
    from cytearc.features.genomic.motifs import match_motifs, nucleotide_frequencies
    from cytearc.features.genomic.sequence import IndexedFasta

    rng = np.random.default_rng(11)
    site = "ATGCAAATGA"
    background = "".join(rng.choice(list("ACGT"), size=2000))
    reverse = site[::-1].translate(str.maketrans("ACGT", "TGCA"))
    assert reverse != site
    genome = (
        background[:100]
        + site  # forward site in region 0 (100..300)
        + background[110:500]
        + reverse  # reverse-complement site in region 1 (500..700)
        + background[510:900]
        + site[:5]
        + "N"
        + site[5:]  # split by N in region 2 (900..1100)
        + background[911:2000]
    )
    fasta = tmp_path / "genome.fa"
    fasta.write_text(
        ">chr1\n"
        + "\n".join(genome[i : i + 70] for i in range(0, len(genome), 70))
        + "\n"
    )
    counts = (
        np.array(
            [[0.0 if base != letter else 20.0 for base in "ACGT"] for letter in site]
        )
        + 0.2
    )
    motif = Motif("M1", "POU", counts)

    with IndexedFasta(fasta) as fasta_genome:
        network = match_motifs(
            ["chr1", "chr1", "chr1"],
            [0, 450, 880],
            [300, 700, 1100],
            fasta_genome,
            [motif],
            background="even",
        )
        sequences = [
            fasta_genome.encode("chr1", s, e)
            for s, e in ((0, 300), (450, 700), (880, 1100))
        ]

    assert network["target"].tolist() == ["chr1:0-300", "chr1:450-700"]
    assert set(network["source"]) == {"M1"} and set(network["motif_name"]) == {"POU"}
    frequencies = nucleotide_frequencies(sequences)
    assert frequencies.sum() == pytest.approx(1.0)
    assert np.all(frequencies > 0.2)

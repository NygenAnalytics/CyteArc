import itertools

import numpy as np
import pytest

from cytearc.features.genomic import motifs as motif_module
from cytearc.features.genomic.motifs import (
    Motif,
    log_odds_matrix,
    match_motifs,
    nucleotide_frequencies,
    read_jaspar_motifs,
    read_meme_motifs,
    reverse_complement_matrix,
    scan_motif_matches,
    score_threshold,
)
from cytearc.features.genomic.sequence import IndexedFasta, encode_sequence

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

# Blank lines may surround the rows of a matrix, as in the second motif.
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


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (">M1 X\nA [1 2]\nC [1 2]\nG [1 2]\n", "needs A, C, G, and T rows"),
        (">M1\nA [1 2]\nC [1 2]\nG [1 2]\nT [1]\n", "rows of different lengths"),
        (">\nA [1]\n", "line 1 has no ID"),
        ("A [1 2]\n", "Malformed JASPAR line 1"),
        (">M1\nA [1]\nA [1]\n", "repeats row A"),
        (">M1\nA [1 x]\n", "Malformed JASPAR line 2"),
        ("", "contains no motifs"),
    ],
)
def test_read_jaspar_rejects_malformed_files(tmp_path, content, match):
    path = tmp_path / "bad.jaspar"
    path.write_text(content)

    with pytest.raises(ValueError, match=match):
        read_jaspar_motifs(path)


def test_read_meme_motifs(tmp_path):
    path = tmp_path / "motifs.meme"
    path.write_text(_MEME)

    motifs = read_meme_motifs(path)

    assert [(motif.motifId, motif.name, motif.width) for motif in motifs] == [
        ("MA0004.1", "Arnt", 3),
        ("MA0006.1", "MA0006.1", 2),
    ]
    # Probabilities become counts of the matrix's nsites (24 here).
    np.testing.assert_allclose(
        motifs[1].counts[1], 24 * np.array([0.0, 0.0, 0.958, 0.042])
    )


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("ALPHABET= ACDEFGHIKLMNPQRSTVWY\nMOTIF M1\n", "Only DNA"),
        ("MOTIF\n", "line 1 has no identifier"),
        # The next motif, or the end of the file, comes before a matrix.
        ("MOTIF M1\nURL x\nMOTIF M2\n", "'M1' has no probability matrix"),
        ("MOTIF M1\n", "'M1' has no probability matrix"),
        ("MOTIF M1\nletter-probability matrix:\n0.5 0.5 0.0\n", "without 4 values"),
        (
            "MOTIF M1\nletter-probability matrix: w= 2\n0.25 0.25 0.25 0.25\n",
            "declares w=2 but has 1 rows",
        ),
        ("MEME version 4\n", "contains no motifs"),
    ],
)
def test_read_meme_rejects_malformed_files(tmp_path, content, match):
    path = tmp_path / "bad.meme"
    path.write_text(content)

    with pytest.raises(ValueError, match=match):
        read_meme_motifs(path)


@pytest.mark.parametrize(
    ("motif_id", "counts", "match"),
    [
        ("M", np.ones((2, 3)), "non-empty"),
        ("M", [[-1.0, 1.0, 1.0, 1.0]], "negative or non-finite"),
        ("M", [[1, 0, 0, 0], [0, 0, 0, 0]], "empty position"),
        ("", [[1.0, 1.0, 1.0, 1.0]], "identifiers must be non-empty"),
    ],
)
def test_motif_rejects_invalid_counts_and_ids(motif_id, counts, match):
    with pytest.raises(ValueError, match=match):
        Motif(motif_id, "M", counts)


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


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"counts": np.ones((2, 3))}, "counts must be"),
        ({"background": [0.5, 0.5, 0.0, 0.0]}, "four positive"),
        ({"pseudocount": -1.0}, "pseudocount"),
        ({"log_base": 1.0}, "log_base"),
    ],
)
def test_log_odds_matrix_rejects_invalid_arguments(override, match):
    arguments = {
        "counts": np.ones((2, 4)),
        "background": [0.25] * 4,
        "pseudocount": 0.8,
    }

    with pytest.raises(ValueError, match=match):
        log_odds_matrix(**(arguments | override))


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

    precision = 1e4
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


def test_score_threshold_admits_every_score_when_p_exceeds_the_background_mass():
    # A background may sum to 1 within 1e-6, so no tail exceeds a p-value
    # above its total mass and the threshold sits one below the lowest score.
    matrix = np.array([[1.0, 0.0, 0.0, -1.0]])

    assert score_threshold(matrix, [0.25, 0.25, 0.25, 0.2499995], 0.9999999) == -2.0


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"matrix": np.zeros((0, 4))}, "non-empty"),
        ({"p_value": 1.0}, "p_value"),
        ({"precision": 0.0}, "precision"),
    ],
)
def test_score_threshold_rejects_invalid_arguments(override, match):
    arguments = {"matrix": np.zeros((2, 4)), "background": [0.25] * 4, "p_value": 0.1}

    with pytest.raises(ValueError, match=match):
        score_threshold(**(arguments | override))


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
        for column, (matrix, (forward, reverse)) in enumerate(
            zip(matrices, thresholds, strict=True)
        ):
            rc = reverse_complement_matrix(matrix)
            width = matrix.shape[0]
            for start in range(len(sequence) - width + 1):
                window = sequence[start : start + width]
                if np.any(window > 3):
                    continue
                for scorer, threshold in ((matrix, forward), (rc, reverse)):
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
    # A background whose A and T frequencies differ gives the strands
    # different thresholds.
    bg = np.array([0.35, 0.2, 0.2, 0.25])
    matrices = [
        log_odds_matrix(rng.integers(0, 10, size=(width, 4)), bg, pseudocount=0.5)
        for width in (3, 5, 6, 8)
    ]
    thresholds = motif_module._strand_thresholds(matrices, bg, 0.02)
    assert np.isfinite(thresholds).all()
    assert (thresholds[:, 0] != thresholds[:, 1]).any()

    hits = scan_motif_matches(sequences, matrices, thresholds, batch_size=batch_size)

    assert hits.shape == (25, 4)
    assert hits.dtype == bool
    np.testing.assert_array_equal(
        hits.toarray(), _brute_force_matches(sequences, matrices, thresholds)
    )


@pytest.mark.parametrize("kernels", ["jit", "python"])
def test_scan_motif_matches_ignores_windows_with_unknown_bases(kernels, monkeypatch):
    if kernels == "python":
        # Line coverage cannot see compiled code, so also run the Python
        # sources of the scanning kernels on the same inputs.
        for name in ("_suffix_maxima", "_strand_hit", "_scan_block"):
            kernel = getattr(motif_module, name).py_func
            monkeypatch.setattr(motif_module, name, kernel)
    matrix = np.log(np.array([[0.97, 0.01, 0.01, 0.01]] * 3) / 0.25)

    hits = scan_motif_matches(
        [encode_sequence(bases) for bases in ("AAN", "TTT", "NAAA", "AA")],
        [matrix],
        [[matrix.max(axis=1).sum() - 1e-9] * 2],
    )

    # TTT is the reverse complement of AAA, and AA is shorter than the motif.
    np.testing.assert_array_equal(hits.toarray()[:, 0], [False, True, True, False])


@pytest.mark.parametrize(
    "background",
    [[0.25, 0.25, 0.25, 0.25], [0.4, 0.1, 0.2, 0.3], [0.45, 0.05, 0.3, 0.2]],
)
def test_each_strand_threshold_keeps_its_false_positive_rate(background):
    background = np.asarray(background)
    counts = np.array(
        [
            [8, 1, 1, 0],
            [9, 0, 1, 0],
            [0, 1, 9, 0],
            [7, 2, 0, 1],
            [8, 0, 0, 2],
            [0, 0, 1, 9],
        ]
    )
    matrix = log_odds_matrix(counts, background, pseudocount=0.8)
    reverse = reverse_complement_matrix(matrix)

    ((forward_threshold, reverse_threshold),) = motif_module._strand_thresholds(
        [matrix], background, 0.01
    )

    # The probability under the background of a window scoring at least each
    # threshold, summed over every possible window.
    def tail(scorer, threshold):
        return sum(
            np.prod(background[list(word)])
            for word in itertools.product(range(4), repeat=6)
            if scorer[np.arange(6), word].sum() >= threshold
        )

    # Each strand keeps the requested rate, up to MOODS's rounding of scores.
    assert tail(matrix, forward_threshold) == pytest.approx(0.01, rel=0.05)
    assert tail(reverse, reverse_threshold) == pytest.approx(0.01, rel=0.05)
    # Only where complementary bases share a frequency would the forward
    # threshold also serve the reverse strand.
    symmetric = background[0] == background[3] and background[1] == background[2]
    reused = tail(reverse, forward_threshold)
    assert (reused == pytest.approx(0.01, rel=0.05)) == symmetric


def test_scan_motif_matches_of_no_sequences_is_empty():
    hits = scan_motif_matches([], [np.zeros((2, 4))], [[0.0, 0.0]])

    assert hits.shape == (0, 1)
    assert hits.dtype == bool


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"matrices": [], "thresholds": []}, "At least one motif matrix"),
        ({"thresholds": [0.0]}, "two strand thresholds"),
        ({"thresholds": [[np.nan, 0.0]]}, "NaN"),
        ({"matrices": [np.zeros((0, 4))]}, "non-empty"),
        ({"matrices": [np.full((2, 4), np.inf)]}, "finite"),
        ({"sequences": [np.zeros(3, dtype=np.int64)]}, "uint8"),
    ],
)
def test_scan_motif_matches_rejects_invalid_inputs(override, match):
    arguments = {
        "sequences": [encode_sequence("ACGT")],
        "matrices": [np.zeros((2, 4))],
        "thresholds": [[0.0, 0.0]],
    }

    with pytest.raises(ValueError, match=match):
        scan_motif_matches(**(arguments | override))


@pytest.mark.parametrize(
    "bg",
    ["subject", "even", [0.3, 0.2, 0.2, 0.3]],
    ids=["subject", "even", "frequencies"],
)
def test_match_motifs_finds_planted_sites_on_both_strands(tmp_path, bg):
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
            ["chr1:0-300", "chr1:450-700", "chr1:880-1100"],
            fasta_genome,
            [motif],
            background=bg,
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


@pytest.mark.parametrize(
    ("region", "motif_ids", "background", "match"),
    [
        ("chr1:0-4", [], "even", "At least one motif"),
        ("chr1:0-4", ["M1", "M1"], "even", "must be unique"),
        ("chr1:0-4", ["M1"], "uniform", "'subject', 'even', or frequencies"),
        # A region of Ns has no subject background.
        ("chr1:4-8", ["M1"], "subject", "no A, C, G, or T base"),
    ],
)
def test_match_motifs_rejects_invalid_inputs(
    tmp_path, region, motif_ids, background, match
):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nACGTNNNN\n")
    motifs = [Motif(motif_id, motif_id, np.eye(4)) for motif_id in motif_ids]

    with IndexedFasta(fasta) as genome, pytest.raises(ValueError, match=match):
        match_motifs([region], genome, motifs, background=background)

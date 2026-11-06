"""What an imported matrix holds, decided from its values, and the record that keeps it.

Each normalization is computed here from its published formula on seeded
counts, so the decision expected for it is known without the code under test.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import zarr

from cytearc.storage.schema import (
    create_zarr_count_assay,
    merged_normalized_values,
    normalized_values_record,
    read_normalized_values,
)
from cytearc.utils.count_values import CountValueRange
from cytearc.utils.matrix_values import (
    DETECTION_CELLS,
    MatrixValues,
    decide_matrix_values,
    fitting_readings,
    linear_coo,
    linear_values,
    matrix_values_argument,
)


def _counts(seed: int = 0, cells: int = 300, genes: int = 400) -> sp.csr_matrix:
    """Poisson counts of cells of varied depth, most of whose genes are rare."""
    rng = np.random.default_rng(seed)
    means = rng.gamma(0.3, 2.0, size=genes)
    depth = rng.lognormal(0.0, 0.5, size=cells)
    counts = rng.poisson(np.outer(depth, means)).astype(np.float64)
    return sp.csr_matrix(counts[counts.sum(axis=1) > 0])


def _scaled(counts: sp.csr_matrix, factors: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(sp.diags(factors) @ counts)


def _applied(matrix: sp.csr_matrix, function, dtype=np.float64) -> sp.csr_matrix:
    out = matrix.astype(np.float64).tocsr(copy=True)
    out.data = function(out.data)
    out.eliminate_zeros()
    return out.astype(dtype)


def _cp10k(counts: sp.csr_matrix) -> sp.csr_matrix:
    return _scaled(counts, 1e4 / np.asarray(counts.sum(axis=1)).ravel())


def _size_factors(counts: sp.csr_matrix) -> sp.csr_matrix:
    """Counts divided by size factors that are not proportional to the totals."""
    totals = np.asarray(counts.sum(axis=1)).ravel()
    noise = np.exp(np.random.default_rng(3).normal(0.0, 0.3, size=totals.size))
    factors = totals / totals.mean() * noise
    return _scaled(counts, factors.mean() / factors)


def _tpm(counts: sp.csr_matrix) -> sp.csr_matrix:
    lengths = np.exp(np.random.default_rng(4).normal(7.5, 0.6, size=counts.shape[1]))
    per_kilobase = sp.csr_matrix(counts.multiply(1e3 / lengths[None, :]))
    return _scaled(per_kilobase, 1e6 / np.asarray(per_kilobase.sum(axis=1)).ravel())


def _top_genes(values: sp.csr_matrix, n: int) -> sp.csr_matrix:
    dense = values.toarray()
    return sp.csr_matrix(dense[:, np.argsort(dense.var(axis=0))[::-1][:n]])


def _per_feature_clr(counts: sp.csr_matrix) -> sp.csr_matrix:
    scale = np.exp(np.log1p(counts.toarray()).mean(axis=0))
    scale[scale == 0] = 1
    return _applied(sp.csr_matrix(counts.multiply(1.0 / scale[None, :])), np.log1p)


def _range(matrix: sp.csr_matrix) -> CountValueRange:
    value_range = CountValueRange()
    value_range.update(matrix.data)
    return value_range


def _decide(matrix: sp.csr_matrix, declared=None) -> MatrixValues:
    return decide_matrix_values(
        declared,
        _range(matrix),
        lambda: matrix[:DETECTION_CELLS],
        assay="RNA",
    )


_COUNTS = _counts()
_LOG_CP10K = _applied(_cp10k(_COUNTS), np.log1p, np.float32)
# Each normalization, written as its tool computes it, and what it holds.
_DECIDED = {
    "counts": (_COUNTS, "counts", None),
    "scanpy normalize_total": (_cp10k(_COUNTS).astype(np.float32), "normalized", 1e4),
    "scanpy normalize_total and log1p": (_LOG_CP10K, "log_normalized", 1e4),
    "Seurat LogNormalize": (_applied(_cp10k(_COUNTS), np.log1p), "log_normalized", 1e4),
    "scran-like size factors": (_size_factors(_COUNTS), "normalized", None),
    "size factors and log2(x + 1), as scater writes them": (
        _applied(_size_factors(_COUNTS), lambda v: np.log2(v + 1)),
        "log2_normalized",
        None,
    ),
    "log1p values of the 100 most variable genes": (
        _top_genes(_LOG_CP10K, 100),
        "log_normalized",
        None,
    ),
    "TPM": (_tpm(_COUNTS), "normalized", 1e6),
    "log2(TPM + 1)": (
        _applied(_tpm(_COUNTS), lambda v: np.log2(v + 1)),
        "log2_normalized",
        1e6,
    ),
    "log1p values rounded to two decimals": (
        _applied(_LOG_CP10K, lambda v: np.round(v, 2)),
        "log_normalized",
        1e4,
    ),
    "log1p values stored as float16": (
        _applied(_LOG_CP10K, lambda v: v.astype(np.float16).astype(np.float64)),
        "log_normalized",
        1e4,
    ),
}
_UNDECIDED = {
    "square roots of normalized values": _applied(_cp10k(_COUNTS), np.sqrt),
    "base-10 logs": _applied(_cp10k(_COUNTS), lambda v: np.log10(v + 1)),
    "fractional counts, such as EM estimates": _applied(
        _COUNTS,
        lambda v: v + np.random.default_rng(5).uniform(-0.45, 0.45, size=v.size),
    ),
    "CLR across cells, as Seurat's margin 1 computes it": _per_feature_clr(_COUNTS),
}


@pytest.mark.parametrize("name", list(_DECIDED))
def test_values_of_each_normalization_are_decided(name):
    matrix, kind, total = _DECIDED[name]

    decided = _decide(matrix)

    assert decided.kind == kind
    if total is None:
        assert decided.cellTotal is None
    else:
        assert decided.cellTotal == pytest.approx(total, rel=0.01)


@pytest.mark.parametrize("name", list(_UNDECIDED))
def test_values_that_show_no_reading_ask_for_a_declaration(name):
    with pytest.raises(ValueError, match="values='normalized'.*values='counts'"):
        _decide(_UNDECIDED[name])


@pytest.mark.parametrize("name", list(_UNDECIDED))
def test_a_declaration_decides_values_that_show_no_reading(name):
    assert _decide(_UNDECIDED[name], "log_normalized") == MatrixValues("log_normalized")


def test_scaled_values_are_refused():
    dense = _LOG_CP10K.toarray()
    scaled = sp.csr_matrix((dense - dense.mean(axis=0)) / (dense.std(axis=0) + 1e-9))

    for declared in (None, "counts", "normalized"):
        with pytest.raises(ValueError, match="negative values"):
            _decide(scaled, declared)


def test_a_negative_value_after_the_sampled_cells_is_refused():
    values = _cp10k(_COUNTS).tolil()
    values[DETECTION_CELLS + 50, 3] = -0.5
    values = values.tocsr()
    assert values[:DETECTION_CELLS].min() >= 0

    with pytest.raises(ValueError, match="negative values"):
        _decide(values)


def test_cells_that_hold_no_value_in_the_sample_ask_for_a_declaration():
    values = _cp10k(_COUNTS).tolil()
    values[:DETECTION_CELLS] = 0
    values = values.tocsr()

    assert fitting_readings(values[:DETECTION_CELLS]) == {}
    with pytest.raises(ValueError, match="first 100 cells do not show"):
        _decide(values)


def test_counts_in_the_sample_beside_normalized_values_ask_for_a_declaration():
    # The sampled counts are whole, but the matrix is not, and counts whose
    # smallest value is above 1 are not whole multiples of it.
    mixed = sp.csr_matrix(
        sp.vstack([_COUNTS[:DETECTION_CELLS], _cp10k(_COUNTS[DETECTION_CELLS:])])
    )

    with pytest.raises(ValueError, match="first 100 cells hold only whole numbers"):
        _decide(mixed)
    with pytest.raises(ValueError, match="declared values='log_normalized'"):
        _decide(mixed, "log_normalized")
    # Library-size normalization cancels a factor of one per cell as well as
    # any other, so the declared mix normalizes consistently.
    assert _decide(mixed, "normalized") == MatrixValues("normalized")


def test_a_log_declared_for_a_whole_sample_is_refused():
    # Counts whose smallest value is 2 fit no reading, so only the whole
    # sample tells that they are not logs.
    counts = sp.csr_matrix(_COUNTS.toarray() * 2 + (_COUNTS.toarray() > 0) * 1.0)
    later = _applied(_cp10k(_COUNTS), np.log1p)
    mixed = sp.csr_matrix(
        sp.vstack([counts[:DETECTION_CELLS], later[DETECTION_CELLS:]])
    )
    assert fitting_readings(mixed[:DETECTION_CELLS]) == {}

    with pytest.raises(ValueError, match="hold only whole numbers, which logs"):
        _decide(mixed, "log_normalized")


def test_declarations_must_be_among_the_readings_that_fit():
    with pytest.raises(ValueError, match="fit only values='log_normalized'"):
        _decide(_LOG_CP10K, "normalized")
    with pytest.raises(ValueError, match="fit only values='normalized'"):
        _decide(_cp10k(_COUNTS), "log2_normalized")
    assert _decide(_LOG_CP10K, "log_normalized") == MatrixValues(
        "log_normalized", pytest.approx(1e4, rel=0.01)
    )


def test_fractional_counts_import_as_counts_when_declared():
    assert _decide(_UNDECIDED["fractional counts, such as EM estimates"], "counts") == (
        MatrixValues("counts")
    )
    # Counts with a few fractional values, as multimapping reads split, are
    # undecided, and halved counts read as normalized; a declaration of
    # counts imports both as counts.
    split = _COUNTS.copy()
    split.data[::500] += 0.5
    with pytest.raises(ValueError, match="values='counts' for fractional counts"):
        _decide(split)
    assert _decide(split, "counts") == MatrixValues("counts")
    halved = _applied(_COUNTS, lambda v: v / 2)
    assert _decide(halved).kind == "normalized"
    assert _decide(halved, "counts") == MatrixValues("counts")


@pytest.mark.parametrize(
    ("matrix", "kind"),
    [
        (_LOG_CP10K, "log_normalized"),
        (_applied(_size_factors(_COUNTS), lambda v: np.log2(v + 1)), "log2_normalized"),
    ],
)
def test_a_declaration_of_counts_is_refused_for_logs(matrix, kind):
    with pytest.raises(
        ValueError, match=f"declared values='counts'.*fit only values='{kind}'"
    ):
        _decide(matrix, "counts")


def test_whole_numbers_are_counts_unless_declared_normalized():
    rounded = _applied(_cp10k(_COUNTS), np.rint)

    assert _decide(rounded) == MatrixValues("counts")
    assert _decide(rounded, "normalized") == MatrixValues("normalized")
    for declared in ("log_normalized", "log2_normalized"):
        with pytest.raises(ValueError, match="every value is a whole number"):
            _decide(rounded, declared)


def test_the_sample_is_read_only_when_a_value_is_not_whole():
    def unread():
        raise AssertionError("whole numbers need no sample")

    assert decide_matrix_values(None, _range(_COUNTS), unread, assay="RNA") == (
        MatrixValues("counts")
    )


def test_readings_report_the_shared_total_or_none():
    readings = fitting_readings(_LOG_CP10K[:DETECTION_CELLS])
    assert list(readings) == ["log_normalized"]
    assert readings["log_normalized"] == pytest.approx(1e4, rel=1e-5)
    assert fitting_readings(_size_factors(_COUNTS)[:DETECTION_CELLS]) == {
        "normalized": None
    }


def test_values_argument_takes_none_or_a_kind():
    assert matrix_values_argument(None) is None
    assert matrix_values_argument("log2_normalized") == "log2_normalized"
    for bad in ("log", "Counts", 1, True):
        with pytest.raises(ValueError, match="values must be None or one of"):
            matrix_values_argument(bad)


def test_linear_values_undo_each_log_in_float64():
    linear = np.array([0.0, 1.0, 37.5, 9999.0])

    np.testing.assert_allclose(
        linear_values(np.log1p(linear), "log_normalized", np.float64, assay="RNA"),
        linear,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        linear_values(np.log2(linear + 1), "log2_normalized", np.float64, assay="RNA"),
        linear,
        rtol=1e-12,
    )
    values = np.array([1.5, 2.0])
    assert linear_values(values, "normalized", np.float32, assay="RNA") is values


def test_linear_values_refuse_a_value_too_large_to_be_a_log():
    with pytest.raises(ValueError, match="1000.0 of assay 'RNA' overflows float32"):
        linear_values(
            np.array([2.0, 1000.0]), "log_normalized", np.float32, assay="RNA"
        )


def test_duplicate_coordinates_are_summed_before_the_log_is_undone():
    block = sp.coo_matrix(([0.5, 0.25, 1.0], ([0, 0, 1], [2, 2, 0])), shape=(2, 3))

    linear = linear_coo(block, "log_normalized", np.float64, assay="RNA").toarray()

    np.testing.assert_allclose(
        linear, [[0.0, 0.0, np.expm1(0.75)], [np.expm1(1.0), 0.0, 0.0]], rtol=1e-12
    )
    assert linear_coo(block, "counts", np.float64, assay="RNA") is block


def test_value_range_keeps_recording_negative_values_after_a_fraction():
    value_range = CountValueRange()
    value_range.update(np.array([0.5, 2.0]))
    assert not value_range.negative
    value_range.update(np.array([3.0, -1.0]))

    assert value_range == CountValueRange(integral=False, maximum=0, negative=True)


def test_the_record_holds_the_import_and_round_trips():
    record = normalized_values_record(MatrixValues("log_normalized", 2878.75))

    assert record == {"importedAs": "log_normalized", "cellTotal": 2878.75}
    assert normalized_values_record(MatrixValues("counts")) is None
    assert read_normalized_values({"normalizedValues": record}, assay="RNA") == record
    assert read_normalized_values({}, assay="RNA") is None


@pytest.mark.parametrize(
    "record",
    [
        "normalized",
        {"importedAs": "normalized"},
        {"importedAs": "counts", "cellTotal": None},
        {"importedAs": "normalized", "cellTotal": 0},
        {"importedAs": "normalized", "cellTotal": True},
        {"importedAs": "normalized", "cellTotal": float("nan")},
        {"importedAs": "normalized", "cellTotal": None, "extra": 1},
    ],
)
def test_a_malformed_record_is_refused(record):
    with pytest.raises(ValueError, match="normalizedValues attribute"):
        read_normalized_values({"normalizedValues": record}, assay="RNA")
    with pytest.raises(ValueError, match="normalizedValues attribute"):
        create_zarr_count_assay(
            zarr.group(),
            "RNA",
            None,
            2,
            ["a"],
            ["a"],
            np.float32,
            normalized_values=record,
        )


def test_combined_records_keep_the_kind_every_source_shares():
    log = {"importedAs": "log_normalized", "cellTotal": 1e4}

    # A combined assay's features can differ from a source's, so no cell total.
    assert merged_normalized_values([None, None]) is None
    assert merged_normalized_values([log, dict(log)]) == {
        "importedAs": "log_normalized",
        "cellTotal": None,
    }
    assert merged_normalized_values([log, None]) == {
        "importedAs": None,
        "cellTotal": None,
    }
    assert merged_normalized_values(
        [log, {"importedAs": "log2_normalized", "cellTotal": 5e3}]
    ) == {"importedAs": None, "cellTotal": None}


def _one_cell(values) -> sp.csr_matrix:
    return sp.csr_matrix(np.asarray(values, dtype=np.float64)[None, :])


def test_whole_multiples_fit_when_99_percent_of_values_are_near_them():
    # One cell holds its smallest value, 1, beside 200 multiples of it, a few
    # of them moved half way to the next.
    base = np.arange(1, 202, dtype=np.float64)
    two_off = base.copy()
    two_off[[10, 20]] += 0.5
    three_off = base.copy()
    three_off[[10, 20, 30]] += 0.5

    assert fitting_readings(_one_cell(two_off)) == {"normalized": None}
    assert fitting_readings(_one_cell(three_off)) == {}


def test_values_within_0_05_of_a_multiple_count_as_near_it():
    near = np.r_[1.0, np.arange(2, 101, dtype=np.float64) + 0.04]
    far = np.r_[1.0, np.arange(2, 101, dtype=np.float64) + 0.06]

    assert fitting_readings(_one_cell(near)) == {"normalized": None}
    assert fitting_readings(_one_cell(far)) == {}


def test_one_cell_cannot_share_a_total():
    # Its values are no multiples of its smallest, and one total agrees with
    # itself, so only a second cell could show a shared total.
    values = np.array([1.0, 1.7, 2.9, 4.2, 5.55])
    assert fitting_readings(_one_cell(values)) == {}
    # A second cell of other values with the same total shares it only as
    # the values are, not after undoing a log.
    two = sp.csr_matrix(np.vstack([values, [2.0, 3.0, 4.0, 5.0, 1.35]]))
    assert fitting_readings(two) == {"normalized": pytest.approx(values.sum())}


def test_whole_multiples_need_evidence_beyond_each_cells_smallest_value():
    # Each cell's smallest value is a multiple of itself, so cells of one or
    # two values say little; 49 informative values cannot decide, 50 can.
    rows = [np.array([1.0, 3.0]) * (index + 1.5) for index in range(49)]
    assert fitting_readings(sp.csr_matrix(np.vstack(rows))) == {}
    rows.append(np.array([1.0, 3.0]) * 60.5)
    assert fitting_readings(sp.csr_matrix(np.vstack(rows))) == {"normalized": None}

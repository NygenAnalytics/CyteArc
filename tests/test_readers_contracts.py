import inspect
from collections.abc import Mapping, Sequence
import pickle
import subprocess
import sys
from typing import get_args, get_type_hints

import numpy as np
import pytest
import zarr

import cytearc.readers as readers_module
from cytearc.readers import (
    CSVReader,
    CrDirReader,
    CrH5Reader,
    CrReader,
    H5adInspectResult,
    H5adReader,
    MtxCandidate,
    MtxReader,
    SeuratInspectResult,
    SeuratReader,
)
from cytearc.readers._text import DuplicateFeatureIds
from tests.signature_contracts import signature_digest


_PUBLIC_CLASS_METHODS = {
    CrReader: (
        "__init__",
        "consume",
        "count_value_ranges",
        "rename_assays",
        "reclassify_features",
        "feature_ids",
        "feature_names",
        "feature_types",
        "cell_names",
    ),
    CrH5Reader: (
        "__init__",
        "cell_names",
        "consume",
        "count_value_ranges",
        "close",
    ),
    CrDirReader: ("__init__",),
    H5adReader: (
        "__init__",
        "from_inspect",
        "cell_ids",
        "feat_ids",
        "feat_names",
        "get_cell_columns",
        "get_feat_columns",
        "feature_types",
        "assay_feature_slices",
        "count_value_ranges",
        "consume_dataset",
        "consume_group",
        "consume",
    ),
    CSVReader: (
        "__init__",
        "cell_ids",
        "feature_ids",
        "consume",
    ),
    MtxReader: (
        "__init__",
        "consume",
        "count_value_ranges",
        "close",
    ),
    SeuratReader: (
        "__init__",
        "close",
        "get_assay",
        "get_reduction",
    ),
}
_PUBLIC_CLASS_SIGNATURE_DIGESTS = {
    CrReader: "0237d8e7c63b64fb6526f3536d5f65c1d704fe56675bf38b41644be419a5607f",
    CrH5Reader: "435e2965b15010e0d1a41a36dae1c492d24eac90386e3e1f58ea8b705570db7d",
    CrDirReader: "47ae6207108e6433f78e3214a0b38c11cadf5cf2d6a79eb218949d6137311bef",
    # H5adReader gained the keyword-only duplicate_cell_ids="error", and its
    # embedding roles gained "coordinates".
    H5adReader: "216ef56b5e006545a348a90b7fde68e51adb532cca98996f85a2fe578dac2a25",
    CSVReader: "8aa6c17c876afb62765584fc7ff64d2838c66ef53095da10d7198ca60ab83851",
    MtxReader: "b0ce7fc6984e0b49abd3e1e3d65ae97194af7b7de083b1d0ea0496735a8c5dd0",
    SeuratReader: "68678f67361080e8121eaae116ace832c1e0defc73ca36e3e3aa02d4d96718e6",
}
_MODULE_SIGNATURE_DIGEST = (
    "21c4a2974e1193da7ebe9be85417b73e2525e875e885c75367a61bf484d4c171"
)


def _write_mixed_cellranger_directory(path) -> None:
    (path / "features.tsv").write_text(
        "f1\tg1\tGene Expression\n"
        "f2\th1\tAntibody Capture\n"
        "f3\tg2\tGene Expression\n"
        "f4\th2\tAntibody Capture\n"
        "f5\ta1\tAntibody Capture\n"
    )
    (path / "barcodes.tsv").write_text("c1\n")
    (path / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n5 1 3\n2 1 2\n4 1 4\n5 1 5\n"
    )


def test_readers_facade_surface_is_stable():
    assert readers_module.__all__ == [
        "CrH5Reader",
        "CrDirReader",
        "CrReader",
        "H5adInspectResult",
        "H5adReader",
        "inspect_h5ad",
        "MtxCandidate",
        "MtxReader",
        "inspect_mtx",
        "SeuratInspectResult",
        "SeuratReader",
        "inspect_seurat",
        "CSVReader",
    ]
    expected = {
        "CSVReader",
        "CrDirReader",
        "CrH5Reader",
        "CrReader",
        "H5adInspectResult",
        "H5adReader",
        "MtxCandidate",
        "MtxReader",
        "SeuratInspectResult",
        "SeuratReader",
        "inspect_h5ad",
        "inspect_mtx",
        "inspect_seurat",
    }
    assert expected.issubset(dir(readers_module))
    assert not hasattr(readers_module, "get_file_handle")
    assert not hasattr(readers_module, "read_file")


def test_readers_facade_loads_format_modules_lazily():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cytearc.readers as readers; "
                "assert 'cytearc.readers.cellranger' not in sys.modules; "
                "assert 'cytearc.readers.csv' not in sys.modules; "
                "assert 'cytearc.readers.h5ad' not in sys.modules; "
                "assert 'cytearc.readers.mtx' not in sys.modules; "
                "assert 'cytearc.readers.seurat' not in sys.modules; "
                "assert 'h5py' not in sys.modules; "
                "assert 'pandas' not in sys.modules; "
                "assert 'scipy' not in sys.modules; "
                "assert 'CSVReader' in dir(readers); "
                "reader = readers.CSVReader; "
                "assert reader.__module__ == 'cytearc.readers'; "
                "assert 'cytearc.readers.csv' in sys.modules; "
                "assert 'cytearc.readers.cellranger' not in sys.modules; "
                "assert 'cytearc.readers.h5ad' not in sys.modules; "
                "assert 'cytearc.readers.mtx' not in sys.modules; "
                "assert 'cytearc.readers.seurat' not in sys.modules; "
                "assert 'pandas' in sys.modules; "
                "assert 'h5py' not in sys.modules; "
                "assert 'scipy' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_matrix_market_exports_load_together_lazily():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cytearc.readers as readers; "
                "assert 'cytearc.readers.mtx' not in sys.modules; "
                "reader = readers.MtxReader; "
                "assert reader.__module__ == 'cytearc.readers'; "
                "assert readers.MtxCandidate.__module__ == 'cytearc.readers'; "
                "assert readers.inspect_mtx.__module__ == 'cytearc.readers'; "
                "assert readers.CrDirReader.__module__ == 'cytearc.readers'; "
                "assert 'cytearc.readers.mtx' in sys.modules; "
                "assert 'cytearc.readers.h5ad' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_seurat_exports_load_together_without_loading_the_writer():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cytearc.readers as readers; "
                "assert 'cytearc.readers.seurat' not in sys.modules; "
                "reader = readers.SeuratReader; "
                "assert reader.__module__ == 'cytearc.readers'; "
                "assert readers.SeuratInspectResult.__module__ == 'cytearc.readers'; "
                "assert readers.inspect_seurat.__module__ == 'cytearc.readers'; "
                "assert 'cytearc.readers.seurat' in sys.modules; "
                "assert 'cytearc.writers.seurat' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_seurat_reader_facade_objects_resolve_annotations_and_pickle():
    for name in ("__init__", "get_assay", "get_reduction"):
        assert get_type_hints(getattr(SeuratReader, name))
    for value in (
        SeuratReader,
        SeuratInspectResult,
        readers_module.inspect_seurat,
    ):
        assert pickle.loads(pickle.dumps(value)) is value


def test_reader_class_and_method_signatures_are_stable():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        methods = {name: getattr(cls, name) for name in names}
        assert signature_digest(methods) == _PUBLIC_CLASS_SIGNATURE_DIGESTS[cls]


def test_count_readers_take_the_duplicate_feature_ids_keyword():
    assert get_args(DuplicateFeatureIds.__value__) == ("error", "make_unique")
    for cls in (CrH5Reader, CrDirReader, MtxReader, H5adReader):
        parameter = inspect.signature(cls.__init__).parameters["duplicate_feature_ids"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == "error"
        hints = get_type_hints(cls.__init__)
        assert hints["duplicate_feature_ids"] is DuplicateFeatureIds
    for cls in (CSVReader, SeuratReader):
        assert "duplicate_feature_ids" not in inspect.signature(cls.__init__).parameters


def test_seurat_reader_and_inspection_select_metadata_and_reduction_roles():
    for function in (SeuratReader.__init__, readers_module.inspect_seurat):
        parameters = inspect.signature(function).parameters
        assert list(parameters)[-2:] == ["cell_metadata", "reduction_roles"]
        for name in ("cell_metadata", "reduction_roles"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is None
        hints = get_type_hints(function)
        assert hints["cell_metadata"] == (
            Sequence[str] | Mapping[str, Sequence[str]] | None
        )
        roles, none = get_args(hints["reduction_roles"])
        assert none is type(None)
        key, role = get_args(roles)
        assert key is str
        assert get_args(role.__value__) == ("umap", "tsne", "coordinates")


def test_reader_module_function_signatures_are_stable():
    methods = {
        name: getattr(readers_module, name)
        for name in (
            "inspect_h5ad",
            "inspect_mtx",
            "inspect_seurat",
        )
    }
    assert signature_digest(methods) == _MODULE_SIGNATURE_DIGEST


def test_reader_public_metadata_remains_on_facade():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        assert cls.__module__ == "cytearc.readers"
        for name in names:
            descriptor = inspect.getattr_static(cls, name)
            method = (
                descriptor.__func__
                if isinstance(descriptor, classmethod | staticmethod)
                else descriptor
            )
            assert method.__module__ == "cytearc.readers"
            assert method.__qualname__.startswith(f"{cls.__name__}.")

    assert H5adInspectResult.__module__ == "cytearc.readers"
    assert MtxCandidate.__module__ == "cytearc.readers"
    assert SeuratInspectResult.__module__ == "cytearc.readers"
    for name in (
        "inspect_h5ad",
        "inspect_mtx",
        "inspect_seurat",
    ):
        assert getattr(readers_module, name).__module__ == "cytearc.readers"


def test_cellranger_reader_hierarchy_and_abstract_contracts_are_stable():
    assert inspect.isabstract(CrReader)
    assert issubclass(CrH5Reader, CrReader)
    assert issubclass(CrDirReader, MtxReader)
    assert issubclass(MtxReader, CrReader)
    for name in ("_handle_version", "_read_dataset", "consume"):
        assert getattr(CrReader, name).__isabstractmethod__


def test_crreader_reclassifies_noncontiguous_features_atomically(tmp_path):
    _write_mixed_cellranger_directory(tmp_path)
    reader = CrDirReader(str(tmp_path))

    reader.reclassify_features([1, 3], "HTO")

    assert reader.feature_types() == [
        "Gene Expression",
        "HTO",
        "Gene Expression",
        "HTO",
        "Antibody Capture",
    ]
    assert list(reader.assayFeats.columns) == ["RNA", "HTO", "RNA", "HTO", "ADT"]
    assert reader.feature_names("HTO") == ["h1", "h2"]
    assert reader.feature_names("ADT") == ["a1"]

    before = reader.assayFeats.copy()
    reader.reclassify_features([1, 3], "HTO")
    assert reader.assayFeats.equals(before)

    with pytest.raises(ValueError, match="conflicting"):
        reader.reclassify_features([1], "RNA", require_previous=None)
    assert reader.feature_types()[1] == "HTO"


def test_crreader_reclassification_validates_before_mutation(tmp_path):
    _write_mixed_cellranger_directory(tmp_path)
    reader = CrDirReader(str(tmp_path))
    original_types = reader.feature_types()
    original_table = reader.assayFeats.copy()

    with pytest.raises(ValueError, match="currently have type"):
        reader.reclassify_features([1, 2], "HTO")
    with pytest.raises(ValueError, match="unique"):
        reader.reclassify_features([1, 1], "HTO")
    with pytest.raises(IndexError, match="out-of-range"):
        reader.reclassify_features([5], "HTO")

    assert reader.feature_types() == original_types
    assert reader.assayFeats.equals(original_table)


def test_crreader_reclassification_locks_when_writer_captures_schema(tmp_path):
    from cytearc.writers import CrToZarr

    source = tmp_path / "source"
    source.mkdir()
    _write_mixed_cellranger_directory(source)
    reader = CrDirReader(str(source))
    reader.reclassify_features([1, 3], "HTO")

    destination = tmp_path / "out.zarr"
    writer = CrToZarr(reader, str(destination))
    writer.dump()
    root = zarr.open_group(str(destination), mode="r")

    assert set(root.group_keys()) >= {"RNA", "HTO", "ADT"}
    np.testing.assert_array_equal(
        np.asarray(root["HTO/featureData/ids"][:]).astype(str),
        ["f2", "f4"],
    )
    np.testing.assert_array_equal(
        np.asarray(root["ADT/featureData/ids"][:]).astype(str),
        ["f5"],
    )
    np.testing.assert_array_equal(root["HTO/counts"][:], [[2, 4]])
    np.testing.assert_array_equal(root["ADT/counts"][:], [[5]])
    with pytest.raises(RuntimeError, match="captures the schema"):
        reader.reclassify_features([4], "HTO")

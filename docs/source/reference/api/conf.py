import sys
from pathlib import Path

import matplotlib

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT.parents[1]))

project = "CyteArc API"
copyright = "2020-2026, Parashar Dhapola"
author = "Parashar Dhapola"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]
autosummary_generate = False
autodoc_type_aliases = {
    "DataStore": "cytearc.datastore.datastore.DataStore",
    "DataStorePlotAccessor": "cytearc.datastore.plot_accessor.DataStorePlotAccessor",
}
master_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints", "agent.md"]
myst_enable_extensions = ["colon_fence"]
myst_all_links_external = True
pygments_style = "sphinx"
language = "en"

html_theme = "sphinx_book_theme"
html_title = "CyteArc API reference"
html_baseurl = "https://docs.nygen.io/CyteArc/api/"
html_favicon = str(SOURCE_ROOT / "favicon.ico")
html_logo = str(SOURCE_ROOT / "_static" / "cytearc-logo-black.png")
html_theme_options = {
    "repository_url": "https://github.com/NygenAnalytics/CyteArc",
    "path_to_docs": "docs/source/reference/api",
    "use_repository_button": True,
    "show_navbar_depth": 2,
    "navigation_with_keys": False,
}

matplotlib.use("agg")

nitpick_ignore = [
    ("py:class", "_duckdb.DuckDBPyConnection"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "numpy.dtype"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "pandas.Series"),
    ("py:class", "pandas.api.extensions.ExtensionArray"),
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "pandas.core.series.Series"),
    ("py:class", "scipy.sparse._csr.csr_matrix"),
    ("py:class", "scipy.sparse._coo.coo_matrix"),
    ("py:class", "zarr.abc.store.Store"),
    ("py:class", "zarr.core.group.Group"),
    ("py:class", "zarr.core.array.Array"),
    ("py:class", "pathlib.Path"),
    ("py:class", "pathlib._local.Path"),
    ("py:class", "os.PathLike"),
    ("py:class", "collections.abc.Callable"),
    ("py:class", "collections.abc.Iterable"),
    ("py:class", "collections.abc.Iterator"),
    ("py:class", "collections.abc.Sequence"),
    ("py:class", "collections.abc.Generator"),
    ("py:class", "cytearc.matrix.ChunkedArray"),
    ("py:class", "cytearc.readers.CrReader"),
    ("py:class", "cytearc.readers.h5ad._H5adAssayFeatures"),
    ("py:class", "cytearc.storage.profiles.StorageProfile"),
    ("py:obj", "cytearc.storage.profiles.StorageProfile"),
    ("py:class", "cytearc.storage.profiles.ZarrLocation"),
    ("py:obj", "cytearc.storage.profiles.ZarrLocation"),
    ("py:class", "cytearc.storage.budget.ResourceBudget"),
    ("py:class", "cytearc.storage.io_policy.StorageIoPolicy"),
    ("py:class", "cytearc.storage.count_matrix.CountMatrixPolicy"),
    ("py:class", "cytearc.metadata.MetaDataRowBlock"),
    ("py:class", "cytearc.metadata.selection.NamedCellArtifact"),
    ("py:class", "cytearc.datastore.mapping_datastore.MappingDatastore"),
    ("py:obj", "numpy.typing.DTypeLike"),
    ("py:obj", "numpy.typing.NDArray"),
    ("py:data", "typing.Any"),
    ("py:data", "typing.Literal"),
    ("py:data", "typing.Optional"),
    ("py:data", "typing.Union"),
    ("py:data", "Ellipsis"),
]

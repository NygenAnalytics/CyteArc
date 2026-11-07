"""The clone stack's modules, for guard tests that check every one of them."""

from pathlib import Path

CYTEARC_ROOT = Path(__file__).resolve().parents[1] / "cytearc"
# The clone stack's modules: its three packages, and its modules in shared
# packages.
STACK_PACKAGES = ("clones", "repertoire", "variants")
STACK_MODULES = (
    "datastore/_operations/cell_barcodes.py",
    "datastore/_operations/cell_groups.py",
    "datastore/_operations/clone_artifacts.py",
    "datastore/_operations/clones.py",
    "datastore/_operations/lineage_barcodes.py",
    "datastore/_operations/phylogeny.py",
    "datastore/_operations/record_tables.py",
    "datastore/_operations/recorder_alleles.py",
    "datastore/_operations/repertoire.py",
    "datastore/_operations/variants.py",
    "metadata/records.py",
    "plotting/_charts.py",
    "plotting/_clone_names.py",
    "plotting/_groups.py",
    "plotting/clones.py",
    "plotting/lineage_barcodes.py",
    "plotting/recorder_alleles.py",
    "plotting/repertoire.py",
    "plotting/variants.py",
    "readers/_contigs.py",
    "utils/delimited.py",
    "readers/_matrix_market.py",
    "readers/_treedata.py",
    "readers/_variant_files.py",
    "readers/airr.py",
    "readers/cellsnp.py",
    "readers/clone_tables.py",
    "readers/giftwrap.py",
    "readers/got.py",
    "readers/lineage_barcodes.py",
    "readers/mgatk.py",
    "readers/mito_afm.py",
    "readers/phylogeny.py",
    "readers/recorder_alleles.py",
    "readers/tapestri.py",
    "readers/vartrix.py",
    "utils/files.py",
    "writers/airr.py",
    "writers/characters.py",
    "writers/clone_tables.py",
    "writers/tapestri.py",
)


def stack_module_paths() -> list[Path]:
    paths = []
    for package in STACK_PACKAGES:
        # A renamed or removed package must fail here instead of passing.
        assert (CYTEARC_ROOT / package / "__init__.py").is_file(), package
        paths.extend(sorted((CYTEARC_ROOT / package).rglob("*.py")))
    for module in STACK_MODULES:
        assert (CYTEARC_ROOT / module).is_file(), module
        paths.append(CYTEARC_ROOT / module)
    return paths

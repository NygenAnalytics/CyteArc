---
description: Install CyteArc with Python 3.13 or newer, configure a C++ compiler, and choose optional dependencies for your analysis.
---

(installation)=
# Installation

Install CyteArc into its own Python environment, check that it imports, and add the optional packages that some analyses need. CyteArc requires Python 3.13 or newer. Use uv where you can; pip works without uv, and conda needs no compiler.

## Install a C++ compiler

```{note}
CyteArc depends on `hnswlib` for nearest-neighbor graphs, and `hnswlib` is published only as a source distribution, so `pip` and `uv` compile it during installation. That step needs a C++ compiler: `build-essential` on Debian and Ubuntu, and the Xcode Command Line Tools (`xcode-select --install`) on macOS. On Windows, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools) with the "Desktop development with C++" workload, or use the conda environment below.
```

On Debian or Ubuntu, install the compiler before installing CyteArc:

```bash
sudo apt install build-essential
```

A pip environment built on the distribution's system Python also needs the interpreter's development headers and virtual-environment support before `python -m venv` runs:

```bash
sudo apt install python3-dev python3-venv
```

These unversioned packages match the distribution's default `python3`; for another system interpreter, install its matching packages, such as `python3.13-dev` and `python3.13-venv`. A Python that uv installs includes both already.

## Install with uv

[uv](https://docs.astral.sh/uv/) creates an isolated `.venv` in the current directory, so installing CyteArc leaves the system Python unchanged. Run these commands in the directory where you will work:

```bash
uv venv --python 3.13
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install CyteArc into the activated environment:

```bash
uv pip install "cytearc[extra]"
```

The `extra` group adds the packages that plotting, AnnData export and the tutorials use: `matplotlib`, `seaborn`, `anndata`, `ipywidgets`, `kneed` and `ipython-autotime`.

Check the installation with the environment's Python. `import cytearc` loads most modules only when they are first used, so the check also imports `DataStore`:

```bash
python -c "from cytearc import DataStore; import cytearc; print(cytearc.__version__)"
```

## Install with pip

Without uv, confirm that `python` is version 3.13 or newer, then create a standard virtual environment:

```bash
python -m venv .venv
```

Activate it with the command for your system above, then install and check CyteArc:

```bash
python -m pip install "cytearc[extra]"
python -c "from cytearc import DataStore; import cytearc; print(cytearc.__version__)"
```

## Install with conda

conda-forge publishes a prebuilt `hnswlib`, so this environment needs no compiler, which makes it the choice on Windows without Visual C++ Build Tools:

```bash
conda create --name cytearc_env --channel conda-forge python=3.13 pip "hnswlib>=0.8"
conda activate cytearc_env
python -m pip install "cytearc[extra]"
python -c "from cytearc import DataStore; import cytearc; print(cytearc.__version__)"
```

## Install the CyteBase catalog client

CyteBase publishes public single-cell datasets as CyteArc stores in Hugging Face buckets, and CyteArc opens them by URI with no extra package, as [](tutorials/cytebase_datasets.md) shows. To search the catalog, install the separate `cytebase` package with `uv pip install cytebase`; the [CyteBase documentation](https://docs.nygen.io/CyteBase/) describes it.

(installation_tsne)=
## Optional t-SNE

`embeddings.tsne` and the `tsne` stage of the pipeline recipes compute t-SNE with the [sgtsnepi](https://github.com/alex-unofficial/sgtsnepi-python) package, which CyteArc does not install by default. Add it with the `tsne` extra, in the same environment as CyteArc:

```bash
uv pip install "cytearc[extra,tsne]"
```

`sgtsnepi` publishes wheels only for Linux x86_64 and for macOS 26 or newer on Apple silicon (arm64). On other platforms the installer compiles it from source, which needs a C++ compiler and FFTW, and Windows is not supported. `sgtsnepi` runs on one thread and is licensed under the GPLv3; CyteArc, which is BSD-3-Clause, installs it only with the `tsne` extra.

Without `sgtsnepi`, computing a new t-SNE embedding raises an `ImportError` that names the `tsne` extra, while reusing, plotting or exporting a t-SNE embedding that a store already holds works. A pipeline run with `tsne=True` therefore still starts without `sgtsnepi`: it logs a warning that names the extra before its first stage, and its `tsne` stage fails unless the store already holds a matching embedding.

(installation_leiden_gpl)=
## Optional GPL Leiden backends

Leiden clustering needs no extra package: its default backend, `"cytearc"`, is CyteArc's own multi-threaded implementation (see {ref}`Leiden clustering <leiden_method>`). The `"igraph"` and `"leidenalg"` backends of `clusters.leiden`, of the pipeline recipes' `leiden` settings and of `cytearc.clustering.leiden_membership` call the [igraph](https://pypi.org/project/igraph/) and [leidenalg](https://pypi.org/project/leidenalg/) packages instead. These backends reproduce a clustering computed with those packages, such as one saved by an earlier CyteArc release, whose default backend was `"igraph"`.

CyteArc does not install these packages and lists them in none of its extras: igraph is licensed under the GPL-2.0-or-later and leidenalg under the GPL-3.0-or-later, while CyteArc is BSD-3-Clause. To use these backends, install the packages yourself in the same environment as CyteArc; leidenalg installs igraph as a dependency:

```bash
uv pip install leidenalg
```

Both run on one thread. Without them, computing a new clustering with either backend raises an `ImportError` that explains how to install them, while reusing, plotting or exporting an `"igraph"` or `"leidenalg"` clustering that a store already holds works.

## Optional regulon databases

`cytearc.regulons.RankingDatabase` reads cisTarget ranking databases with [polars](https://pola.rs), which CyteArc does not install by default. Add it with the `regulons` extra:

```bash
uv pip install "cytearc[extra,regulons]"
```

Without `polars`, opening a ranking database raises an `ImportError` that names the extra.

## Optional parquet allele tables

`ds.clones.import_alleles` reads Cassiopeia 3.0's parquet allele tables with [polars](https://pola.rs) 1.30 or later, which reads the attributes that Cassiopeia keeps in a parquet file's metadata. CyteArc does not install it by default, and the `regulons` extra accepts earlier versions, so install it directly:

```bash
uv pip install "polars>=1.30"
```

Without it, or with an earlier version, reading such a table raises an `ImportError` that names the version.

## Run CyteArc in JupyterLab

Install and launch JupyterLab from the same activated environment as CyteArc. For the uv environment:

```bash
uv pip install jupyterlab
jupyter lab
```

For the pip environment:

```bash
python -m pip install jupyterlab
jupyter lab
```

For the conda environment, install JupyterLab and register a named kernel:

```bash
conda install --channel conda-forge jupyterlab ipykernel
python -m ipykernel install --user --name cytearc_env --display-name "Python (cytearc_env)"
jupyter lab
```

Select `Python (cytearc_env)` when you open a notebook from another Jupyter environment.

## Enable long paths on Windows

CyteArc's nested Zarr stores can exceed the default Windows path limit. Enable long paths in PowerShell run as Administrator:

````{note}
```{code-block} powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Restart Windows after changing the registry so that running processes pick up the new value. The [Microsoft long-path documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation) explains the limit.
````

## Upgrade CyteArc

Activate the environment that holds CyteArc, then upgrade with the installer you used:

```bash
uv pip install --upgrade "cytearc[extra]"
```

For pip and conda environments:

```bash
python -m pip install --upgrade "cytearc[extra]"
```

## Install from source

Clone the repository and let uv create an editable project environment:

```bash
git clone https://github.com/NygenAnalytics/CyteArc.git
cd CyteArc
uv sync --extra extra
uv run python -c "from cytearc import DataStore; import cytearc; print(cytearc.__version__)"
```

Read [](developers/contributing.md) before changing the package or the documentation.

## Fix installation errors

- **No virtual environment found:** activate `.venv`, or name it explicitly with `uv pip install --python .venv "cytearc[extra]"`.
- **`hnswlib` fails to build:** install the compiler and, for a system Python, the matching development headers described above; on Windows without Build Tools, use the conda environment.
- **`embeddings.tsne` raises `ImportError` naming the `tsne` extra:** install the extra as {ref}`Optional t-SNE <installation_tsne>` describes, or use UMAP.
- **`clusters.leiden` raises `ImportError` naming igraph or leidenalg:** the call asked for `backend="igraph"` or `backend="leidenalg"`, whose GPL-licensed packages CyteArc does not install. Install them as {ref}`Optional GPL Leiden backends <installation_leiden_gpl>` describes, or use the default `backend="cytearc"`.
- **CyteArc imports in a terminal but not in a notebook:** compare the interpreter that `python -c "import sys; print(sys.executable)"` prints in the terminal with the notebook's kernel, then launch Jupyter from the CyteArc environment or select its registered kernel.

## Next steps

After the import check succeeds, continue with the {ref}`Quick start <quickstart>`.

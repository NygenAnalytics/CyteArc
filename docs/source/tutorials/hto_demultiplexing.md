---
description: Assign sample identities from hashtag oligo counts and interpret singlet, negative, and doublet labels.
---

(hto_demultiplexing)=

# HTO demultiplexing

Cell hashing labels cells from each sample with a distinct oligonucleotide tag before the
samples are pooled. Depending on the protocol, tags are attached through antibodies or lipids.
The pooled library then contains RNA reads and hashtag oligo (HTO) counts for each droplet.
Demultiplexing uses those counts to recover the sample assignments: a singlet has one confident
hashtag, a negative has none, and a doublet has more than one.

Pooling lets samples share a capture, but an incorrect assignment can mix their biological
signals. Inspect the assignments before comparing samples. This is separate from integrating
RNA and protein measurements in [](cite_seq.ipynb).

CyteArc follows the general HTOdemux strategy: normalize hashtag counts with CLR, cluster cells
into one more group than the number of hashtags, and use low-count clusters to estimate each
tag's background. A negative-binomial fit defines a positive-count cutoff for each tag.
The number of tags above their cutoffs determines whether a droplet is negative, a singlet,
or a doublet. Singlets receive the identifier of their strongest normalized hashtag.

## Run HTO demultiplexing

CyteArc expects an HTO assay, named `HTO` by default, in the same datastore as the biological assays.
Open that datastore as `ds` before following the examples. The assay must be declared as type
`HTO`; naming an ordinary RNA or ADT assay `HTO` is not enough. Declare the type when importing, or
open the store once with `zarr_mode="r+"` and `assay_types={"HTO": "HTO"}`, which records the type
in the store for later opens. A merged store keeps the `HTO` type when every source that holds the
assay declares it.
`qc.hto_demultiplexing` normalizes the hashtag counts, estimates background, and returns an immutable
identity artifact without changing shared cell metadata.

```python
# Freeze the active cells for hashtag demultiplexing.
cell_selection = ds.snapshot_cell_selection("I")

# Assign hashtag identities, negatives, and doublets.
identities = ds.qc.hto_demultiplexing(cell_selection)

# Inspect the assigned hashtag identities.
ds.artifacts.load(identities)["values"][:]
```

## Interpret singlet, negative, and doublet labels

Inspect the loaded values and compare identity counts with the experiment's expected loading.
Singlet labels can define downstream selections or pseudobulk groups. To retain all singlets
without creating a metadata column, select the exact HTO identifiers:

```python
# Read the hashtag identifiers that represent singlets.
singlet_labels = ds.HTO.feats.fetch_all("ids").astype(str).tolist()

# Keep cells assigned to one of those hashtags.
singlets = ds.select_cells(identities, include=singlet_labels)

# Count cells assigned to a single hashtag.
int(ds.artifacts.load(singlets)["values"][:].sum())
```

Negative cells do not have a confident hashtag assignment. Doublets carry evidence for more
than one hashtag and should not be silently relabelled as one sample. A common next step is to
analyze confident singlets, but first check which populations an exclusion would remove.

Thresholds depend on panel chemistry, loading, and background.
Review the hashtag count distributions and manually retain or exclude negative and doublet classes according to the analysis question.
The method does not replace RNA doublet scoring, because homotypic and untagged multiplets can remain.

## Catalog limitation

CyteArc's public dataset catalog does not currently contain a cell-hashing dataset, so this page cannot provide an executable result without inventing unrepresentative data.
Adding a licensed public HTO dataset to the catalog is required before this guide can become executable.
See the [DataStore API reference](/api/datastore.html) reference for the current signature.

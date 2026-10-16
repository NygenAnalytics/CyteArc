"""Prune co-expression modules to regulons with cisTarget motif enrichment."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.arguments import float_argument, integer_argument

from .cistarget import _envelope, _rank_at_max, auc_rank_cutoff, recovery_aucs
from .databases import RankingDatabase

__all__ = [
    "PruningParameters",
    "SCENIC_PRUNING_PARAMETERS",
    "prune_modules",
    "regulons_from_motifs",
]

MOTIF_COLUMNS = (
    "TF",
    "direction",
    "module",
    "context",
    "database",
    "motif",
    "AUC",
    "NES",
    "motif_similarity_qvalue",
    "orthologous_identity",
    "annotation",
    "rank_at_max",
    "targets",
    "target_weights",
)
REGULON_COLUMNS = ("source", "TF", "direction", "target", "weight")


@dataclass(frozen=True, slots=True)
class PruningParameters:
    """cisTarget settings for pruning modules.

    The defaults are those of ``pyscenic ctx`` 0.12.1.

    Attributes:
        rankThreshold: Length of the recovery curves that define leading
            edges, in ranks.
        aucThreshold: Fraction of each motif's ranking covered by the AUC.
        nesThreshold: Smallest NES of an enriched motif.
        maxMissingFraction: Skip a module once this fraction of its genes,
            the factor included, is missing from a database.
    """

    rankThreshold: int = 5000
    aucThreshold: float = 0.05
    nesThreshold: float = 3.0
    maxMissingFraction: float = 0.2

    def __post_init__(self) -> None:
        integer_argument(self.rankThreshold, "rankThreshold", minimum=1)
        float_argument(self.nesThreshold, "nesThreshold")
        if not 0 < self.aucThreshold <= 1:
            raise ValueError("aucThreshold must be in (0, 1]")
        if not 0 < self.maxMissingFraction <= 1:
            raise ValueError("maxMissingFraction must be in (0, 1]")


#: ``pyscenic ctx`` 0.12.1 defaults.
SCENIC_PRUNING_PARAMETERS = PruningParameters()


def _best_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    """One annotation per (TF, motif): smallest q-value, then highest identity."""
    ordered = annotations.sort_values(
        ["motif_similarity_qvalue", "orthologous_identity"],
        ascending=[True, False],
        kind="mergesort",
        na_position="last",
    )
    best = ordered.drop_duplicates(["TF", "motif"], keep="first")
    return best.set_index(["TF", "motif"])


@dataclass(frozen=True, slots=True)
class _Module:
    name: str
    tf: str
    direction: str
    context: str
    genes: np.ndarray
    weights: np.ndarray


def _modules(modules: pd.DataFrame) -> list[_Module]:
    required = {"source", "TF", "direction", "context", "target", "weight"}
    missing = required - set(modules.columns)
    if missing:
        raise ValueError(f"modules lacks columns {sorted(missing)}")
    result = []
    for name, group in modules.groupby("source", sort=False):
        first = group.iloc[0]
        if group["target"].duplicated().any():
            raise ValueError(f"Module {name!r} repeats a gene")
        result.append(
            _Module(
                name=str(name),
                tf=str(first["TF"]),
                direction=str(first["direction"]),
                context=str(first["context"]),
                genes=group["target"].astype(str).to_numpy(),
                weights=group["weight"].to_numpy(dtype=np.float64),
            )
        )
    return result


def prune_modules(
    modules: pd.DataFrame,
    databases: Sequence[RankingDatabase],
    annotations: pd.DataFrame,
    *,
    parameters: PruningParameters = SCENIC_PRUNING_PARAMETERS,
    nthreads: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Keep the module targets that an enriched motif of the module's factor recovers.

    For every module and database, each motif's ranking is scored by the AUC
    of the module's recovery curve and standardized against all motifs of
    the database (NES). Enriched motifs (``NES >= nesThreshold``) that the
    annotation table assigns to the module's factor contribute their leading
    edge: the module genes ranked at or before the rank where the motif's
    recovery curve rises furthest above the mean plus two standard deviations
    of all motifs' curves. This is pySCENIC's ``prune2df`` with each leading
    edge taken from its own motif's curve. pySCENIC 0.12.1 pairs the
    annotated motifs, after sorting them by annotation quality, with curves in
    database order, so most of its leading edges come from another enriched
    motif and its regulon targets differ from these.

    Args:
        modules: Output of ``modules_from_adjacencies``.
        databases: Gene-based ranking databases.
        annotations: Output of ``read_motif_annotations``; every listed
            (factor, motif) pair counts, whatever its category.
        parameters: cisTarget settings; ``pyscenic ctx`` defaults when omitted.
        nthreads: Modules pruned concurrently.
        progress: Optional ``callback(done, total)`` after each module and
            database.

    Returns:
        One row per module, database, and enriched annotated motif, with the
        motif statistics, the annotation, and the leading-edge ``targets``
        with their module ``target_weights``.
    """
    if not databases:
        raise ValueError("At least one ranking database is required")
    integer_argument(nthreads, "nthreads", minimum=1)
    annotation_columns = {
        "motif",
        "TF",
        "motif_similarity_qvalue",
        "orthologous_identity",
        "description",
    }
    if not annotation_columns <= set(annotations.columns):
        raise ValueError("annotations must come from read_motif_annotations")
    best_annotations = _best_annotations(annotations)
    module_list = _modules(modules)
    all_genes = sorted({gene for module in module_list for gene in module.genes})
    records: list[dict[str, object]] = []
    total_tasks = len(module_list) * len(databases)
    done = 0

    for database in databases:
        auc_rank_cutoff(parameters.aucThreshold, database.total)
        if not 0 < parameters.rankThreshold < database.total:
            raise ValueError(
                f"rankThreshold must be below the {database.total} ranked features "
                f"of {database.name}"
            )
        cutoff = int(round(parameters.aucThreshold * database.total))
        if cutoff > parameters.rankThreshold:
            raise ValueError(
                "The AUC covers more ranks than rankThreshold; raise rankThreshold "
                "or lower aucThreshold"
            )
        found, ranks = database.ranks(all_genes)
        column = {gene: index for index, gene in enumerate(found.tolist())}
        motifs = database.rowNames

        def prune_one(module: _Module) -> list[dict[str, object]]:
            present = np.asarray([gene in column for gene in module.genes])
            if (len(module.genes) - present.sum()) / len(module.genes) >= (
                parameters.maxMissingFraction
            ):
                return []
            # ctxcore loads the present genes sorted by name.
            order = np.argsort(module.genes[present], kind="stable")
            genes = module.genes[present][order]
            weights = module.weights[present][order]
            module_ranks = np.ascontiguousarray(
                ranks[:, [column[gene] for gene in genes]]
            )
            unit = np.ones(len(genes), dtype=np.float64)
            aucs = recovery_aucs(
                module_ranks, database.total, unit, parameters.aucThreshold
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                nes = (aucs - aucs.mean()) / aucs.std()
            enriched = np.flatnonzero(nes >= parameters.nesThreshold)
            if len(enriched) == 0:
                return []
            keys = [(module.tf, str(motifs[index])) for index in enriched]
            known = [key in best_annotations.index for key in keys]
            if not any(known):
                return []
            envelope = _envelope(module_ranks, unit, parameters.rankThreshold)
            rows: list[dict[str, object]] = []
            for index, key, annotated in zip(enriched, keys, known, strict=True):
                if not annotated:
                    continue
                annotation = best_annotations.loc[key]
                motif_ranks = module_ranks[index]
                rank_at_max = _rank_at_max(motif_ranks, unit, envelope)
                keep = np.flatnonzero(motif_ranks <= rank_at_max)
                keep = keep[np.argsort(motif_ranks[keep], kind="stable")]
                rows.append(
                    {
                        "TF": module.tf,
                        "direction": module.direction,
                        "module": module.name,
                        "context": module.context,
                        "database": database.name,
                        "motif": key[1],
                        "AUC": float(aucs[index]),
                        "NES": float(nes[index]),
                        "motif_similarity_qvalue": float(
                            annotation["motif_similarity_qvalue"]
                        ),
                        "orthologous_identity": float(
                            annotation["orthologous_identity"]
                        ),
                        "annotation": str(annotation["description"]),
                        "rank_at_max": int(rank_at_max),
                        "targets": genes[keep].tolist(),
                        "target_weights": weights[keep].tolist(),
                    }
                )
            return rows

        def run(module: _Module) -> list[dict[str, object]]:
            return prune_one(module)

        if nthreads == 1:
            results = map(run, module_list)
            for rows in results:
                records.extend(rows)
                done += 1
                if progress is not None:
                    progress(done, total_tasks)
        else:
            with ThreadPoolExecutor(max_workers=nthreads) as executor:
                for rows in executor.map(run, module_list):
                    records.extend(rows)
                    done += 1
                    if progress is not None:
                        progress(done, total_tasks)
    if not records:
        return pd.DataFrame({column: [] for column in MOTIF_COLUMNS})
    return pd.DataFrame.from_records(records, columns=list(MOTIF_COLUMNS))


def regulons_from_motifs(motifs: pd.DataFrame) -> pd.DataFrame:
    """Merge the leading edges of each factor and direction into one regulon.

    A regulon ``TF(+)`` (activating) or ``TF(-)`` (repressing) holds the union
    of its factor's leading-edge targets over all modules, databases, and
    enriched motifs, each with its largest weight, as pySCENIC's
    ``df2regulons`` builds them. The factor is a member only when some
    leading edge holds it. No minimum size is applied.

    Args:
        motifs: Output of ``prune_modules``.

    Returns:
        A network with ``source`` (the regulon), ``TF``, ``direction``,
        ``target``, and ``weight`` columns, which ``ds.scores.aucell`` scores.
    """
    missing = {"TF", "direction", "targets", "target_weights"} - set(motifs.columns)
    if missing:
        raise ValueError(f"motifs lacks columns {sorted(missing)}")
    rows = []
    for (tf, direction), group in motifs.groupby(["TF", "direction"], sort=True):
        best: dict[str, float] = {}
        for targets, weights in zip(
            group["targets"], group["target_weights"], strict=True
        ):
            for target, weight in zip(targets, weights, strict=True):
                if target not in best or weight > best[target]:
                    best[target] = float(weight)
        suffix = "(+)" if direction == "activating" else "(-)"
        for target in sorted(best):
            rows.append((f"{tf}{suffix}", tf, direction, target, best[target]))
    return pd.DataFrame.from_records(rows, columns=list(REGULON_COLUMNS))

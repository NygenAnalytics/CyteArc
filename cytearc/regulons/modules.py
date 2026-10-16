"""Co-expression modules from transcription-factor to target importances."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.arguments import integer_argument

__all__ = [
    "ModuleParameters",
    "SCENIC_MODULE_PARAMETERS",
    "modules_from_adjacencies",
]

MODULE_COLUMNS = ("source", "TF", "direction", "context", "target", "weight")


@dataclass(frozen=True, slots=True)
class ModuleParameters:
    """Rules that turn importance links into candidate regulon modules.

    The defaults are pySCENIC 0.12.1's (``modules_from_adjacencies`` and the
    ``pyscenic ctx`` command line).

    Attributes:
        thresholds: Importance quantiles over all links; a module holds a
            factor's targets above each quantile.
        topTargets: A module holds each factor's top targets by importance.
        topRegulators: A target joins the modules of its top factors by
            importance.
        minGenes: Smallest module size, counting the factor itself.
        rhoThreshold: Links whose Pearson correlation exceeds this are
            activating; those below its negative are repressing; the rest are
            ignored.
        keepOnlyActivating: Build only activating modules.
    """

    thresholds: tuple[float, ...] = (0.75, 0.90)
    topTargets: tuple[int, ...] = (50,)
    topRegulators: tuple[int, ...] = (5, 10, 50)
    minGenes: int = 20
    rhoThreshold: float = 0.03
    keepOnlyActivating: bool = True

    def __post_init__(self) -> None:
        if any(not 0 <= value <= 1 for value in self.thresholds):
            raise ValueError("thresholds must be quantiles in [0, 1]")
        for label, values in (
            ("thresholds", self.thresholds),
            ("topTargets", self.topTargets),
            ("topRegulators", self.topRegulators),
        ):
            # A repeated rule would merge two modules under one name.
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must not repeat a value")
            if label != "thresholds":
                for value in values:
                    integer_argument(value, label, minimum=1)
        integer_argument(self.minGenes, "minGenes", minimum=1)
        if not self.rhoThreshold > 0:
            raise ValueError("rhoThreshold must be positive")


#: pySCENIC 0.12.1 defaults.
SCENIC_MODULE_PARAMETERS = ModuleParameters()


def _first_largest(frame: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    """Rows of the ``n`` largest values per ``by`` group, ties by input order.

    This is pandas ``nlargest(n, keep="first")`` applied per group.
    """
    ordered = frame.sort_values(
        [by, "importance", "_row"], ascending=[True, False, True], kind="mergesort"
    )
    return ordered.groupby(by, sort=False).head(n)


def modules_from_adjacencies(
    adjacencies: pd.DataFrame,
    *,
    parameters: ModuleParameters = SCENIC_MODULE_PARAMETERS,
) -> pd.DataFrame:
    """Build pySCENIC's co-expression modules from a TF-target network.

    Each factor gets one module per rule: its targets above each importance
    quantile of all links (``weight>75.0%``), its top targets (``top50``),
    and the targets for which it is among the top factors
    (``top5perTarget``). Links are split by the sign of their correlation
    into activating and repressing modules, the factor joins each of its
    modules with weight 1, and modules smaller than ``minGenes`` are dropped.

    Ties in importance are broken by row order, so pass links in the order of
    the network that produced them.

    Args:
        adjacencies: ``TF``, ``target``, ``importance``, and ``correlation``
            columns, such as ``GrnResult.adjacencies``.
        parameters: Module rules; pySCENIC defaults when omitted.

    Returns:
        A long table with one row per module gene: ``source`` (a unique
        module name), ``TF``, ``direction`` (``activating`` or ``repressing``),
        ``context`` (the rule), ``target``, and ``weight`` (the link
        importance, 1 for the factor).
    """
    required = {"TF", "target", "importance", "correlation"}
    missing = required - set(adjacencies.columns)
    if missing:
        raise ValueError(f"adjacencies lacks columns {sorted(missing)}")
    links = adjacencies.loc[:, ["TF", "target", "importance", "correlation"]].copy()
    links["TF"] = links["TF"].astype(str)
    links["target"] = links["target"].astype(str)
    links["importance"] = links["importance"].astype(np.float64)
    links["correlation"] = links["correlation"].astype(np.float64)
    if links.duplicated(["TF", "target"]).any():
        raise ValueError("adjacencies repeats a TF-target link")
    links["_row"] = np.arange(len(links), dtype=np.int64)
    rho = links["correlation"].to_numpy()
    with np.errstate(invalid="ignore"):
        regulation = (rho > parameters.rhoThreshold).astype(int) - (
            rho < -parameters.rhoThreshold
        ).astype(int)
    importance = links["importance"].to_numpy()
    cutoffs = (
        np.quantile(importance, parameters.thresholds, method="linear")
        if len(links)
        else np.asarray([], dtype=np.float64)
    )

    rows: list[pd.DataFrame] = []
    directions = [("activating", regulation > 0)]
    if not parameters.keepOnlyActivating:
        directions.append(("repressing", regulation < 0))
    for direction, mask in directions:
        subset = links.loc[mask]
        if subset.empty:
            continue
        families: list[tuple[str, pd.DataFrame]] = []
        for fraction, cutoff in zip(parameters.thresholds, cutoffs, strict=True):
            families.append(
                (f"weight>{fraction * 100}%", subset.loc[subset["importance"] > cutoff])
            )
        for n in parameters.topTargets:
            families.append((f"top{n}", _first_largest(subset, "TF", n)))
        for n in parameters.topRegulators:
            families.append((f"top{n}perTarget", _first_largest(subset, "target", n)))
        for context, members in families:
            if members.empty:
                continue
            for tf, group in members.groupby("TF", sort=True):
                genes = dict(
                    zip(group["target"].tolist(), group["importance"].tolist())
                )
                genes[str(tf)] = 1.0
                if len(genes) < parameters.minGenes:
                    continue
                rows.append(
                    pd.DataFrame(
                        {
                            "source": f"{tf}|{direction}|{context}",
                            "TF": str(tf),
                            "direction": direction,
                            "context": context,
                            "target": list(genes),
                            "weight": list(genes.values()),
                        }
                    )
                )
    if not rows:
        return pd.DataFrame({column: [] for column in MODULE_COLUMNS})
    return pd.concat(rows, ignore_index=True).loc[:, list(MODULE_COLUMNS)]

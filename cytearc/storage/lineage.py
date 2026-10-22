import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

import networkx as nx
import zarr

from .artifacts import ArtifactStatus, inspect_artifact
from .refs import ArtifactLocator, ArtifactRef, ExternalArtifactRef

type LineageTarget = ArtifactRef | Mapping[str, ArtifactRef]


_DETAIL_ITEM_LIMIT = 12
_DETAIL_VALUE_LIMIT = 160
_OMITTED = object()


def _ref_sort_key(ref: ArtifactRef) -> tuple[str, str, str, str]:
    return (ref.scope, ref.assay or "", ref.kind, ref.artifact_id)


def _locator_sort_key(
    locator: ArtifactLocator,
) -> tuple[str, str, str, str, str, str, str]:
    if isinstance(locator, ExternalArtifactRef):
        return (
            "external",
            locator.dataset_fingerprint,
            locator.anchor_assay or "",
            *_ref_sort_key(locator.ref),
        )
    return ("local", "", "", *_ref_sort_key(locator))


def _input_path(parts: tuple[str, ...]) -> str:
    value = ""
    for part in parts:
        if part.startswith("["):
            value += part
        elif value:
            value += f".{part}"
        else:
            value = part
    return value or "input"


def _artifact_inputs(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[str, ArtifactLocator]]:
    if isinstance(value, ExternalArtifactRef):
        yield _input_path(path), value
        return
    if isinstance(value, ArtifactRef):
        yield _input_path(path), value
        return
    if isinstance(value, Mapping):
        if value.get("type") == "external_artifact":
            try:
                external_ref = ExternalArtifactRef.from_dict(value)
            except (TypeError, ValueError) as exc:
                location = _input_path(path)
                raise ValueError(
                    f"Invalid external artifact reference at lineage input {location!r}"
                ) from exc
            yield _input_path(path), external_ref
            return
        if value.get("type") == "artifact":
            try:
                artifact_ref = ArtifactRef.from_dict(value)
            except (TypeError, ValueError) as exc:
                location = _input_path(path)
                raise ValueError(
                    f"Invalid artifact reference at lineage input {location!r}"
                ) from exc
            yield _input_path(path), artifact_ref
            return
        for key in sorted(value):
            yield from _artifact_inputs(value[key], (*path, str(key)))
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            yield from _artifact_inputs(item, (*path, f"[{index}]"))


def _external_value(value: Any) -> Any:
    if isinstance(value, ArtifactRef | ExternalArtifactRef):
        return _OMITTED
    if isinstance(value, Mapping):
        if value.get("type") in ("artifact", "external_artifact"):
            return _OMITTED
        result = {}
        for key in sorted(value):
            item = _external_value(value[key])
            if item is not _OMITTED:
                result[str(key)] = item
        return result if result else _OMITTED
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        result = {}
        for index, raw_item in enumerate(value):
            item = _external_value(raw_item)
            if item is not _OMITTED:
                result[f"[{index}]"] = item
        return result if result else _OMITTED
    return value


def _external_inputs(inputs: Mapping[str, Any] | None) -> dict[str, Any]:
    if inputs is None:
        return {}
    value = _external_value(inputs)
    return {} if value is _OMITTED else cast(dict[str, Any], value)


def _normalize_outputs(target: LineageTarget) -> dict[str, ArtifactRef]:
    if isinstance(target, ArtifactRef):
        return {"output": target}
    if not isinstance(target, Mapping):
        raise TypeError("lineage target must be an ArtifactRef or a named mapping")
    if not target:
        raise ValueError("lineage output mapping cannot be empty")
    outputs = {}
    for name, ref in target.items():
        if not isinstance(name, str) or not name:
            raise TypeError("lineage output names must be non-empty strings")
        if not isinstance(ref, ArtifactRef):
            raise TypeError(f"lineage output {name!r} must be an ArtifactRef")
        outputs[name] = ref
    return dict(sorted(outputs.items()))


def _normalize_external_roots(
    external_roots: Mapping[str, zarr.Group] | None,
) -> dict[str, zarr.Group]:
    if external_roots is None:
        return {}
    if not isinstance(external_roots, Mapping):
        raise TypeError("external_roots must be a mapping or None")
    roots = {}
    for fingerprint, root in external_roots.items():
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("External root fingerprints must be non-empty strings")
        if not isinstance(root, zarr.Group):
            raise TypeError(
                f"External root for fingerprint {fingerprint!r} must be a Zarr group"
            )
        roots[fingerprint] = root
    return roots


def _check_external_root(
    root: zarr.Group,
    locator: ExternalArtifactRef,
) -> None:
    """Raise unless ``root`` holds the dataset that ``locator`` names."""
    assay_name = locator.fingerprint_assay
    if assay_name not in root or not isinstance(root[assay_name], zarr.Group):
        raise ValueError(
            f"External root for dataset fingerprint "
            f"{locator.dataset_fingerprint!r} is missing assay group "
            f"{assay_name!r}"
        )
    assay = root[assay_name]
    received = assay.attrs.get("dataset_fingerprint")
    if received is None:
        raise ValueError(
            f"External root assay {assay_name!r} has no stored "
            f"dataset_fingerprint; expected {locator.dataset_fingerprint!r}"
        )
    if received is not None and received != locator.dataset_fingerprint:
        raise ValueError(
            f"External root assay {assay_name!r} dataset fingerprint mismatch. "
            f"Expected {locator.dataset_fingerprint!r}, received {received!r}"
        )


def _inspect_node(
    root: zarr.Group,
    locator: ArtifactLocator,
    external_roots: Mapping[str, zarr.Group],
) -> tuple[ArtifactStatus | None, list[tuple[str, ArtifactLocator]], str | None]:
    """Return one node's status, its inputs, and why it is corrupt, if it is.

    An external input without a supplied root has no status. A node whose
    record cannot be read, such as one whose ``artifact_id`` attribute names
    another artifact, or whose provenance holds a malformed reference, is
    corrupt: it keeps the error and is not followed further. An external root
    that does not hold the named dataset is the caller's error and raises.
    """
    if isinstance(locator, ExternalArtifactRef):
        node_root = external_roots.get(locator.dataset_fingerprint)
        if node_root is None:
            return None, [], None
        _check_external_root(node_root, locator)
        ref = locator.ref
    else:
        node_root, ref = root, locator
    try:
        status = inspect_artifact(node_root, ref)
    except (KeyError, TypeError, ValueError) as error:
        return None, [], f"{type(error).__name__}: {error}"
    try:
        inputs = list(_artifact_inputs(status.inputs))
    except ValueError as error:
        cause = f" ({error.__cause__})" if error.__cause__ is not None else ""
        return status, [], f"{type(error).__name__}: {error}{cause}"
    return status, inputs, None


def _build_graph(
    root: zarr.Group,
    outputs: Mapping[str, ArtifactRef],
    external_roots: Mapping[str, zarr.Group],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    visited: set[ArtifactLocator] = set()
    # An explicit stack, not recursion, so a chain of any depth is traced.
    pending: list[ArtifactLocator] = [
        *sorted(set(outputs.values()), key=_ref_sort_key, reverse=True)
    ]
    while pending:
        locator = pending.pop()
        if locator in visited:
            continue
        visited.add(locator)
        status, inputs, error = _inspect_node(root, locator, external_roots)
        graph.add_node(locator, status=status, error=error, outputs=())
        dependencies = sorted(
            inputs, key=lambda item: (item[0], _locator_sort_key(item[1]))
        )
        for input_name, input_locator in dependencies:
            dependency: ArtifactLocator = input_locator
            if isinstance(locator, ExternalArtifactRef) and isinstance(
                input_locator, ArtifactRef
            ):
                # A local input of an external artifact lives in the same
                # external datastore, which the same assay fingerprint names.
                anchor = locator.fingerprint_assay
                dependency = ExternalArtifactRef(
                    dataset_fingerprint=locator.dataset_fingerprint,
                    ref=input_locator,
                    anchor_assay=None if input_locator.assay == anchor else anchor,
                )
            if graph.has_edge(dependency, locator):
                labels = set(graph.edges[dependency, locator]["inputs"])
                labels.add(input_name)
                graph.edges[dependency, locator]["inputs"] = tuple(sorted(labels))
            else:
                graph.add_edge(dependency, locator, inputs=(input_name,))
            if dependency not in visited:
                pending.append(dependency)

    output_names: dict[ArtifactRef, list[str]] = defaultdict(list)
    for name, ref in outputs.items():
        output_names[ref].append(name)
    for locator in graph:
        names = output_names[locator] if isinstance(locator, ArtifactRef) else []
        graph.nodes[locator]["outputs"] = tuple(sorted(names))

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Artifact provenance contains a dependency cycle")
    return nx.freeze(graph)


def _status_label(status: ArtifactStatus | None, error: str | None = None) -> str:
    if error is not None:
        return "corrupt"
    if status is None:
        return "unresolved external"
    if not status.exists:
        return "missing"
    if not status.complete:
        return "incomplete"
    if status.superseded_by:
        return "stale"
    return "complete"


def _revision_detail(status: ArtifactStatus) -> str | None:
    """Describe a recorded revision other than a current revision 1."""
    revision, current = status.revision, status.current_revision
    if not status.complete or revision is None or current is None:
        return None
    if revision < current:
        changes = "; ".join(entry.change for entry in status.superseded_by)
        return f"{revision}, current {current}: {changes}"
    if revision > current:
        return f"{revision}, newer than revision {current} of this CyteArc release"
    return None if revision == 1 else str(revision)


def _located_ref(locator: ArtifactLocator) -> ArtifactRef:
    if isinstance(locator, ExternalArtifactRef):
        return locator.ref
    return locator


def _display_scope(locator: ArtifactLocator) -> str:
    ref = _located_ref(locator)
    scope = ref.assay if ref.assay is not None else "datastore"
    if isinstance(locator, ExternalArtifactRef):
        return f"external {locator.dataset_fingerprint[:12]} / {scope}"
    return ref.assay if ref.assay is not None else "datastore"


def _mermaid_text(value: str) -> str:
    return " ".join(value.replace('"', "'").replace("\\", "/").split())


def _detail_value(value: Any) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    text = " ".join(text.replace("`", "'").split())
    if len(text) > _DETAIL_VALUE_LIMIT:
        return f"{text[: _DETAIL_VALUE_LIMIT - 3]}..."
    return text


def _detail_mapping(values: Mapping[str, Any]) -> str:
    items = sorted(values.items())
    shown = [
        f"{key}={_detail_value(value)}" for key, value in items[:_DETAIL_ITEM_LIMIT]
    ]
    hidden = len(items) - len(shown)
    if hidden:
        shown.append(f"... {hidden} more")
    return "; ".join(shown)


class ArtifactLineage:
    """A snapshot of the upstream lineage of artifact outputs."""

    __slots__ = ("_graph", "_outputs")

    def __init__(
        self,
        graph: nx.DiGraph,
        outputs: Mapping[str, ArtifactRef],
    ) -> None:
        self._graph = graph
        self._outputs = MappingProxyType(dict(outputs))

    @classmethod
    def from_store(
        cls,
        root: zarr.Group,
        target: LineageTarget,
        *,
        external_roots: Mapping[str, zarr.Group] | None = None,
    ) -> "ArtifactLineage":
        outputs = _normalize_outputs(target)
        roots = _normalize_external_roots(external_roots)
        return cls(_build_graph(root, outputs, roots), outputs)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def outputs(self) -> Mapping[str, ArtifactRef]:
        return self._outputs

    def _ordered_refs(self) -> list[ArtifactLocator]:
        return list(
            nx.lexicographical_topological_sort(
                self._graph,
                key=_locator_sort_key,
            )
        )

    def to_mermaid(self) -> str:
        locators = self._ordered_refs()
        node_ids = {
            locator: f"artifact{index}" for index, locator in enumerate(locators)
        }
        lines = ["flowchart LR"]
        for locator in locators:
            status = cast(
                ArtifactStatus | None,
                self._graph.nodes[locator]["status"],
            )
            ref = _located_ref(locator)
            error = cast(str | None, self._graph.nodes[locator]["error"])
            if status is not None:
                operation = status.operation or "operation unavailable"
            elif error is not None:
                operation = "record unreadable"
            else:
                operation = "external source unresolved"
            label_parts = [
                f"{_display_scope(locator)} / {ref.kind}",
                operation,
                ref.artifact_id[:12],
            ]
            output_names = cast(
                tuple[str, ...],
                self._graph.nodes[locator]["outputs"],
            )
            if output_names:
                label_parts.append(f"outputs: {', '.join(output_names)}")
            status_name = _status_label(status, error)
            if status_name == "stale":
                assert status is not None
                label_parts.append(
                    f"status: stale (revision {status.revision}, current "
                    f"{status.current_revision})"
                )
            elif status_name != "complete":
                label_parts.append(f"status: {status_name}")
            label = _mermaid_text(" | ".join(label_parts))
            lines.append(f'    {node_ids[locator]}["{label}"]')

        edges = sorted(
            self._graph.edges,
            key=lambda edge: (node_ids[edge[1]], node_ids[edge[0]]),
        )
        for source, target in edges:
            input_names = cast(
                tuple[str, ...],
                self._graph.edges[source, target]["inputs"],
            )
            label = _mermaid_text(", ".join(input_names))
            lines.append(f'    {node_ids[source]} -->|"{label}"| {node_ids[target]}')
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            "```mermaid",
            self.to_mermaid(),
            "```",
            "",
            "### Artifact details",
        ]
        for locator in self._ordered_refs():
            status = cast(
                ArtifactStatus | None,
                self._graph.nodes[locator]["status"],
            )
            ref = _located_ref(locator)
            error = cast(str | None, self._graph.nodes[locator]["error"])
            lines.extend(
                [
                    "",
                    (
                        f"#### {_display_scope(locator)} / {ref.kind} / "
                        f"{ref.artifact_id[:12]}"
                    ),
                    f"- Status: `{_status_label(status, error)}`",
                ]
            )
            if isinstance(locator, ExternalArtifactRef):
                lines.append(f"- Dataset fingerprint: `{locator.dataset_fingerprint}`")
            if error is not None:
                text = " ".join(error.replace("`", "'").split())
                lines.append(f"- Error: `{text}`")
            if status is None:
                if error is None:
                    lines.append(
                        "- Resolution: `No matching external root was supplied`"
                    )
                continue
            lines.append(f"- Path: `{status.path}`")
            if status.operation is not None:
                lines.append(f"- Operation: `{status.operation}`")
            revision_detail = _revision_detail(status)
            if revision_detail is not None:
                text = " ".join(revision_detail.replace("`", "'").split())
                lines.append(f"- Revision: `{text}`")
            output_names = cast(
                tuple[str, ...],
                self._graph.nodes[locator]["outputs"],
            )
            if output_names:
                lines.append(
                    "- Outputs: "
                    + ", ".join(
                        f"`{name.replace('`', chr(39))}`" for name in output_names
                    )
                )
            parameters = status.parameters or {}
            if parameters:
                lines.append(f"- Parameters: `{_detail_mapping(parameters)}`")
            execution_options = status.execution_options or {}
            if execution_options:
                lines.append(
                    f"- Execution options: `{_detail_mapping(execution_options)}`"
                )
            external_inputs = _external_inputs(status.inputs)
            if external_inputs:
                lines.append(f"- Other inputs: `{_detail_mapping(external_inputs)}`")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def __repr__(self) -> str:
        return (
            f"ArtifactLineage(outputs={len(self._outputs)}, "
            f"artifacts={self._graph.number_of_nodes()}, "
            f"dependencies={self._graph.number_of_edges()})"
        )

from .base_datastore import BaseDataStore
from .namespaces import _Accessor, _bind

__all__ = ["ArtifactAccessor"]


class ArtifactAccessor(_Accessor):
    """Read and discover the store's saved results."""

    list = _bind(BaseDataStore._artifacts_list)
    inspect = _bind(BaseDataStore._artifacts_inspect)
    load = _bind(BaseDataStore._artifacts_load)
    load_values = _bind(BaseDataStore._artifacts_load_values)
    lineage = _bind(BaseDataStore._artifacts_lineage)
    find = _bind(BaseDataStore._artifacts_find)

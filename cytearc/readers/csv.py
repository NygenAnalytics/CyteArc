import codecs
import csv
import itertools
import re
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.io.common import get_handle

from ..utils.count_values import CountValueRange
from ..utils.files import (
    COMPRESSION_SIGNATURE_BYTES,
    TAR_SIGNATURE_BYTES,
    compression_of,
    is_tar,
)
from ..utils.progress import iter_progress
from ._text import as_text, require_unique_identifiers
from ..utils.arguments import bool_argument, quoted_value

# read_csv settings that, besides the separator and ``quoting``, decide how
# pandas splits a line into fields. The field check splits with the same ones.
_QUOTING_SETTINGS = ("quotechar", "doublequote", "escapechar", "skipinitialspace")
# read_csv settings that split lines in ways Python's csv module cannot follow.
_UNSUPPORTED_SETTINGS = frozenset({"comment", "dialect", "lineterminator"})
# read_csv settings that the reader sets from its own arguments, with the
# argument that sets each. delimiter, and delim_whitespace before pandas 3,
# set the separator as sep does.
_READER_SETTINGS = {
    "chunksize": "batch_size",
    "delim_whitespace": "sep",
    "delimiter": "sep",
    "header": "has_header",
    "index_col": "id_column",
    "sep": "sep",
    "skiprows": "skip_rows",
}
# The most repeated texts that the refusal of repeated feature names lists.
_NAMED_REPEATS = 3
# The error handler with which the reader finds the bytes that the file's
# encoding cannot decode, and the lone surrogates that stand for them.
_UNDECODABLE_BYTES = "cytearc.csv.undecodable_bytes"
_MARKED_BYTES = re.compile("([\udc00-\udcff]+)")
# The compressions of a tar archive that pandas' tar reader decompresses
# itself, None for an archive that is not compressed.
_TAR_COMPRESSIONS = (None, "gzip", "bz2", "xz")
# The name that the reader gives pandas for the ID column when names in
# pandas_kwargs leaves that column unnamed or names it by something other than
# text. pandas looks a converter's key up as a column name, then as a
# position, so only a name of text reaches the ID column alone.
_ID_COLUMN_LABEL = "__cytearc_cell_ids__"


class _FieldText:
    """One field of the ID column, as the file holds it.

    read_csv reads text such as ``NA`` or ``null`` as a missing value, and
    its python engine does so for what a converter returns too, unless that
    is not text. The ID column's converter wraps each field in this class,
    so that every field stays an ID.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def _count_dtype(frame: pd.DataFrame, value_range: CountValueRange) -> np.dtype[Any]:
    """Return the dtype of one chunk of count columns after validating it.

    The chunk's values are added to ``value_range``.
    """
    for dtype in frame.dtypes:
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(
            dtype
        ):
            raise ValueError(
                "CSV count columns must contain numbers; move text columns to "
                "cell_data_cols or skip_cols"
            )
    if frame.shape[1] == 0:
        # An assay without features cannot be opened, so the import stops here.
        raise ValueError(
            "CSV file contains no count columns; every column is the ID column "
            "or is listed in cell_data_cols or skip_cols"
        )
    # Column reductions avoid copying the chunk into one array.
    minimum = frame.min(axis=0, skipna=False).to_numpy(dtype=np.float64)
    maximum = frame.max(axis=0, skipna=False).to_numpy(dtype=np.float64)
    if not (np.isfinite(minimum).all() and np.isfinite(maximum).all()):
        raise ValueError("CSV counts contain missing or non-finite values")
    if bool((minimum < 0).any()):
        raise ValueError("CSV counts must not be negative")
    if value_range.integral:
        value_range.update(frame.to_numpy())
    return np.dtype(np.result_type(*frame.dtypes))


def _metadata_dtype(series: pd.Series) -> np.dtype[Any]:
    """Return a numeric dtype for numeric columns and ``object`` for text."""
    dtype = series.dtype
    resolved: np.dtype[Any] = np.dtype(object)
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        resolved = np.dtype(dtype)
    return resolved


def _merged_dtype(current: np.dtype[Any] | None, dtype: np.dtype[Any]) -> np.dtype[Any]:
    if current is None:
        return dtype
    if current.kind == "O" or dtype.kind == "O":
        return np.dtype(object)
    return np.result_type(current, dtype)


def _id_column_argument(id_column: Any) -> int | None:
    """Return a validated ``id_column``: None, or a column's position from 0.

    Raises:
        TypeError: If ``id_column`` is neither None nor an integer.
        ValueError: If ``id_column`` is negative.
    """
    if id_column is None:
        return None
    if isinstance(id_column, bool) or not isinstance(id_column, int | np.integer):
        raise TypeError(
            "id_column must be None or the position of the ID column, an "
            f"integer, not {id_column!r}"
        )
    if id_column < 0:
        raise ValueError(
            f"id_column counts columns from 0, so it cannot be {id_column}"
        )
    return int(id_column)


def _names_for_pandas(names: Any, n_fields: int, id_column: int | None) -> Any:
    """Return ``names`` of pandas_kwargs as the reader passes it to read_csv.

    ``names`` names every column of the file or, with ``id_column`` 0, every
    column after the ID column, which pandas reads as an unnamed index then.
    The reader names such an ID column, and one that ``names`` names by
    something other than text, ``_ID_COLUMN_LABEL``; other names stay as
    given.

    Raises:
        TypeError: If ``names`` is not an ordered collection of names.
        ValueError: If ``names`` names another number of columns, or a column
            other than the ID column by bytes that are not UTF-8 text.
    """
    if not pd.api.types.is_list_like(names, allow_sets=False):
        raise TypeError(
            f"names in pandas_kwargs must list the column names in order, not {names!r}"
        )
    labels = list(names)
    unnamed_id_column = id_column == 0 and len(labels) == n_fields - 1
    if unnamed_id_column:
        labels.insert(0, _ID_COLUMN_LABEL)
    if len(labels) != n_fields:
        columns = (
            "every column, or every column after the ID column,"
            if id_column == 0
            else "every column"
        )
        raise ValueError(
            f"names in pandas_kwargs names {len(names)} columns, but the rows of "
            f"the CSV file have {n_fields} fields; name {columns} in the file's order"
        )
    # The reader decodes the names of the other columns as UTF-8, and names
    # the ID column itself.
    for position, label in enumerate(labels):
        if position == id_column or not isinstance(label, bytes):
            continue
        try:
            label.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                "CSV column names must be UTF-8 text; names in pandas_kwargs names "
                f"the column at position {position}, counted from 0, "
                f"{quoted_value(bytes(label))}"
            ) from None
    if id_column is not None and not isinstance(labels[id_column], str):
        labels[id_column] = _ID_COLUMN_LABEL
    elif not unnamed_id_column:
        return names
    return labels


def _named_positions(
    option: str,
    names: list[Any],
    labels: list[Any],
    texts: list[str],
    file_positions: list[int],
) -> list[int]:
    """Return the positions of the columns that ``names`` name, in their order.

    A name is a column's label as pandas holds it, in ``labels``, or else its
    text, in ``texts``, as :meth:`CSVReader.feature_ids` gives it. The
    positions index ``labels``; messages give the columns' ``file_positions``.

    Raises:
        KeyError: If a name names no column.
        ValueError: If a name is no column's label but the text of several.
    """
    positions: list[int] = []
    absent: list[Any] = []
    for name in names:
        matches = [position for position, label in enumerate(labels) if name == label]
        if not matches:
            matches = [position for position, text in enumerate(texts) if name == text]
        if len(matches) > 1:
            raise ValueError(
                f"{option} names {name!r}, which is the text of the columns at "
                f"positions {[file_positions[p] for p in matches]}; name one of "
                "them by its label as pandas holds it"
            )
        if matches:
            positions.append(matches[0])
        else:
            absent.append(name)
    if absent:
        raise KeyError(f"{option} are not CSV columns: {absent}")
    return positions


def _blank_or_missing(label: Any) -> bool:
    """Return whether a column's name is missing, such as NaN, or blank text."""
    if pd.api.types.is_scalar(label) and pd.isna(label):
        return True
    return not as_text(label).strip()


def _listing(items: list[str]) -> str:
    """Join ``items`` as a sentence lists them: ``a``, ``a and b``, or ``a, b and c``."""
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _repeated_texts(
    texts: list[str], positions: list[int]
) -> list[tuple[str, list[int]]]:
    """Return each text of several columns with their positions, in the file's order."""
    columns: dict[str, list[int]] = {}
    for text, position in zip(texts, positions, strict=True):
        columns.setdefault(text, []).append(position)
    return [(text, places) for text, places in columns.items() if len(places) > 1]


def _mark_undecodable_bytes(error: UnicodeError) -> tuple[str, int]:
    """Decode each byte that the encoding cannot decode as U+DC00 plus the byte.

    A decode gives no lone surrogates otherwise, so these mark such bytes
    alone. The ``surrogateescape`` handler marks bytes alike, but only from
    0x80, which leaves out bytes of UTF-16 and UTF-32 text.
    """
    if not isinstance(error, UnicodeDecodeError):
        raise error
    undecodable = error.object[error.start : error.end]
    return "".join(chr(0xDC00 + byte) for byte in undecodable), error.end


codecs.register_error(_UNDECODABLE_BYTES, _mark_undecodable_bytes)


def _field_bytes(field: str, encoding: str) -> bytes:
    """Return a field that holds marked bytes as the bytes of the file.

    The marked bytes are the file's own, and the text around them is encoded
    again in ``encoding``.
    """
    encoder = codecs.getincrementalencoder(encoding)()
    # Some encoders start with a byte order mark, which a field does not hold.
    encoder.encode("")
    # The text and the runs of marked bytes alternate, from a run of text.
    parts = _MARKED_BYTES.split(field)
    return b"".join(
        bytes(ord(mark) - 0xDC00 for mark in part)
        if index % 2
        else encoder.encode(part)
        for index, part in enumerate(parts)
    )


class CSVReader:
    """A class to read in data from a CSV file.

    Construction checks that every row has as many fields as the header, or
    as the first row of a file without one, and raises ValueError otherwise:
    pandas reads in chunks, pads a short row with missing values, and drops
    the extra fields of a row that starts a chunk. Python's csv module counts
    the fields with ``sep`` and the ``quotechar``, ``quoting``,
    ``doublequote``, ``escapechar``, and ``skipinitialspace`` settings that
    pandas splits the rows with.

    Args:
        csv_fn: Path to the CSV file
        has_header: Whether the first row names the columns. Without a
                    header, ``names`` in ``pandas_kwargs`` can name them.
                    (Default value: True)
        id_column: The position of the column that holds the cell IDs,
                   counted from 0. Its fields are the IDs as the file holds
                   them, so ``001`` stays ``"001"`` and ``NA`` is an ID; an
                   empty or blank ID raises ValueError. (Default value: None)
        rows_are_cells: If True then each row represents a cell and hence each column is a feature. If False then each
                        row is feature and each column in a cell
        sep: The column separator in the CSV file, one character. (Default value: ',')
        skip_rows: Number of rows to skip from the top of the file. (Default value: 0)
        skip_cols: Names of columns to skip. Must be provided as a list even if just one column.
                   A name is a column's label as pandas holds it, or its
                   text, as :meth:`feature_ids` gives it, so ``"10"`` also
                   names a column that ``names`` in ``pandas_kwargs`` names
                   ``10``. A name of no column raises KeyError.
                   (Default value: None)
        cell_data_cols: Names of columns to include in cell metadata rather than count matrix. Must be provided as a
                       list even if just one column. Columns are found as
                       for ``skip_cols``, and each name, which the cell
                       metadata column takes, must be text.
                       (Default value: None)
        batch_size: Number of lines to read at a time. Decrease this value if you have too many columns.
                    (Default value: 50,000)
        pandas_kwargs: A dictionary of keyword arguments to be passed to Pandas read_csv function.
                       It cannot set ``comment``, ``dialect``, ``lineterminator``, or
                       ``usecols``, nor what the reader sets from its own arguments:
                       ``sep``, ``delimiter`` and ``delim_whitespace`` from ``sep``,
                       ``header`` from ``has_header``, ``skiprows`` from ``skip_rows``,
                       ``chunksize`` from ``batch_size``, ``index_col`` from
                       ``id_column``, and a converter of the ID column, which the reader
                       reads as the text of its fields. ``names`` names every column of
                       the file, with or without a header, or with ``id_column=0`` every
                       column after the ID column. Without ``compression`` here, the
                       file's first bytes decide whether it is gzip, bzip2, xz, Zstandard
                       or ZIP-compressed, or a tar archive, whatever its name. The file
                       must be text in ``encoding`` here, or UTF-8 without it: bytes that
                       this encoding cannot decode raise ValueError, naming the first line
                       that holds them, unless pandas reads them under
                       ``encoding_errors`` here.
        temp_dir: Parent directory of the temporary files of an import that
                  shuffles cells. None uses the system temporary directory.

    Attributes:
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        countDtype: Dtype that holds the count values of every row.
        countRange: Range of the count values of every row, which decides
                    their storage dtype.
        cellDataDtypes: Dtype of each ``cell_data_cols`` column across every
                        row; ``object`` marks a text column.
        tempDir: Parent directory of the temporary files of an import that
                 shuffles cells, or None for the system temporary directory.
    """

    def __init__(
        self,
        csv_fn: str,
        has_header: bool = True,
        id_column: int | None = None,
        rows_are_cells: bool = True,
        sep: str = ",",
        skip_rows: int = 0,
        skip_cols: list[str] | None = None,
        cell_data_cols: list[str] | None = None,
        batch_size: int = 10000,
        pandas_kwargs: dict[str, Any] | None = None,
        *,
        temp_dir: str | Path | None = None,
    ) -> None:
        has_header = bool_argument(has_header, "has_header")
        self.tempDir = temp_dir
        rows_are_cells = bool_argument(rows_are_cells, "rows_are_cells")
        self._fn = csv_fn
        if rows_are_cells is False:
            raise NotImplementedError(
                "Currently CyteArc supports only those CSV files where cells are along the rows"
            )
        if pandas_kwargs is None:
            pandas_kwargs = {}
        elif not isinstance(pandas_kwargs, dict):
            raise TypeError("pandas_kwargs must be a dictionary")
        if len(sep) != 1:
            raise ValueError("sep must be one character")
        unsupported = sorted(_UNSUPPORTED_SETTINGS.intersection(pandas_kwargs))
        if unsupported:
            raise ValueError(
                f"pandas_kwargs cannot set {', '.join(unsupported)}: the reader "
                "checks the fields of every row with sep and the quoting "
                "settings alone"
            )
        if "usecols" in pandas_kwargs:
            raise ValueError(
                "pandas_kwargs cannot set usecols: pandas then counts columns "
                "among those it selects, where id_column and names count every "
                "column of the file; leave columns out with skip_cols"
            )
        overridden = sorted(set(_READER_SETTINGS).intersection(pandas_kwargs))
        if overridden:
            arguments = list(dict.fromkeys(_READER_SETTINGS[key] for key in overridden))
            raise ValueError(
                f"pandas_kwargs cannot set {', '.join(overridden)}, which the reader "
                f"sets from its own arguments; pass {_listing(arguments)} to "
                "CSVReader instead"
            )
        self._idColumn = _id_column_argument(id_column)
        names = pandas_kwargs.get("names")
        # The header or names in pandas_kwargs name the columns.
        self._namedColumns = has_header or names is not None
        self.skipCols = [] if skip_cols is None else skip_cols
        self.cellDataCols = [] if cell_data_cols is None else cell_data_cols
        if not self._namedColumns and (self.skipCols or self.cellDataCols):
            raise ValueError(
                "Named columns require a CSV header or names in pandas_kwargs"
            )
        unnamed = [name for name in self.cellDataCols if not isinstance(name, str)]
        if unnamed:
            raise ValueError(
                f"cell_data_cols names columns by {unnamed!r}, but a cell metadata "
                "column takes its name from cell_data_cols, which must be text; "
                f"name these columns by their text, {[as_text(name) for name in unnamed]!r}"
            )
        # A copy, so that the reader's settings stay out of the caller's dict.
        self.pandas_kwargs: dict[str, Any] = dict(pandas_kwargs)
        self.pandas_kwargs["sep"] = sep
        self.pandas_kwargs["header"] = 0 if has_header else None
        self.pandas_kwargs["skiprows"] = skip_rows
        self.pandas_kwargs["chunksize"] = batch_size
        # The ID column is read as a column and taken out of each chunk, since
        # pandas reads the text NA in an index as a missing value.
        self.pandas_kwargs["index_col"] = None
        if "compression" not in self.pandas_kwargs:
            # pandas infers compression from the file's name; the file's first
            # bytes decide here, as they decide for every reader.
            self.pandas_kwargs["compression"] = self._compression()
        n_fields = self._read_text(self._check_field_counts)
        if n_fields is None:
            raise ValueError("CSV file contains no data rows")
        if self._idColumn is not None and self._idColumn >= n_fields:
            raise ValueError(
                f"id_column is {self._idColumn}, but the rows of the CSV file have "
                f"{n_fields} fields, at positions 0 to {n_fields - 1}"
            )
        # The names that the caller gives the ID column, which the reader may
        # replace with its own.
        id_names: list[Any] = []
        if names is not None:
            self.pandas_kwargs["names"] = _names_for_pandas(
                names, n_fields, self._idColumn
            )
            if self._idColumn is not None and len(names) == n_fields:
                id_names.append(list(names)[self._idColumn])
        converter_keys = list(self.pandas_kwargs.get("converters") or {})
        if self._idColumn is not None:
            # Cell IDs are the text of their fields: type inference would read
            # "001" as the number 1, and NA as a missing value. Without names,
            # an integer key reaches the column at that position alone, since
            # header names are text and positions name the columns otherwise.
            labels = self.pandas_kwargs.get("names")
            key = self._idColumn if labels is None else labels[self._idColumn]
            converters = dict(self.pandas_kwargs.get("converters") or {})
            converters[key] = _FieldText
            self.pandas_kwargs["converters"] = converters
        (
            self.nCells,
            self.nFeatures,
            self.cellIds,
            self.featureIds,
            self.keepCols,
            self.cellDataDtypes,
            self.cellDataIdx,
            self.countDtype,
            self.countRange,
        ) = self._read_text(lambda: self._consistency_check(converter_keys, id_names))

    def _compression(self) -> str | None:
        """Return the file's compression by its first bytes, as pandas names it.

        A tar archive, whose first member's header follows its compression,
        is ``"tar"``, since pandas' tar reader decompresses it too.
        """
        head = self._first_bytes(None)
        compression = compression_of(head[:COMPRESSION_SIGNATURE_BYTES])
        if compression in _TAR_COMPRESSIONS:
            text = head if compression is None else self._first_bytes(compression)
            if is_tar(text):
                return "tar"
        return compression

    def _first_bytes(self, compression: str | None) -> bytes:
        """Return the file's first bytes, decompressed as ``compression`` names.

        The bytes are read through the handle that ``read_csv`` opens, so a
        path that pandas opens with ``storage_options`` is read alike.
        """
        with get_handle(
            self._fn,
            "rb",
            compression=compression,
            is_text=False,
            storage_options=self.pandas_kwargs.get("storage_options"),
        ) as handles:
            head: bytes = handles.handle.read(TAR_SIGNATURE_BYTES)
        return head

    def _get_streamer(self) -> Generator[pd.DataFrame, None, None]:
        reader = pd.read_csv(self._fn, **self.pandas_kwargs)
        yield from reader

    def _read_text[T](self, read: Callable[[], T]) -> T:
        """Return what ``read`` returns from the file's text, refusing bytes that its encoding cannot decode.

        Raises:
            ValueError: If the file holds such bytes, naming the first line
                that holds them, from 1, and the field that holds them.
        """
        try:
            return read()
        except UnicodeDecodeError:
            found = self._undecodable_field()
            if found is None:
                # The error came from elsewhere than the file's text, such as
                # a converter.
                raise
            line, field = found
            encoding = self.pandas_kwargs.get("encoding")
            expectation = (
                "UTF-8 text"
                if encoding is None
                else "text in the encoding that pandas_kwargs names, "
                f"{quoted_value(encoding)}"
            )
            raise ValueError(
                f"CSV files must be {expectation}; line {line} of "
                f"{quoted_value(str(self._fn))} holds {quoted_value(field)}. Pass "
                "the file's encoding as encoding in pandas_kwargs"
            ) from None

    def _undecodable_field(self) -> tuple[int, bytes] | None:
        """Return the first line that holds bytes the file's encoding cannot decode, and their field.

        Lines count from 1, as the field check counts them, and the field is
        split from its line as pandas splits rows. None means that the
        encoding decodes every byte.
        """
        kwargs = self.pandas_kwargs
        encoding = kwargs.get("encoding") or "utf-8"
        with get_handle(
            self._fn,
            "r",
            encoding=encoding,
            compression=kwargs["compression"],
            errors=_UNDECODABLE_BYTES,
            storage_options=kwargs.get("storage_options"),
        ) as handles:
            for line_number, line in enumerate(handles.handle, start=1):
                if _MARKED_BYTES.search(line) is None:
                    continue
                fields = next(csv.reader([line], **self._split_settings()))
                field = next(field for field in fields if _MARKED_BYTES.search(field))
                return line_number, _field_bytes(field, encoding)
        return None

    def _split_settings(self) -> dict[str, Any]:
        """Return the settings with which Python's csv module splits rows as pandas does."""
        kwargs = self.pandas_kwargs
        settings = {key: kwargs[key] for key in _QUOTING_SETTINGS if key in kwargs}
        settings["delimiter"] = kwargs["sep"]
        # pandas reads quotes under every quoting setting but QUOTE_NONE, and
        # the csv module would convert unquoted fields under QUOTE_NONNUMERIC.
        settings["quoting"] = (
            csv.QUOTE_NONE
            if kwargs.get("quoting") == csv.QUOTE_NONE
            else csv.QUOTE_MINIMAL
        )
        return settings

    def _check_field_counts(self) -> int | None:
        """Return the fields of each row, after checking that every row has as many as the first.

        The rows are the ones pandas reads: the text that ``read_csv`` opens,
        after ``skip_rows`` rows, without blank lines. None means that there
        is no such row.

        Raises:
            ValueError: If a row has another number of fields than the first.
        """
        kwargs = self.pandas_kwargs
        # read_csv opens its source with this function, so the check reads the
        # text that pandas parses.
        with get_handle(
            self._fn,
            "r",
            encoding=kwargs.get("encoding"),
            compression=kwargs["compression"],
            errors=kwargs.get("encoding_errors", "strict"),
            storage_options=kwargs.get("storage_options"),
        ) as handles:
            lines = iter(handles.handle)
            # pandas drops a byte order mark before it splits the first line.
            first = next(lines, "").removeprefix("\ufeff")
            rows = csv.reader(
                itertools.chain((first,), lines), **self._split_settings()
            )
            expected: int | None = None
            expected_line = 0
            for row in itertools.islice(rows, kwargs["skiprows"], None):
                count = len(row)
                # pandas skips lines that hold nothing but spaces and tabs.
                if count == expected or (count <= 1 and not "".join(row).strip(" \t")):
                    continue
                if expected is not None:
                    raise ValueError(
                        f"CSV line {rows.line_num} has {count} fields, but line "
                        f"{expected_line} has {expected}"
                    )
                expected = count
                expected_line = rows.line_num
        return expected

    def _consistency_check(
        self, converter_keys: list[Any], id_names: list[Any]
    ) -> tuple[
        int,
        int,
        np.ndarray | None,
        np.ndarray | None,
        list[int] | None,
        list[np.dtype] | None,
        list[int] | None,
        np.dtype,
        CountValueRange,
    ]:
        """Stream every row once to fix the shape, names, dtypes, and count range.

        ``converter_keys`` are the keys of the caller's converters, and
        ``id_names`` the names that the caller gives the ID column.
        """
        stream = self._get_streamer()
        n_cells = 0
        n_features = 0
        feature_ids: np.ndarray | None = None
        keep_cols: list[int] | None = None
        cell_data_dtypes: list[np.dtype] | None = None
        cell_data_idx: list[int] | None = None
        count_dtype: np.dtype | None = None
        count_range = CountValueRange()
        id_fields: list[_FieldText] = []
        for df in iter_progress(
            stream,
            desc="Checking CSV consistency",
        ):
            if df.shape[0] == 0:
                # A header-only file yields one empty chunk of untyped columns.
                continue
            if n_cells == 0:
                self._check_id_column_converters(
                    list(df.columns), converter_keys, id_names
                )
            n_cells += df.shape[0]
            fields = self._pop_id_column(df)
            if fields is not None:
                id_fields.extend(fields)
            if n_features == 0:
                n_features = df.shape[1]
                if self._namedColumns:
                    # The labels as their array holds them: iterating the
                    # index would turn a float32 label into a Python float,
                    # whose text has more digits.
                    feature_ids, keep_cols, cell_data_idx = self._column_roles(
                        list(df.columns.to_numpy())
                    )
            counts = df if keep_cols is None else df.iloc[:, keep_cols]
            count_dtype = _merged_dtype(count_dtype, _count_dtype(counts, count_range))
            if cell_data_idx is not None:
                chunk_dtypes = [
                    _metadata_dtype(df.iloc[:, index]) for index in cell_data_idx
                ]
                cell_data_dtypes = (
                    chunk_dtypes
                    if cell_data_dtypes is None
                    else [
                        _merged_dtype(current, dtype)
                        for current, dtype in zip(
                            cell_data_dtypes, chunk_dtypes, strict=True
                        )
                    ]
                )
        if count_dtype is None:
            raise ValueError("CSV file contains no data rows")
        cell_ids: np.ndarray | None = None
        if self._idColumn is not None:
            # Every field is an ID, an empty one too, which is refused as blank.
            cell_ids = np.asarray([field.text for field in id_fields], dtype=str)
            require_unique_identifiers(cell_ids, "CSV cell IDs")
        if keep_cols is not None:
            n_features = len(keep_cols)
        return (
            n_cells,
            n_features,
            cell_ids,
            feature_ids,
            keep_cols,
            cell_data_dtypes,
            cell_data_idx,
            count_dtype,
            count_range,
        )

    def _pop_id_column(self, frame: pd.DataFrame) -> pd.Series | None:
        """Take the ID column out of one chunk and return it, None without one.

        The chunk keeps the other columns in the file's order.
        """
        if self._idColumn is None:
            return None
        return frame.pop(frame.columns[self._idColumn])

    def _check_id_column_converters(
        self, labels: list[Any], converter_keys: list[Any], id_names: list[Any]
    ) -> None:
        """Refuse the caller's converters for the ID column.

        pandas applies a converter to the column that its key names, or else
        to the column at its key's position, so the reader's converter of the
        ID column would replace such a converter or be replaced by it.
        ``labels`` are the names of the columns as pandas holds them, the ID
        column's included, ``converter_keys`` the keys of the caller's
        converters, and ``id_names`` the names that ``names`` gives the ID
        column, which count too, since the reader may name that column itself.

        Raises:
            ValueError: If a converter's key is a name of the ID column, or its
                position where no column has that name.
        """
        position = self._idColumn
        if position is None:
            return
        # Sets compare keys as dictionaries do, so a converter's key matches
        # what pandas finds it by.
        id_column_names = {labels[position], *id_names}
        named = set(labels)
        for key in converter_keys:
            if key in id_column_names or (key in {position} and key not in named):
                raise ValueError(
                    "pandas_kwargs cannot set a converter for the ID column, which "
                    "the reader reads as the text of its fields; remove "
                    f"converters[{quoted_value(key)}]"
                )

    def _column_roles(
        self, labels: list[Any]
    ) -> tuple[np.ndarray, list[int] | None, list[int] | None]:
        """Return the feature IDs, count columns and cell metadata columns.

        ``labels`` are the names of a chunk's columns as pandas holds them,
        the ID column taken out. The count and cell metadata columns are
        positions in ``labels``, None when every column holds counts or none
        holds cell metadata.

        Raises:
            KeyError: If ``cell_data_cols`` or ``skip_cols`` names no column.
            ValueError: If one of their names names two columns, or a count
                column's name is missing or blank, or reads as the same text
                as another count column's.
        """
        # A column's position in the file, which messages give, counts the ID
        # column too.
        file_positions = [
            position
            if self._idColumn is None or position < self._idColumn
            else position + 1
            for position in range(len(labels))
        ]
        texts = [as_text(label) for label in labels]
        cell_data = _named_positions(
            "cell_data_cols", self.cellDataCols, labels, texts, file_positions
        )
        skipped = set(
            _named_positions("skip_cols", self.skipCols, labels, texts, file_positions)
        ).union(cell_data)
        counts = [
            position for position in range(len(labels)) if position not in skipped
        ]
        unnamed = [
            file_positions[position]
            for position in counts
            if _blank_or_missing(labels[position])
        ]
        if unnamed:
            raise ValueError(
                "CSV feature IDs must contain non-empty values; the count columns "
                f"at positions {unnamed}, counted from 0, have blank or missing "
                "names. Name them through names in pandas_kwargs, or leave them "
                "out with skip_cols"
            )
        # Names that pandas holds apart, such as 10 and "10", can share a text.
        repeated = _repeated_texts(
            [texts[position] for position in counts],
            [file_positions[position] for position in counts],
        )
        if repeated:
            named = [
                f"{quoted_value(text)} at positions {places}"
                for text, places in repeated[:_NAMED_REPEATS]
            ]
            if len(repeated) > len(named):
                named.append(f"{len(repeated) - len(named)} more")
            raise ValueError(
                "CSV feature IDs must contain unique values; the names of count "
                f"columns repeat once read as text: {_listing(named)}, counted "
                "from 0. Give them names whose text differs through names in "
                "pandas_kwargs, or leave them out with skip_cols"
            )
        feature_ids = np.asarray([texts[position] for position in counts], dtype=str)
        return feature_ids, counts if skipped else None, cell_data or None

    def cell_ids(self) -> np.ndarray:
        """Return the cell IDs in the file's order, as the text that an import stores.

        The IDs are a new NumPy array of ``str`` on each call: the fields of
        the ID column as the file holds them, or ``cell_0``, ``cell_1``, and
        so on without ``id_column``. Per-cell values keyed by these IDs, such
        as ``pd.Series(values, index=reader.cell_ids())`` for values in the
        file's order, match the imported store's cells in ``ds.cells.insert``.
        """
        if self.cellIds is None:
            return np.array([f"cell_{x}" for x in range(self.nCells)])
        return self.cellIds.copy()

    def feature_ids(self) -> np.ndarray:
        """Return the feature IDs in the file's column order, as the text that an import stores.

        The IDs are a new NumPy array of ``str`` on each call: the names of
        the count columns, which ``names`` in ``pandas_kwargs`` gives, or
        else the header, or ``feature_0``, ``feature_1``, and so on without
        either. A name that is not text, such as the number ``10`` in
        ``names``, becomes its ``str`` text, ``"10"``, and a byte string is
        decoded as UTF-8. Names of bytes that are not UTF-8 text, a count
        column whose name is missing, such as ``None`` in ``names``, or blank,
        and count columns whose names read as the same text, such as ``10``
        and ``"10"``, raise ValueError when the reader is built. Values keyed
        by these IDs, such as
        ``pd.Series(values, index=reader.feature_ids())``, match the imported
        assay's features in ``feats.insert``, for example
        ``ds.RNA.feats.insert``.
        """
        if self.featureIds is None:
            return np.array([f"feature_{x}" for x in range(self.nFeatures)])
        return self.featureIds.copy()

    def consume(self) -> Generator[tuple[np.ndarray, np.ndarray | None], None, None]:
        """Returns a generator that yield chunks of data."""
        for df in self._get_streamer():
            # The ID column holds neither counts nor cell metadata.
            self._pop_id_column(df)
            if self.keepCols is None:
                yield df.values, None
            elif self.cellDataIdx is not None:
                yield (
                    df.iloc[:, self.keepCols].to_numpy(),
                    df.iloc[:, self.cellDataIdx].to_numpy(dtype=object),
                )
            else:
                yield df.iloc[:, self.keepCols].to_numpy(), None

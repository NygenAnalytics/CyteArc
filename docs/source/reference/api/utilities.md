---
myst:
  html_meta:
    description: CyteArc utility functions for loading stores, controlling computation, reading gene sets, and configuring logging and progress.
---

# Utilities API reference

For example downloads, see {doc}`datasets`. The CyteBase catalog of cloud-hosted datasets is the
separate `cytebase` package, documented at <https://docs.nygen.io/CyteBase/>.

## Store and computation helpers

```{eval-rst}
.. autofunction:: cytearc.load_zarr
```

```{eval-rst}
.. autofunction:: cytearc.controlled_compute
```

```{eval-rst}
.. autofunction:: cytearc.read_gmt
```

## Logging and progress

CyteArc starts with `INFO` logs and progress enabled, without timestamps.
Notebook sessions do not need an output setup call.
For a batch pipeline, configure the independent settings once at startup:

```python
cytearc.configure_output(progress=False, timestamps=True)
```

Arguments that are omitted retain their current values.
`configure_output` is the primary runtime interface:

- `level` controls CyteArc's log severity.
- `progress` controls progress bars, including `compute_with_progress`. Pipeline progress events
  are emitted either way ({doc}`pipeline`).
- `timestamps` controls timestamps in CyteArc's installed log sink.
- `stdlib` sends CyteArc's records to the standard-library logger `"cytearc"` instead of stdout or
  the file that `set_verbosity` chose; `stdlib=False` restores that destination.

Progress is independent of log level.
Setting a quiet level such as `WARNING` does not disable progress.
Call `configure_output(progress=False)` to suppress progress bars.
Interactive notebooks animate these bars.
The documentation shows deterministic completed snapshots from the committed notebook cache instead of replaying an animation.

```{eval-rst}
.. autofunction:: cytearc.configure_output
```

```{eval-rst}
.. autofunction:: cytearc.set_verbosity
```

`set_verbosity(level=..., filepath=...)` remains the interface for selecting a log level and optional file destination.
Use it when a durable batch log is needed; use `configure_output` for ordinary notebook and console behavior.

An application that configures the standard library's `logging`, such as a web service, can
receive CyteArc's records there:

```python
import logging

logging.basicConfig(level=logging.INFO)
cytearc.configure_output(stdlib=True)
```

With `stdlib=True`, CyteArc's sink forwards each of its records at or above `level` to the
standard-library logger named `"cytearc"`, instead of writing it to stdout or to the file that
`set_verbosity` chose. A forwarded record keeps the details of the call that logged it: its level
number, loguru's level name as `levelname`, such as `SUCCESS` or `TRACE`, its message and
exception, and the `module`, `pathname`, `lineno`, and `funcName` of that call. The
standard-library handlers then choose the format and destination, so `timestamps` has no effect. A
record logged during a pipeline stage carries the run's ID and the stage as the attributes
`cytearc_run_id` and `cytearc_stage`, which the handlers and filters of the `"cytearc"` logger can
read, such as with the format `"%(cytearc_stage)s %(message)s"`; other forwarded records carry None
in both. Handlers that an application adds to loguru find the same values in a record's `extra`
mapping. `configure_output(stdlib=False)` restores the stdout sink, or the file that
`set_verbosity(filepath=...)` chose, and `set_verbosity()` also ends the bridge, because it selects
stdout or a file. Importing CyteArc does not change the standard-library logging configuration.

CyteArc's sink shows only the records that CyteArc logs, and `configure_output` and
`set_verbosity` change only that sink, so an application or library that also logs with loguru
keeps its own handlers, levels, and formats. Loguru prints every record on stderr through its
default handler until an application removes that handler; while it is still installed, importing
CyteArc's logging replaces it with a handler of the same sink and settings, including
`LOGURU_LEVEL`, `LOGURU_FORMAT`, and `LOGURU_FILTER`, that leaves out CyteArc's records, which
CyteArc's sink already shows. Handlers that an application added receive CyteArc's records as well.

```{eval-rst}
.. py:data:: cytearc.logger

    CyteArc's `loguru` logger, bound to mark each record that it logs. Library code logs
    through this object, so :func:`cytearc.configure_output` controls the CyteArc-owned
    sink while preserving sinks added by callers.
```

```{eval-rst}
.. autofunction:: cytearc.tqdmbar
```

```{eval-rst}
.. py:data:: cytearc.tqdm_params

    Default keyword arguments applied by :func:`cytearc.tqdmbar`, including bar
    format, dynamic width, and colour. Override any of them per call.
```

```{eval-rst}
.. autofunction:: cytearc.compute_with_progress
```

## Array helpers

These operate on plain arrays and are used by CyteArc's own embeddings, graph construction, trajectory dynamics, and merge code.

```{eval-rst}
.. autofunction:: cytearc.clean_array
```

```{eval-rst}
.. autofunction:: cytearc.rescale_array
```

```{eval-rst}
.. autofunction:: cytearc.rolling_window
```

```{eval-rst}
.. autofunction:: cytearc.permute_into_chunks
```

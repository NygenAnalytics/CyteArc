# Utilities API reference

For dataset discovery, remote exploration, and example downloads, see {doc}`cytebase`.

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
- `progress` controls progress bars, including `compute_with_progress`.
- `timestamps` controls timestamps in CyteArc's installed log sink.

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

```{eval-rst}
.. py:data:: cytearc.logger

    CyteArc's `loguru` logger. Library code logs through this object, so
    :func:`cytearc.configure_output` controls the CyteArc-owned sink while preserving
    sinks added by callers.
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

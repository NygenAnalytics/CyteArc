"""The claim that one process holds while it prepares a fresh import.

A writable ``DataStore`` open prepares the assays of a fresh import: it
derives their cell and feature summaries and publishes their identity. Of the
processes that open the same fresh import at once, one prepares it. That
process holds a claim, the group record ``.cytearc-preparation`` below the
root or workspace group, which it creates with the store's conditional write
and removes when its preparation ends, however it ends. Another process that
finds the claim raises ``RuntimeError`` instead of preparing the same columns
beside it. A process that is killed while it prepares leaves its claim, which
names the process; remove the claim once that process has stopped.
"""

import datetime
import json
import os
import secrets
import socket
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import zarr
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.core.sync import sync

from ..utils.logging import logger

PREPARATION_CLAIM = ".cytearc-preparation"
"""Name of the claim group below the root or workspace group being prepared."""

_CLAIM_ATTRIBUTE = "cytearc:preparationClaim"
# A claim can disappear between this process's conditional write and its read
# when its owner finishes; each disappearance allows one more attempt.
_ATTEMPTS = 3


def _claim_document(token: str) -> Buffer:
    """Return the group record that holds a claim made by this process."""
    record = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            _CLAIM_ATTRIBUTE: {
                "token": token,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "startedAtNs": time.time_ns(),
            }
        },
    }
    return default_buffer_prototype().buffer.from_bytes(
        json.dumps(record, sort_keys=True).encode("utf-8")
    )


def _claim_holder(stored: Buffer) -> Mapping[str, Any]:
    """Return the attributes of a stored claim, or no attributes if unreadable."""
    try:
        record = json.loads(stored.to_bytes().decode("utf-8"))
        holder = record["attributes"][_CLAIM_ATTRIBUTE]
    except (KeyError, TypeError, UnicodeDecodeError, ValueError):
        return {}
    return holder if isinstance(holder, Mapping) else {}


def _held_message(claim: Any, holder: Mapping[str, Any]) -> str:
    started = holder.get("startedAtNs")
    when = (
        datetime.datetime.fromtimestamp(started / 1e9, datetime.UTC).isoformat(
            timespec="seconds"
        )
        if isinstance(started, int) and not isinstance(started, bool)
        else "an unknown time"
    )
    process = f"process {holder.get('pid', 'unknown')} on host {holder.get('host', 'unknown')}"
    return (
        f"This store is being prepared by another process ({process}, since "
        f"{when}). Open it again once that open has finished. If no process is "
        "preparing it, a preparation was killed and left its claim: remove "
        f"{claim} and open the store again."
    )


@contextmanager
def preparation_claim(group: zarr.Group) -> Iterator[None]:
    """Hold the claim to prepare the store at ``group``, a root or workspace.

    A store that cannot make the conditional write, such as an object store
    without conditional puts, is prepared without the claim, as before claims
    existed, with a warning.

    Raises:
        RuntimeError: If another process holds the claim; nothing is written.
    """
    claim = group.store_path / PREPARATION_CLAIM
    document = claim / "zarr.json"
    token = secrets.token_hex(16)
    holder: Mapping[str, Any] = {}
    claimable = True
    for _ in range(_ATTEMPTS):
        try:
            sync(document.set_if_not_exists(_claim_document(token)))
        except Exception as error:
            logger.warning(
                f"Could not claim the preparation of this store ({error}); it is "
                "prepared without a guard against another process that opens it "
                "at the same time."
            )
            claimable = False
            break
        stored = sync(document.get(prototype=default_buffer_prototype()))
        if stored is not None:
            holder = _claim_holder(stored)
            break
    if not claimable:
        # Outside the handler, so an error of the preparation is not chained
        # to the refused conditional write.
        yield
        return
    if holder.get("token") != token:
        raise RuntimeError(_held_message(claim, holder))
    try:
        yield
    finally:
        try:
            sync(claim.delete_dir())
        except Exception as error:
            # The preparation's own error, if any, is the one that propagates.
            logger.warning(
                f"Could not remove the preparation claim {claim}: {error}. If the "
                "store is not prepared, remove the claim before opening it again."
            )

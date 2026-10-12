"""Deploy the CyteBase Modal app to a chosen Modal environment and bucket.

Use ``uv run python -m cytearc.cytebase.pipeline.deploy`` for supported deployment.
This command validates a supplied ``CYTEBASE_PIPELINE_VERSION`` or resolves the
Git revision. Direct Modal app invocation bypasses this preparation and requires
the caller to supply a nonblank ``CYTEBASE_PIPELINE_VERSION``.

The target environment needs two Modal secrets: ``cytearc-env`` with ``HF_TOKEN``
and the bucket key, and ``cytebase-api`` with ``CYTEBASE_API_TOKEN``, at least
32 visible ASCII characters without whitespace, which only the web function
receives. The deployment fails while either is missing. Every environment
needs both, including one whose operators never call the HTTP API and start
work only with ``python -m cytearc.cytebase.pipeline reset-run`` or the Modal
SDK: the app deploys its web function with the workers, so ``modal deploy``
requires the ``cytebase-api`` secret. Every request to the deployed HTTP API
sends a workspace proxy auth token, created for example with
``modal workspace proxy-tokens create``, in the ``Modal-Key`` and
``Modal-Secret`` headers, and the API token in the ``Cytebase-Token`` header.
A proxy token alone is valid for every proxy-authenticated endpoint of the
workspace, so the app answers 401 without the API token. ``GET /jobs/{call_id}``
reports only calls that this deployment's API started, until Modal expires the
API's record of the call 7 days after the call started or was last polled. The
``cytearc.cytebase.pipeline.app`` module describes how to create the secret and
which jobs the API reports.

``--bucket-version 2610.1`` writes to the public release bucket
``Nygen/cytebase-2610.1`` instead of the bucket the secret names, and the secret
then needs only ``HF_TOKEN``. The app and its run state belong to the Modal
environment, not the bucket, so drain any active run before deploying another
bucket version to the same environment.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .._storage import versioned_bucket

APP_MODULE = "cytearc.cytebase.pipeline.app"


def _bucket_version(value: str) -> str:
    """Accept a ``YYMM`` or ``YYMM.N`` release version on the command line."""
    try:
        versioned_bucket(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def deploy_command(environment: str | None) -> list[str]:
    """Return the ``modal deploy`` command for the CyteBase app."""
    command = [sys.executable, "-m", "modal", "deploy"]
    if environment is not None:
        command += ["--env", environment]
    return [*command, "-m", APP_MODULE]


def deploy_settings(args: argparse.Namespace) -> dict[str, str]:
    """Return the variables ``app.py`` reads while defining the deployment.

    The bucket key and version are always set, so values left in the shell never
    select the development bucket or another release for a deployment.
    """
    settings = {
        "CYTEBASE_BUCKET_KEY": "CYTEBASE_BUCKET_DEV" if args.dev else "CYTEBASE_BUCKET",
        "CYTEBASE_BUCKET_VERSION": args.bucket_version or "",
    }
    revision = os.environ.get("CYTEBASE_PIPELINE_VERSION")
    if revision is None:
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "Set CYTEBASE_PIPELINE_VERSION when deploying without a Git revision"
            ) from error
    if not revision.strip():
        raise ValueError("CYTEBASE_PIPELINE_VERSION must not be blank")
    settings["CYTEBASE_PIPELINE_VERSION"] = revision
    if args.process_containers is not None:
        settings["CYTEBASE_PROCESS_CONTAINERS"] = str(args.process_containers)
    if args.download_connections is not None:
        settings["CYTEBASE_DOWNLOAD_CONNECTIONS"] = str(args.download_connections)
    if args.hub_api_quota is not None:
        settings["CYTEBASE_HUB_API_QUOTA"] = str(args.hub_api_quota)
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e", "--env", help="Modal environment; Modal's default when omitted"
    )
    bucket = parser.add_mutually_exclusive_group()
    bucket.add_argument(
        "--dev",
        action="store_true",
        help="Use the bucket named by CYTEBASE_BUCKET_DEV in the cytearc-env secret",
    )
    bucket.add_argument(
        "--bucket-version",
        type=_bucket_version,
        help="Use the release bucket Nygen/cytebase-VERSION, for example 2610.1",
    )
    parser.add_argument(
        "--process-containers", type=int, help="Dataset worker container limit"
    )
    parser.add_argument(
        "--download-connections", type=int, help="Source connections, 1 to 4"
    )
    parser.add_argument(
        "--hub-api-quota",
        type=int,
        help=(
            "Hub API calls the token's account may make per five minutes "
            "(default 1000); dataset starts are paced to stay under it"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the deployment without running it"
    )
    args = parser.parse_args(argv)

    settings = deploy_settings(args)
    command = deploy_command(args.env)
    print(
        shlex.join([f"{key}={value}" for key, value in settings.items()] + command),
        file=sys.stderr,
    )
    if args.dry_run:
        return 0
    return subprocess.run(
        command, env={**os.environ, **settings}, check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())

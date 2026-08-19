"""Record what produced a run, so its outputs can be traced back and reproduced."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Where to run git: the package source tree, not the process working directory,
# which may be anywhere.
_REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent


def _git(*args: str) -> str | None:
    """Run a git command in the repo, returning None if it is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def run_info() -> dict:
    """Collect provenance for the current process.

    Returns
    -------
    dict
        Command line, working directory, timestamp, and the git commit the
        package was imported from, with a ``git_dirty`` flag that is True when
        the working tree had uncommitted changes to tracked files. Git fields
        are None when git is unavailable or the source is not a checkout, in
        which case the commit simply is not recorded.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def write_run_info(path: str | Path) -> dict:
    """Write :func:`run_info` to a JSON file.

    Parameters
    ----------
    path : str or Path
        Destination file path.

    Returns
    -------
    dict
        The provenance that was written.
    """
    info = run_info()
    Path(path).write_text(json.dumps(info, indent=2) + "\n")
    return info

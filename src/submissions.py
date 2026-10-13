"""Kaggle submissions: one call to save, number and log a submission, one call to share it.

Workflow (the same for every group member):

    from src.submissions import submit, publish, record_kaggle_score

    version = submit(submission, notebook="my_notebook", model="...", cv_f1=0.771, notes="...")
    publish(version)                         # commit + push: the version number is now taken
    # ... upload the CSV on Kaggle ...
    record_kaggle_score(version, 0.77033)
    publish(version)                         # share the score

Version numbers are assigned automatically, the author comes from `git config user.name`, and
re-running a notebook with the same predictions reuses its version instead of creating a new one.
"""

import getpass
import os
import re
import subprocess
import warnings
from datetime import date

import pandas as pd

from src.paths import DATA_RAW, PROJECT_ROOT, SUBMISSIONS

LOG = PROJECT_ROOT / "docs" / "submissions.md"
TARGET = "repeat_purchase_90d"
FILE_PATTERN = re.compile(r"ML_Group_41_Version(\d+)\.csv")
COLUMNS = ["Version", "Date", "Author", "Notebook / model", "CV F1", "Kaggle public F1", "Notes"]


# ------------------------------------------------------------------ files

def submission_path(version):
    """Path of a submission file, named as the competition rules require."""
    return SUBMISSIONS / f"ML_Group_41_Version{version:02d}.csv"


def check_submission(submission):
    """Raise if the frame does not match sample_submission.csv (columns, IDs and their order, 0/1 labels)."""
    sample = pd.read_csv(DATA_RAW / "sample_submission.csv")
    if list(submission.columns) != list(sample.columns):
        raise ValueError(f"columns must be {list(sample.columns)}, got {list(submission.columns)}")
    if len(submission) != len(sample) or not submission["ID"].reset_index(drop=True).equals(sample["ID"]):
        raise ValueError("IDs must be exactly those of sample_submission.csv, in the same order")
    if not submission[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0/1 labels (apply the threshold before submitting)")


def _saved_versions():
    """{version number: path} of every submission file in submissions/."""
    return {int(m.group(1)): path for path in SUBMISSIONS.glob("*.csv")
            if (m := FILE_PATTERN.fullmatch(path.name))}


# ------------------------------------------------------------------ log table

def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _read_log():
    """Split docs/submissions.md into the text above the table and the rows ({version tag: {column: value}})."""
    intro, rows, columns = [], {}, None
    for line in LOG.read_text().splitlines():
        cells = _cells(line) if line.startswith("|") else None
        if cells and cells[0] == "Version":
            columns = cells
        elif cells and set(line) <= set("|- "):
            continue  # separator row
        elif cells and columns and cells[0].isdigit():
            row = dict(zip(columns, cells))
            rows[f"{int(row['Version']):02d}"] = {col: row.get(col, "") for col in COLUMNS}
        else:
            intro.append(line)
    while intro and not intro[-1].strip():
        intro.pop()
    return intro, rows


def _write_log(intro, rows):
    """Rewrite the table with the current columns, rows sorted by version."""
    table = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    table += ["| " + " | ".join(rows[tag][col] for col in COLUMNS) + " |" for tag in sorted(rows)]
    LOG.write_text("\n".join(intro + [""] + table) + "\n")


def _logged_versions():
    return {int(tag) for tag in _read_log()[1]}


def next_version():
    """First version number not used by any submission file or log row."""
    used = set(_saved_versions()) | _logged_versions()
    return max(used, default=0) + 1


def _author():
    name = _git("config", "user.name").stdout.strip()
    return name or getpass.getuser()


def log_submission(version, notebook, model, cv_f1, notes=""):
    """Add the row of `version` to the log, or update it. Date, author and Kaggle score already there are kept."""
    intro, rows = _read_log()
    tag = f"{version:02d}"
    old = rows.get(tag, {})
    text = lambda s: str(s).replace("|", "/").replace("\n", " ")  # keep the Markdown table intact
    rows[tag] = {
        "Version": tag,
        "Date": old.get("Date") or f"{date.today():%Y-%m-%d}",
        "Author": old.get("Author") or text(_author()),
        "Notebook / model": text(f"{notebook} · {model}"),
        "CV F1": f"{cv_f1:.3f}",
        "Kaggle public F1": old.get("Kaggle public F1", ""),
        "Notes": text(notes),
    }
    _write_log(intro, rows)
    return rows[tag]


def record_kaggle_score(version, score):
    """Write the public leaderboard score of `version` into the log."""
    intro, rows = _read_log()
    tag = f"{version:02d}"
    if tag not in rows:
        raise KeyError(f"version {tag} is not in {LOG.name}")
    rows[tag]["Kaggle public F1"] = f"{score:.5f}"
    _write_log(intro, rows)
    print(f"V{tag}: Kaggle public F1 = {score:.5f} recorded. Run publish({version}) to share it.")


# ------------------------------------------------------------------ git

def _git(*args, timeout=60):
    """Run a git command in the project, never waiting for a password prompt."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"}
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


def _git_checked(*args):
    result = _git(*args)
    if result.returncode != 0:
        raise RuntimeError(f"`git {' '.join(args)}` failed:\n{result.stderr.strip() or result.stdout.strip()}")
    return result


def check_up_to_date():
    """Raise if GitHub has commits you have not pulled: a teammate may have taken the next version number."""
    try:
        fetched = _git("fetch", "--quiet").returncode == 0
    except subprocess.TimeoutExpired:
        fetched = False
    if not fetched:
        warnings.warn("Could not reach GitHub: the version number is not checked against teammates' pushes.")
        return
    behind = _git("rev-list", "--count", "HEAD..@{u}").stdout.strip()
    if behind and int(behind) > 0:
        raise RuntimeError(f"Your branch is {behind} commit(s) behind GitHub. "
                           "Run `git pull` in the terminal, then run this cell again.")


def _relative(path):
    return str(path.relative_to(PROJECT_ROOT))


def _is_tracked(path):
    return _git("ls-files", "--error-unmatch", _relative(path)).returncode == 0


def _publish_once(version):
    """One attempt of publish(). Returns (version, pushed, message)."""
    tag = f"{version:02d}"
    path = submission_path(version)
    row = _read_log()[1].get(tag)
    if row is None:
        raise KeyError(f"version {tag} is not in {LOG.name}: run submit() first")
    predictions = pd.read_csv(path)
    new_file = not _is_tracked(path)

    # 1. Step aside: put the log back as on GitHub and move our new file away, so the pull cannot conflict.
    #    Our row and predictions are kept in memory and written back afterwards.
    _git_checked("checkout", "HEAD", "--", _relative(LOG))
    if new_file:
        path.unlink()
    pulled = _git("pull", "--rebase", "--autostash")
    if pulled.returncode != 0:
        if new_file:
            predictions.to_csv(path, index=False)
        intro, rows = _read_log()
        rows[tag] = row
        _write_log(intro, rows)
        raise RuntimeError("`git pull` failed, nothing was published (your file and log row are restored). "
                           "Fix it in the terminal (`git status`), then run publish() again.\n" + pulled.stderr.strip())

    # 2. A teammate may have published the same number in the meantime: take the next free one.
    if new_file and path.exists() and not pd.read_csv(path).equals(predictions):
        version = next_version()
        warnings.warn(f"V{tag} was taken by a teammate: your submission is now V{version:02d}. "
                      f"Upload {submission_path(version).name} on Kaggle.")
        tag, path = f"{version:02d}", submission_path(version)
        row = {**row, "Version": tag, "Kaggle public F1": ""}
    if new_file:
        predictions.to_csv(path, index=False)

    # 3. Write our row into the up-to-date log, keeping a Kaggle score a teammate may have added.
    intro, rows = _read_log()
    if not row["Kaggle public F1"] and tag in rows:
        row["Kaggle public F1"] = rows[tag]["Kaggle public F1"]
    rows[tag] = row
    _write_log(intro, rows)

    # 4. Commit only these two files and push.
    paths = [_relative(path), _relative(LOG)]
    _git_checked("add", *paths)
    if _git("diff", "--cached", "--quiet", "--", *paths).returncode == 0:
        return version, True, None
    message = f"Submission V{tag}: {row['Notebook / model']}"
    if row["Kaggle public F1"]:
        message += f" (Kaggle {row['Kaggle public F1']})"
    _git_checked("commit", "--only", *paths, "-m", message)
    if _git("push").returncode != 0:
        # Someone pushed a few seconds ago: undo our commit (files stay as they are) and let the caller retry.
        _git_checked("reset", "--soft", "HEAD~1")
        _git_checked("restore", "--staged", *paths)
        return version, False, message
    return version, True, message


def publish(version, attempts=3):
    """Share a submission: commit its file and the log (nothing else), merging teammates' work, then push.

    Safe to run again. Returns the version, which changes only if a teammate took the number first.
    """
    for _ in range(attempts):
        version, pushed, message = _publish_once(version)
        if pushed:
            print(f"V{version:02d}: nothing new to publish." if message is None
                  else f"V{version:02d} published: {message}")
            return version
    raise RuntimeError(f"Could not push after {attempts} attempts: check `git status` and your connection.")


# ------------------------------------------------------------------ main entry point

def submit(submission, notebook, model, cv_f1, notes="", check_remote=True):
    """Check, number, save and log a submission. Returns its version number.

    - Same predictions as an existing file -> that version is reused (re-running a notebook is safe).
    - Otherwise the next free version number is assigned automatically.
    - `cv_f1` should be the honest validation estimate, the number to compare with Kaggle.
    """
    check_submission(submission)
    for version, path in sorted(_saved_versions().items()):
        if pd.read_csv(path).equals(submission.reset_index(drop=True)):
            break
    else:
        if check_remote:
            check_up_to_date()
        version = next_version()
        submission.to_csv(submission_path(version), index=False)

    row = log_submission(version, notebook, model, cv_f1, notes)
    print(f"V{version:02d} saved as {submission_path(version).relative_to(PROJECT_ROOT)} and logged:")
    print("  " + " | ".join(f"{k}: {v}" for k, v in row.items() if v))
    print(f"Next: publish({version}) to reserve the version, then upload the file on Kaggle.")
    return version

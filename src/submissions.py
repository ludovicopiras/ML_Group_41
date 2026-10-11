"""Kaggle submission helpers: save the CSV and keep docs/submissions.md up to date."""

from datetime import date

import pandas as pd

from src.paths import PROJECT_ROOT, SUBMISSIONS

LOG = PROJECT_ROOT / "docs" / "submissions.md"


def submission_path(version):
    """Path of a submission file, named as the competition rules require."""
    return SUBMISSIONS / f"ML_Group_41_Version{version:02d}.csv"


def save_submission(submission, version):
    """Write the CSV. Refuses to overwrite an existing version that holds different predictions."""
    path = submission_path(version)
    if path.exists() and not pd.read_csv(path).equals(submission):
        raise FileExistsError(f"{path.name} already exists with different predictions: raise the version")
    submission.to_csv(path, index=False)
    return path


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def log_submission(version, notebook, model, cv_f1, notes=""):
    """Add the row of `version` to docs/submissions.md, or update it if it is already there.

    Re-running a notebook does not duplicate its row, and the original date and any Kaggle
    score filled in by hand are kept. Rows stay sorted by version.
    """
    tag = f"{version:02d}"
    header, rows = [], {}
    for line in LOG.read_text().splitlines():
        cells = _cells(line)
        if line.startswith("|") and cells[0].isdigit():
            rows[cells[0]] = cells
        else:
            header.append(line)

    old = rows.get(tag)
    logged_on = old[1] if old else f"{date.today():%Y-%m-%d}"
    kaggle_f1 = old[4] if old else ""
    text = lambda s: str(s).replace("|", "/")  # a "|" would break the Markdown table
    rows[tag] = [tag, logged_on, text(f"{notebook} · {model}"), f"{cv_f1:.3f}", kaggle_f1, text(notes)]

    body = ["| " + " | ".join(rows[key]) + " |" for key in sorted(rows)]
    while header and not header[-1].strip():  # drop trailing blank lines before the table body
        header.pop()
    LOG.write_text("\n".join(header + body) + "\n")
    return body[sorted(rows).index(tag)]

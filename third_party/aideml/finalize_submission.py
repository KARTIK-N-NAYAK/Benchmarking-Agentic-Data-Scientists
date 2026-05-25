#!/usr/bin/env python3
"""
Finalize submission by:
1. Finding submission.csv in logs/workspaces/**/working/
2. Moving it to data/output/
3. Deleting the entire logs directory
"""

import os
import shutil
from pathlib import Path
import sys


def finalize_submission(workspace_root: str = ".") -> None:
    """Move submissions.csv from logs to output and clean up logs."""
    workspace_root = Path(workspace_root).resolve()
    logs_dir = workspace_root / "aide_logs"
    output_dir = workspace_root / "output"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find submissions.csv in aide_logs/workspaces/**/working/
    submission_files = list(logs_dir.glob("workspaces/*/working/submission.csv"))
    
    if not submission_files:
        print(f"No submissions.csv found in aide_logs/workspaces/**/working/")
        return False
    
    if len(submission_files) > 1:
        print(f"Found multiple submissions.csv files. Using the latest one.")
        # Use the most recently modified one
        submission_file = max(submission_files, key=lambda p: p.stat().st_mtime)
    else:
        submission_file = submission_files[0]
    
    # Move to data/output/submissions.csv
    dest_file = output_dir / "submissions.csv"
    print(f"Moving {submission_file} → {dest_file}")
    shutil.copy2(submission_file, dest_file)
    print(f"Copied submissions.csv to {dest_file}")
    
    # Delete aide_logs directory
    print(f"Deleting aide_logs directory: {logs_dir}")
    shutil.rmtree(logs_dir, ignore_errors=True)
    print(f"Aide_logs directory deleted")
    
    # Verify the file exists
    if dest_file.exists():
        file_size = dest_file.stat().st_size
        print(f"Finalization complete! submissions.csv ({file_size} bytes) is ready at {dest_file}")
        return True
    else:
        print(f"Failed to move submissions.csv")
        return False


if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else "."
    success = finalize_submission(workspace)
    sys.exit(0 if success else 1)

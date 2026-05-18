# Git for Data Scientists, ML Engineers, and AI Engineers

*A production-first guide covering the Git commands, workflows, and ML-specific patterns that every data scientist and engineer must know — from daily commit hygiene to managing model artifacts and parallel experiments.*

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [A Brief History](#2-a-brief-history)
3. [Core Daily Workflow](#3-core-daily-workflow)
4. [Undoing Changes: The Safe Way](#4-undoing-changes-the-safe-way)
5. [Branches, PRs, and Collaboration](#5-branches-prs-and-collaboration)
6. [Rebase, Stash, and Advanced Operations](#6-rebase-stash-and-advanced-operations)
7. [Worktrees: Parallel Experiments](#7-worktrees-parallel-experiments)
8. [Sparse Checkout for Large Repos](#8-sparse-checkout-for-large-repos)
9. [Large Files: Git LFS vs DVC](#9-large-files-git-lfs-vs-dvc)
10. [History Inspection and Bisect](#10-history-inspection-and-bisect)
11. [ML-Specific Git Patterns](#11-ml-specific-git-patterns)
12. [The Modern Recipe](#12-the-modern-recipe)
13. [References](#13-references)

---

## 1. The Problem

A data science team without Git discipline is a team that regularly loses work, ships the wrong model version, and spends Friday afternoons tracing which preprocessing change broke the production pipeline. The failure modes are predictable: a data scientist accidentally commits 500MB model weights directly to main, bloating the repo history permanently and causing every future clone to be slow. Another engineer commits secrets to a public repo. A third runs a week of GPU experiments without any version control, then can't reproduce the model from three days ago that had better validation loss. The PM asks "which code is running in production?" and no one is certain.

Git's mental model — commits, branches, the staging area, remote refs — is not immediately intuitive, but once internalized it makes these problems straightforward to avoid. This blog covers the Git knowledge a data scientist needs daily, the ML-specific patterns for experiment tracking and artifact management, and the commands that save you when things go wrong.

---

## 2. A Brief History

Linus Torvalds created Git in 2005 over two weeks, born out of frustration when the Linux kernel project lost access to the proprietary BitKeeper VCS. His design goals were speed, distributed operation (every clone is a full repository), and data integrity via SHA-1 hashing of content. Git won over Subversion (SVN) and Mercurial through a combination of performance, branching model, and the rise of GitHub (2008), which made Git-based collaboration the default for open source.

For data science teams, Git became standard later — many teams ran notebooks and scripts in shared directories on remote machines well into the 2010s. The adoption of MLflow (2018), DVC (2017), and cloud-based notebook environments forced teams toward proper version control. Today, Git is assumed for all ML and AI engineering roles, and knowing DVC (Data Version Control) is increasingly expected for ML engineers.

---

## 3. Core Daily Workflow

### Most-Used Git Commands (by Daily Frequency)

Studies of real developer workflows consistently show the same small set of commands accounts for 90%+ of daily Git usage. Know these cold:

| Rank | Command | What It Does | When You Use It |
|---|---|---|---|
| 1 | `git status` | Shows staged, unstaged, and untracked changes | Every time before adding or committing |
| 2 | `git add <file>` | Stage specific file(s) for the next commit | After making intentional changes to a file |
| 3 | `git commit -m "..."` | Save staged changes as a new commit | After staging everything for a logical unit of work |
| 4 | `git pull --rebase` | Fetch remote changes and rebase your local commits on top | At start of day and before pushing |
| 5 | `git push` | Send local commits to the remote branch | After committing, to share or back up work |
| 6 | `git log --oneline` | View recent commit history in compact form | When checking what changed recently |
| 7 | `git diff` | Show line-by-line changes not yet staged | Before staging, to review what you changed |
| 8 | `git checkout -b <branch>` | Create and switch to a new branch | Start of any new feature, fix, or experiment |
| 9 | `git stash` | Save uncommitted changes temporarily | When you need to switch branches without committing |
| 10 | `git merge` / `git rebase` | Integrate changes from one branch into another | Bringing feature branches up to date or combining work |

### Scenario → Command Mapping

**"I want to start a new feature without affecting main"**
```bash
git checkout -b feature/user-embeddings   # create + switch to new branch
```

**"I made changes across 5 files but only want to commit 2 of them"**
```bash
git add src/features.py src/utils.py      # stage only specific files
git status                                # verify what's staged
git commit -m "feat: add user embedding feature"
```

**"Someone else pushed to main — I need their changes before I push mine"**
```bash
git pull --rebase origin main             # rebase your commits on top of remote changes
```

**"I committed too early and want to add one more change to that commit"**
```bash
git add forgotten_file.py
git commit --amend --no-edit              # add to last commit (only if not pushed yet!)
```

**"I want to see exactly what I changed in the last 3 commits"**
```bash
git log --oneline -5                      # see last 5 commits
git show abc1234                          # see full diff of a specific commit
git diff HEAD~3..HEAD                     # diff of last 3 commits combined
```

**"I need to quickly save my work-in-progress to test something else"**
```bash
git stash push -m "WIP: feature X halfway done"
git checkout main                         # work on something else
git stash pop                             # restore your in-progress work
```

---

The commands you will type every single day:

```bash
# See what changed
git status                    # overview of all changes
git diff                      # unstaged changes (line by line)
git diff --staged             # staged changes (what will be committed)

# Stage and commit
git add src/features.py       # stage specific file (NEVER git add .)
git add -p                    # stage specific hunks interactively
git commit -m "feat: add 3-month rolling avg features for user spend"

# Sync with remote
git pull --rebase origin main # fetch + rebase (cleaner than merge)
git push origin feature/my-feature

# Check where you are
git log --oneline --graph --decorate  # visual branch history
git branch -a                         # all branches (local + remote)
```

### 3.1 The .gitignore for ML Projects

```
# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints/
*.ipynb  # often exclude notebooks from core repo, track in separate branch

# Data and models (use DVC or artifact storage instead)
data/raw/
data/processed/
models/
*.pkl
*.pt
*.onnx
*.h5

# Experiment tracking
mlruns/
wandb/
outputs/
logs/

# Secrets
.env
*.pem
credentials.json
secrets.yaml

# OS
.DS_Store
.idea/
```

> 🏭 **Production note**: Use `git add -p` (patch mode) instead of `git add .` for every commit in ML projects. Notebook outputs, large log files, and temporary data often sneak into `git add .` commits. Interactive patch staging lets you review every hunk before staging.

---

## 4. Undoing Changes: The Safe Way

**When you need this:** you committed something wrong, staged the wrong file, introduced a bug in a pushed commit, or need to roll back a bad deploy. The single most important question is: *has the change been pushed to a shared remote branch?*

This is where most engineers make mistakes. The decision tree:

```
Has the change been pushed to remote?
├── No → reset is safe
│   ├── Staged but not committed → git restore --staged <file>
│   ├── Committed locally → git reset --soft HEAD~N (keep changes)
│   └── Want to discard everything → git reset --hard HEAD (destructive!)
└── Yes → revert is the safe path
    └── git revert <commit> (creates new commit that undoes old one)
```

```bash
# Undo unstaged file changes (discard local edits)
git restore path/to/file.py
git restore .                     # all files (careful!)

# Unstage a file (keep the changes, just remove from staging area)
git restore --staged path/to/file.py

# Undo last local commit, KEEP the changes (soft reset)
git reset --soft HEAD~1           # changes go back to staged
git reset --mixed HEAD~1          # changes go back to unstaged (default)
git reset --hard HEAD~1           # DESTROYS the changes — be careful

# Undo a pushed commit SAFELY (adds a new revert commit)
git revert HEAD                   # reverts last commit
git revert abc1234                # reverts specific commit
git push origin main              # push the revert commit

# Revert a range (e.g., last 3 commits)
git revert HEAD~3..HEAD           # reverts 3 commits one by one
```

### 4.1 Recovering Lost Work with reflog

`git reflog` records every HEAD movement — even commits you thought you lost with a hard reset.

```bash
# Find the lost commit
git reflog
# Output:
# abc1234 HEAD@{0}: reset: moving to HEAD~1
# def5678 HEAD@{1}: commit: add important feature  ← the lost one
# ...

# Restore it
git checkout def5678              # detached HEAD state
git checkout -b recover-lost-work # save it as a branch
```

> 🎯 **Interview prep**: "You accidentally ran `git reset --hard HEAD~3` and lost 3 commits. How do you recover?" — Use `git reflog` to find the SHA of the lost commits, then `git checkout <sha>` and create a new branch from there.

---

## 5. Branches, PRs, and Collaboration

### 5.1 Feature Branch Workflow

```bash
# Create feature branch from latest main
git checkout main
git pull --rebase origin main
git checkout -b feature/campaign-response-v2

# Work on your feature...
git add -p
git commit -m "feat(features): add 6-month trailing engagement window

Adds recency-weighted aggregation for user engagement events.
Uses exponential decay with half-life=90 days.
Validated: no leakage detected for dates in test period."

# Push and open PR
git push -u origin feature/campaign-response-v2
gh pr create --fill --draft  # draft PR to get early feedback
```

### 5.2 Good Commit Messages

The Git conventional commits format used by most ML teams:
```
<type>(<scope>): <short description>

<longer description if needed>

Breaking change: <if applicable>
```

Types: `feat` (new feature), `fix` (bug fix), `refactor`, `test`, `docs`, `chore` (dependency updates), `perf` (performance).

For ML projects, include in the commit body: what changed, why, and key metrics if relevant.

### 5.3 Checking Out a Single File from Another Branch

```bash
# Grab a specific config file from main without switching branches
git checkout origin/main -- config/model_config.yaml

# Grab an entire directory
git checkout origin/main -- src/features/
```

---

## 6. Rebase, Stash, and Advanced Operations

### 6.1 Rebase vs Merge

**Merge** preserves the full branch history with a merge commit. **Rebase** replays your commits on top of the target branch, producing a linear history.

```
MERGE result:
  A---B---C (main)
       \   \
        D---E---M (merge commit)

REBASE result:
  A---B---C---D'---E' (linear, clean)
```

```bash
# Update your feature branch with latest main changes
git fetch origin
git rebase origin/main

# If there are conflicts during rebase:
# 1. Fix conflicts in files
git add resolved_file.py
git rebase --continue    # continue to next commit
# or:
git rebase --abort       # go back to pre-rebase state

# Interactive rebase: clean up history before PR
git rebase -i HEAD~3    # edit/squash/reword last 3 commits
```

### 6.2 Stash

```bash
# Save dirty working directory without committing
git stash push -u -m "wip: half-done feature window logic"

# List stashes
git stash list
# stash@{0}: On feature/x: wip: half-done feature window logic

# Apply and pop
git stash pop              # apply and remove from stash list
git stash apply stash@{0} # apply without removing

# Stash only specific files
git stash push -m "save this" -- src/feature_eng.py
```

### 6.3 Interactive Staging

```bash
# Stage only specific changes within a file
git add -p src/features.py

# Options: y (stage), n (skip), s (split hunk), e (edit hunk), q (quit)
# This is the right way to make clean, focused commits
```

---

## 7. Worktrees: Parallel Experiments

`git worktree` lets you have multiple working directories from the same repository simultaneously. This is the ML engineer's superpower for running parallel experiments without recloning.

```bash
# You're on main, running a production model training
# Now you need to test a different feature set simultaneously

# Create worktree for experiment branch
git worktree add ../exp-feature-v2 -b experiment/feature-v2

# Now you have:
# ./                     ← original, on main
# ../exp-feature-v2/     ← new, on experiment/feature-v2

# Run experiments in parallel
# Terminal 1 (main repo):
python train.py --config config/baseline.yaml

# Terminal 2 (worktree):
cd ../exp-feature-v2
python train.py --config config/exp_features.yaml

# List all worktrees
git worktree list

# Remove when done
git worktree remove ../exp-feature-v2
git branch -d experiment/feature-v2
```

> 🏭 **Production note**: Worktrees are far better than recloning for ML experiments. Each worktree shares the same git object store (`.git/` directory) so there's no disk duplication of code history. The only duplication is the working files themselves.

---

## 8. Sparse Checkout for Large Repos

When a monorepo is 20GB but you only need 2 folders:

```bash
# Partial clone: don't download large blobs (model weights, data files)
git clone --filter=blob:none --sparse https://github.com/org/repo

cd repo

# Specify which folders to checkout
git sparse-checkout set 06-machine-learning/scripts 12-llm-inferencing

# Add more folders later
git sparse-checkout add 09-rag

# See what's included
git sparse-checkout list

# Go back to full checkout
git sparse-checkout disable
```

---

## 9. Large Files: Git LFS vs DVC

Git was designed for code — text files that diff well. Model weights (`.pt`, `.pkl`, `.h5`), datasets (`.csv`, `.parquet`), and embeddings are binary, don't diff meaningfully, and can be hundreds of gigabytes.

### 9.1 Git LFS

Git LFS stores large file content in a separate server and puts a small pointer file in the Git repository. It integrates transparently — you still do `git add`, `git commit`, `git push`.

```bash
# Install
git lfs install

# Track large file types
git lfs track "*.pt"
git lfs track "*.pkl"
git lfs track "*.parquet"
git add .gitattributes  # must commit this

# Use exactly as normal
git add models/best_model.pt
git commit -m "add best model checkpoint"
git push  # LFS handles large files automatically

# Check what's tracked
git lfs ls-files
git lfs status
```

**Limits**: GitHub Free: 1GB LFS storage, 1GB/month bandwidth. Large ML artifacts quickly exceed this. Git LFS also doesn't handle versioning multiple model checkpoints well — it just tracks the latest.

### 9.2 DVC (Data Version Control): The ML-First Approach

DVC is purpose-built for ML artifacts. It stores files in your own remote storage (S3, GCS, Azure, local server) and tracks them with lightweight `.dvc` pointer files in Git.

```bash
# Install
pip install dvc dvc-s3

# Initialize (inside a git repo)
dvc init
git commit -m "init dvc"

# Add remote storage (S3 example)
dvc remote add -d myremote s3://my-bucket/dvc-storage
git add .dvc/config && git commit -m "add dvc remote"

# Track a dataset
dvc add data/raw/users.parquet
# Creates data/raw/users.parquet.dvc (pointer file, goes to Git)
# Adds data/raw/users.parquet to .gitignore automatically
git add data/raw/.gitignore data/raw/users.parquet.dvc
git commit -m "add user dataset"

# Push data to remote storage
dvc push  # uploads actual data to S3

# Track a model
dvc add models/lgbm_v3.pkl
git add models/lgbm_v3.pkl.dvc && git commit -m "add lgbm v3"
dvc push

# Later: restore exact data + model for any git commit
git checkout <commit_sha>
dvc pull  # downloads the exact data that corresponds to that code commit
```

**DVC Pipeline**: track the full experiment as a reproducible pipeline.

```yaml
# dvc.yaml — defines the pipeline
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - data/raw/users.parquet
      - src/prepare.py
    outs:
      - data/processed/features.parquet

  train:
    cmd: python src/train.py
    deps:
      - data/processed/features.parquet
      - src/train.py
    outs:
      - models/model.pkl
    metrics:
      - metrics/eval.json
```

```bash
dvc repro        # run only stages that need re-running
dvc metrics show # compare metrics across commits
```

> 🎯 **Interview prep**: "How do you version datasets and model artifacts in ML projects?" — Use Git for code; DVC for data and model artifacts with cloud storage (S3/GCS) as the backend. `.dvc` pointer files go into Git, creating an exact correspondence between code commits and data versions. Avoid storing large files directly in Git — it permanently bloats the repo history.

| | Git LFS | DVC |
|---|---|---|
| Best for | Medium files, simple versioning | Large ML datasets, models, pipelines |
| Storage backend | LFS server (GitHub/Bitbucket) | Your own S3/GCS/Azure |
| Pipeline tracking | No | Yes — `dvc run`, `dvc repro` |
| Experiment comparison | No | Yes — `dvc metrics show` |
| Setup complexity | Low | Medium |
| Cost control | Limited (quota-based) | Full control |

---

## 10. History Inspection and Bisect

```bash
# Visual branch history
git log --oneline --graph --decorate --all

# Find who changed a specific line
git blame src/features.py -L 45,60  # lines 45-60

# Show all changes in a commit
git show abc1234

# Find which commit introduced a bug (binary search)
git bisect start
git bisect bad           # current commit is broken
git bisect good v1.2.0   # this release was good
# Git checks out the midpoint commit
# Test it, then:
git bisect good          # or: git bisect bad
# Git narrows down... eventually identifies the exact bad commit
git bisect reset         # back to normal

# Search commit history
git log --grep="CUPED"                    # commits mentioning CUPED
git log -p -- src/statistics.py           # all changes to a file
git log --author="Nitish" --since="1 month ago"
```

---

## 11. ML-Specific Git Patterns

### 11.1 Pre-commit Hooks

Pre-commit hooks run automatically before each commit, catching formatting and quality issues before they reach the repo.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
      - id: nbstripout  # strips Jupyter notebook outputs before commit
```

```bash
pip install pre-commit
pre-commit install    # registers the hooks
pre-commit run --all-files  # run on all files once
```

### 11.2 Tagging Model Versions

```bash
# Tag a model version after training
git tag -a v1.3.0-lgbm -m "LightGBM v3: AUC=0.87, deployed 2024-01-15"
git push origin v1.3.0-lgbm

# List tags
git tag --list "v*" --sort=-version:refname

# Find the tag for a deployed model
git describe --tags HEAD
```

### 11.3 Experiment Branch Naming Convention

```
feature/<description>          → new feature for production
experiment/<model>-<hypothesis> → exploratory, may be deleted
fix/<bug-description>           → bug fixes
chore/<maintenance-task>        → non-functional changes

Examples:
experiment/lgbm-v4-3m-window
experiment/neural-net-tabular-prototype
feature/add-recency-decay-feature
fix/rolling-window-leakage
```

---

## 12. The Modern Recipe

Git discipline for ML teams in 2025:

1. **Never commit large files to Git directly**: use DVC for data and model artifacts (>10MB). Use `.gitignore` liberally.

2. **Use worktrees for parallel experiments**: instead of recloning or juggling `git stash` and branch switches, create a worktree per experiment. Each gets its own working directory but shares history.

3. **Pre-commit hooks are non-negotiable**: `black` for formatting, `nbstripout` for notebook outputs, `flake8` for linting. Prevents noisy diffs and enforces quality at the source.

4. **Commit messages as experiment logs**: include key metrics, what changed, and why in commit bodies. `git log --oneline` should give a readable experiment history.

5. **Revert, don't reset, for pushed commits**: `git revert` is safe on shared branches. `git reset --hard` is only for local branches.

6. **Sparse checkout for monorepos**: if working in a large platform repo, only pull the folders you need.

**Command cookbook**:

| Situation | Command |
|---|---|
| Undo last local commit, keep changes | `git reset --soft HEAD~1` |
| Undo pushed commit safely | `git revert HEAD && git push` |
| Run two experiments in parallel | `git worktree add ../exp2 -b experiment/v2` |
| Get one file from main | `git checkout origin/main -- path/to/file` |
| Pull only needed folders from huge repo | `git clone --filter=blob:none --sparse <url>` |
| Track model checkpoint | `dvc add models/model.pkl && git add models/model.pkl.dvc` |
| Recover lost work | `git reflog` then `git checkout <sha>` |
| Find which commit broke something | `git bisect start/bad/good` |

---

## 13. References

### Official Documentation
- [git-revert](https://git-scm.com/docs/git-revert)
- [git-worktree](https://git-scm.com/docs/git-worktree)
- [Sparse checkout](https://git-scm.com/docs/git-sparse-checkout)
- [GitHub LFS docs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [DVC documentation](https://dvc.org/doc)

### Key Guides
- [Atlassian Git tutorials](https://www.atlassian.com/git/tutorials) — the most comprehensive free Git reference
- [Pro Git book (free)](https://git-scm.com/book/en/v2) — the definitive Git resource
- [Conventional Commits specification](https://www.conventionalcommits.org/)
- [pre-commit framework](https://pre-commit.com/)

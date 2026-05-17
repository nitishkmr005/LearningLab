# 05 — Git

Production-first study outline for Git workflows used by data scientists, ML engineers, and AI engineers.

---

## Format Used In This Outline
- `Concept`: what to learn.
- `Why it matters`: the day-to-day workflow reason.
- `Typical command`: the command pattern you should know.

## 01 — Core Daily Workflow
- `Concept`: `status`, `add`, `commit`, `pull`, `push`, `diff`, `.gitignore`.
- `Why it matters`: most DS work is small experiments, notebook changes, scripts, and config updates.
- `Typical command`: `git status` and `git diff` before every commit.

## 02 — Undo Uncommitted Changes
- `Concept`: discard or restore local file edits safely.
- `Why it matters`: useful when a notebook, config, or script is half-broken locally.
- `Typical command`: `git restore path/to/file.py`

## 03 — Undo Local Commits That Are Not Pushed
- `Concept`: `git reset --soft HEAD~1`, `git reset --mixed HEAD~1`.
- `Why it matters`: common when you committed too early or grouped the wrong files together.
- `Typical command`: `git reset --soft HEAD~1`
- `Example`: keep your code changes, remove the last local commit, recommit with a better message or smaller scope.

## 04 — Undo Changes Already Pushed To Remote
- `Concept`: `git revert` creates a new commit that undoes an older one.
- `Why it matters`: this is the safe default for shared branches and already-pushed work.
- `Typical command`: `git revert <commit_sha>`
- `Example`: a bad preprocessing change reached `main`; revert it without rewriting shared history.

## 05 — Recover Lost Work
- `Concept`: `git reflog`.
- `Why it matters`: when you reset, rebase, or delete a branch and panic.
- `Typical command`: `git reflog`

## 06 — Pull Only Part Of A Huge Repo
- `Concept`: partial clone and sparse checkout.
- `Why it matters`: large monorepos or research repos are expensive to clone fully.
- `Typical command`: `git clone --filter=blob:none --sparse <repo_url>`
- `Follow-up`: `git sparse-checkout set 05-git 06-machine-learning`
- `Example`: pull only the training and feature folders from a multi-team platform repo, edit them, then commit and push as normal.

## 07 — Pull Only One File Or Folder From Another Branch
- `Concept`: checkout specific paths from another ref.
- `Why it matters`: useful when you need a single config, notebook, or utility from `main`.
- `Typical command`: `git checkout origin/main -- path/to/file_or_folder`

## 08 — Worktrees
- `Concept`: multiple working directories for one repository.
- `Why it matters`: very useful for parallel ML tasks, hotfixes, and PR review without recloning.
- `Typical command`: `git worktree add ../experiment-branch -b experiment-branch`
- `Example`: keep one worktree on `main`, another on `feature/new-features`, and a third for urgent bugfix review.

## 09 — Feature Branches and PR Flow
- `Concept`: branch, commit, push, open PR, respond to review, merge.
- `Why it matters`: this is the standard collaboration path in GitHub-based teams.
- `Typical command`: `git push -u origin feature/campaign-response-v2`

## 10 — Creating PRs
- `Concept`: branch hygiene, small diffs, draft PRs, reviewable commit history.
- `Why it matters`: model changes are easier to review when code, config, and metric output are scoped tightly.
- `Typical command`: `gh pr create --fill`
- `Example`: open a draft PR with model metrics, schema changes, and rollout notes before final sign-off.

## 11 — Rebase and Update Your Branch
- `Concept`: `fetch`, `rebase`, resolve conflicts, push updated branch.
- `Why it matters`: long-lived model branches drift quickly.
- `Typical command`: `git fetch origin && git rebase origin/main`

## 12 — Stash For Temporary Context Switching
- `Concept`: save dirty changes without committing.
- `Why it matters`: useful when an urgent issue interrupts a training or feature-engineering task.
- `Typical command`: `git stash push -u -m "wip feature window logic"`

## 13 — Interactive Staging
- `Concept`: commit only the relevant hunks.
- `Why it matters`: notebooks and config files often contain unrelated noise.
- `Typical command`: `git add -p`

## 14 — Large Files and GitHub Limits
- `Concept`: Git is poor at large binary artifacts; use Git LFS for large models, datasets, and binaries.
- `Why it matters`: pushing raw checkpoints or huge CSVs directly bloats repo history and may hit hosting limits.
- `Typical command`: `git lfs track "*.pt"`
- `Example`: track model checkpoints, embeddings, and sample datasets with LFS instead of plain Git.

## 15 — Data Science Reality: Git LFS Is Not MLOps
- `Concept`: LFS helps with storage, but DVC, object storage, or model registries are usually better for large evolving datasets.
- `Why it matters`: versioning a 20 GB feature table inside Git is usually the wrong design.
- `Typical command`: keep code in Git; keep large mutable data in artifact storage.

## 16 — Useful History Inspection
- `Concept`: `log`, `show`, `blame`, `diff`, `bisect`.
- `Why it matters`: needed when a model metric regressed and you must identify the exact code change.
- `Typical command`: `git log --oneline --graph --decorate`

## 17 — Common DS/ML Scenarios You Should Practice
- `Scenario 1`: accidentally committed a secret or wrong config locally and need to uncommit before push.
- `Scenario 2`: already pushed a broken feature-engineering change and need a safe revert.
- `Scenario 3`: want two parallel experiments using the same repo without recloning.
- `Scenario 4`: need only one folder from a huge remote repository.
- `Scenario 5`: must push model files safely using Git LFS or, preferably, store them outside Git.

## 18 — Minimal Command Cookbook
- `Undo last local commit, keep code`: `git reset --soft HEAD~1`
- `Undo last pushed commit safely`: `git revert HEAD`
- `Create worktree`: `git worktree add ../my-branch -b my-branch`
- `Sparse clone huge repo`: `git clone --filter=blob:none --sparse <repo_url>`
- `Limit checkout to folders`: `git sparse-checkout set path/a path/b`
- `Get one file from remote branch`: `git checkout origin/main -- path/to/file`
- `Track large model files`: `git lfs track "*.bin"`

## References
- `git revert`: https://git-scm.com/docs/git-revert
- `git worktree`: https://git-scm.com/docs/git-worktree
- partial clone and sparse checkout: https://git-scm.com/docs/git-clone and https://git-scm.com/docs/sparse-checkout
- GitHub LFS docs: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage

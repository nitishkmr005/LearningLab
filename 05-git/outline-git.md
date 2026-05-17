# 05 — Git

Exhaustive learning path for Git: core internals, branching workflows, collaboration patterns, and advanced operations for data science and ML projects.

---

## 01 — Git Object Model & Internals
Blobs, trees, commits, tags; SHA-1 content addressing; .git directory layout; how a commit points to a tree; pack files and object storage.
- https://git-scm.com/book/en/v2/Git-Internals-Git-Objects
- https://github.blog/developer-skills/programming-languages-and-frameworks/commits-are-snapshots-not-diffs/

## 02 — Staging Area & Basic Workflow
init, add, commit, status, diff (staged vs unstaged); .gitignore patterns; partial staging with git add -p; amend last commit.
- https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository

## 03 — Branching & Merging
Branch as a pointer; create, switch, delete branches; fast-forward vs three-way merge; merge conflicts and resolution; git log --graph.
- https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging

## 04 — Rebasing
git rebase vs merge; interactive rebase (squash, fixup, reorder, edit); rebase onto; golden rule (never rebase shared history); rebase conflict resolution.
- https://git-scm.com/book/en/v2/Git-Branching-Rebasing
- https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase

## 05 — Remote Repositories
origin, upstream; clone, fetch, pull, push; tracking branches; git remote add/rename/remove; push --set-upstream; force push safety.
- https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes

## 06 — Tags & Releases
Lightweight vs annotated tags; semantic versioning (v1.2.3); push tags to remote; delete tags; GitHub releases tied to tags.
- https://git-scm.com/book/en/v2/Git-Basics-Tagging

## 07 — Undoing Changes
git restore, git reset (--soft, --mixed, --hard); git revert (safe undo for shared history); git clean; ORIG_HEAD; recovery with reflog.
- https://git-scm.com/book/en/v2/Git-Basics-Undoing-Things
- https://git-scm.com/docs/git-reflog

## 08 — Stashing & Worktrees
git stash push/pop/apply/list/drop; stash with untracked files; git worktree for parallel branches without re-cloning.
- https://git-scm.com/docs/git-stash
- https://git-scm.com/docs/git-worktree

## 09 — Git Log & History Inspection
log formatting (--oneline, --graph, --decorate); filtering (--author, --since, --grep, -S pickaxe); git blame; git bisect for bug hunting; git shortlog.
- https://git-scm.com/book/en/v2/Git-Basics-Viewing-the-Commit-History

## 10 — Branching Workflows
Git Flow (main/develop/feature/release/hotfix); GitHub Flow (main + feature PRs); trunk-based development; when to choose each; ML repo conventions.
- https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
- https://docs.github.com/en/get-started/using-github/github-flow

## 11 — Pull Requests & Code Review
PR lifecycle; draft PRs; reviewers, labels, milestones; resolving review comments; squash vs merge vs rebase merge; protected branches.
- https://docs.github.com/en/pull-requests/collaborating-with-pull-requests

## 12 — Git Hooks
client-side hooks (pre-commit, commit-msg, pre-push); server-side hooks; pre-commit framework for linting/formatting; husky for JS projects; bypassing hooks.
- https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
- https://pre-commit.com/

## 13 — Submodules & Monorepos
git submodule add/update/sync; detached HEAD in submodules; sparse checkout; monorepo tooling (Bazel, nx, turborepo); pros/cons for ML projects.
- https://git-scm.com/book/en/v2/Git-Tools-Submodules

## 14 — Large File Storage (Git LFS)
Why Git struggles with large binaries; git-lfs track, push, pull; pointer files; LFS in ML repos (models, datasets); DVC as an alternative for ML artifacts.
- https://git-lfs.com/
- https://dvc.org/doc/use-cases/versioning-data-and-models

## 15 — CI/CD Integration & Git for ML
GitHub Actions triggered on push/PR; branch protection rules; semantic-release for automated versioning; MLflow / DVC for experiment tracking alongside git history.
- https://docs.github.com/en/actions/writing-workflows/quickstart
- https://dvc.org/doc/start/experiments

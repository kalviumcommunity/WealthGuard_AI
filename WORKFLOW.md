# Team Workflow & Conventions

This document defines our team's GitHub workflow: how we branch, commit, review, and track work.

---

## 1. Branching Strategy

| Rule | Detail |
|------|--------|
| **Main branch** | Always holds releasable, working code. Nobody pushes directly to `main`. |
| **Feature branches** | All new work happens on a branch created from `main`. |
| **Naming convention** | `[type]/[short-description]` — e.g. `feature/data-ingestion`, `fix/validation-logic`, `docs/data-dictionary`. |
| **Types** | `feature`, `fix`, `docs`, `refactor`, `chore` |
| **Lifecycle** | Branch is deleted after its PR is merged. |

### Example

```bash
git checkout main
git pull origin main
git checkout -b feature/churn-model
# ... do work, commit, push ...
git push origin feature/churn-model
# open PR → review → merge → delete branch
```

---

## 2. Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/).

```
[type]: [description]

[optional body explaining *why*]
```

| Type | When to use |
|------|-------------|
| `feat` | A new feature or capability |
| `fix` | A bug fix or correction |
| `docs` | Documentation-only changes |
| `refactor` | Code cleanup without changing behaviour |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks (deps, CI, configs) |

### Why this matters

- **Readable history** — anyone can scan `git log` and understand what happened.
- **Automated changelogs** — tools can parse types to generate release notes.
- **Easier rollbacks** — you know exactly what a commit introduced.

---

## 3. Pull Request (PR) Process

1. Push your feature branch to `origin`.
2. Open a PR on GitHub targeting `main`.
3. Write a clear **title** (not "update code") and **description** (what changed, why).
4. Link the related issue: `Closes #<issue-number>`.
5. Request at least **one reviewer**.
6. Address review feedback with follow-up commits.
7. Reviewer approves → author merges → branch is deleted.

### Code Review Focus Areas

- **Correctness** — Does it do what it claims?
- **Clarity** — Can a new teammate understand this code?
- **Data integrity** — Are edge cases handled?
- **Test coverage** — Are changes tested?
- **Commit messages** — Do they follow the convention?

---

## 4. GitHub Issue Tracking

| Principle | Detail |
|-----------|--------|
| **Every change starts with an issue** | No code without a corresponding tracked task. |
| **Title** | Action-oriented — "Implement data validation for CSV intake", not "validation". |
| **Description** | Why the task exists, what success looks like, constraints. |
| **Labels** | Categorise: `feature`, `bug`, `documentation`, `high-priority`, etc. |
| **Assignee** | One person accountable per issue. |
| **Closing** | Issues are auto-closed when the linked PR merges (via `Closes #N`). |

---

## 5. Quick-Start for New Contributors

```bash
# 1. Clone the repo
git clone https://github.com/kalviumcommunity/WealthGuard_AI.git
cd WealthGuard_AI

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Copy env template and fill in keys
cp .env.example .env

# 4. Create a feature branch
git checkout -b feature/my-new-feature

# 5. Make changes, commit with conventional messages
git add .
git commit -m "feat: add my new feature"

# 6. Push and open a PR
git push origin feature/my-new-feature
# → Open PR on GitHub, link issue, request review
```

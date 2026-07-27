#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/b200/sync_main.sh [options] [-- PATH...]

Safely sync the B200 checkout with the upstream main branch:
  1. Ensure origin points to the main TurboDiffusion repository.
  2. Fetch origin/main.
  3. Commit local changes if any.
  4. Rebase local commits on top of origin/main.
  5. Push the result back to origin/main.

By default, only tracked file changes are committed. New untracked files are
left alone unless --include-untracked is passed.

Options:
  -m, --message MSG       Commit message (default: "B200 sync YYYY-MM-DD HH:MM:SS")
  --remote-url URL        Origin URL (default: https://github.com/xwqtju/TurboDiffusion.git)
  --branch BRANCH         Branch to sync (default: main)
  --include-untracked     Also stage untracked files under PATH...
  --allow-large-files     Allow newly tracked files larger than 10 MiB
  --no-push               Commit and rebase, but do not push
  --dry-run               Print actions without changing the repository
  -h, --help              Show this message

Examples:
  scripts/b200/sync_main.sh -m "Update B200 environment notes"
  scripts/b200/sync_main.sh --include-untracked -m "Add B200 scripts" -- docs scripts/b200
  scripts/b200/sync_main.sh --no-push
EOF
}

REMOTE_NAME="origin"
REMOTE_URL="https://github.com/xwqtju/TurboDiffusion.git"
BRANCH="main"
COMMIT_MESSAGE=""
INCLUDE_UNTRACKED=0
ALLOW_LARGE_FILES=0
PUSH=1
DRY_RUN=0
PATHS=()
MAX_NEW_FILE_BYTES=$((10 * 1024 * 1024))

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--message)
            COMMIT_MESSAGE="$2"
            shift 2
            ;;
        --remote-url)
            REMOTE_URL="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --include-untracked)
            INCLUDE_UNTRACKED=1
            shift
            ;;
        --allow-large-files)
            ALLOW_LARGE_FILES=1
            shift
            ;;
        --no-push)
            PUSH=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            PATHS+=("$@")
            break
            ;;
        -*)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            PATHS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$COMMIT_MESSAGE" ]]; then
    COMMIT_MESSAGE="B200 sync $(date '+%Y-%m-%d %H:%M:%S')"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

file_size_bytes() {
    local path="$1"
    if stat -c%s "$path" >/dev/null 2>&1; then
        stat -c%s "$path"
    else
        stat -f%z "$path"
    fi
}

git_ls_untracked() {
    if (( ${#PATHS[@]} > 0 )); then
        git ls-files --others --exclude-standard -- "${PATHS[@]}"
    else
        git ls-files --others --exclude-standard
    fi
}

git_status_matching() {
    if (( ${#PATHS[@]} > 0 )); then
        git status --porcelain -- "${PATHS[@]}"
    else
        git status --porcelain
    fi
}

git_status_tracked_matching() {
    if (( ${#PATHS[@]} > 0 )); then
        git status --porcelain --untracked-files=no -- "${PATHS[@]}"
    else
        git status --porcelain --untracked-files=no
    fi
}

git_add_all_matching() {
    if (( ${#PATHS[@]} > 0 )); then
        run git add -A -- "${PATHS[@]}"
    else
        run git add -A
    fi
}

git_add_tracked_matching() {
    if (( ${#PATHS[@]} > 0 )); then
        run git add -u -- "${PATHS[@]}"
    else
        run git add -u
    fi
}

ensure_not_in_rebase() {
    local rebase_merge rebase_apply
    rebase_merge="$(git rev-parse --git-path rebase-merge)"
    rebase_apply="$(git rev-parse --git-path rebase-apply)"
    if [[ -d "$rebase_merge" || -d "$rebase_apply" ]]; then
        echo "A rebase is already in progress. Resolve it first, then rerun this script." >&2
        exit 1
    fi
}

ensure_branch() {
    local current_branch
    current_branch="$(git branch --show-current)"
    if [[ "$current_branch" != "$BRANCH" ]]; then
        echo "Current branch is '$current_branch', but this script syncs '$BRANCH'." >&2
        echo "Switch branches manually if that is intended." >&2
        exit 1
    fi
}

ensure_remote() {
    local current_url
    if ! git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
        echo "Adding $REMOTE_NAME -> $REMOTE_URL"
        run git remote add "$REMOTE_NAME" "$REMOTE_URL"
        return
    fi

    current_url="$(git remote get-url "$REMOTE_NAME")"
    if [[ "$current_url" != "$REMOTE_URL" ]]; then
        echo "Updating $REMOTE_NAME from $current_url to $REMOTE_URL"
        run git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
    fi
}

check_large_untracked_files() {
    if [[ "$INCLUDE_UNTRACKED" -ne 1 || "$ALLOW_LARGE_FILES" -eq 1 ]]; then
        return
    fi

    local large_files=()
    local path size
    while IFS= read -r path; do
        [[ -f "$path" ]] || continue
        size="$(file_size_bytes "$path")"
        if (( size > MAX_NEW_FILE_BYTES )); then
            large_files+=("${path} (${size} bytes)")
        fi
    done < <(git_ls_untracked)

    if (( ${#large_files[@]} > 0 )); then
        echo "Refusing to add untracked files larger than 10 MiB:" >&2
        printf '  %s\n' "${large_files[@]}" >&2
        echo "Move them out, add them to .gitignore, or rerun with --allow-large-files." >&2
        exit 1
    fi
}

stage_and_commit() {
    check_large_untracked_files

    if [[ "$INCLUDE_UNTRACKED" -eq 1 ]]; then
        echo "Staging tracked and untracked changes."
        git_add_all_matching
    else
        echo "Staging tracked changes only. Use --include-untracked to add new files."
        git_add_tracked_matching
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        if [[ "$INCLUDE_UNTRACKED" -eq 1 ]]; then
            if [[ -n "$(git_status_matching)" ]]; then
                echo "Would create commit: $COMMIT_MESSAGE"
            else
                echo "No matching changes to commit."
            fi
        else
            if [[ -n "$(git_status_tracked_matching)" ]]; then
                echo "Would create commit: $COMMIT_MESSAGE"
            else
                echo "No matching tracked changes to commit."
            fi
        fi
        return
    fi

    if git diff --cached --quiet; then
        echo "No staged changes to commit."
        return
    fi

    echo "Creating commit: $COMMIT_MESSAGE"
    run git commit -m "$COMMIT_MESSAGE"
}

show_summary() {
    echo
    git status -sb
    echo
    git log -1 --oneline --decorate
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "This script must be run inside a Git repository." >&2
    exit 1
fi

ensure_not_in_rebase
ensure_branch
ensure_remote

echo "Fetching ${REMOTE_NAME}/${BRANCH}."
run git fetch "$REMOTE_NAME" "$BRANCH"

stage_and_commit

echo "Rebasing ${BRANCH} on ${REMOTE_NAME}/${BRANCH}."
if ! run git rebase "${REMOTE_NAME}/${BRANCH}"; then
    echo
    echo "Rebase stopped because of a conflict." >&2
    echo "Resolve conflicts, run 'git rebase --continue', then rerun this script." >&2
    exit 1
fi

if [[ "$PUSH" -eq 1 ]]; then
    echo "Pushing ${BRANCH} to ${REMOTE_NAME}/${BRANCH}."
    run git push "$REMOTE_NAME" "$BRANCH"
else
    echo "Skipping push because --no-push was set."
fi

show_summary

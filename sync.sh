#!/bin/bash
# ============================================
# Sync Script for Gitea <-> GitHub
# Linux/Mac Shell Script
# ============================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "================================================"
echo "   Speed Climbing Project - Sync Tool"
echo "================================================"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}[ERROR] Not a git repository!${NC}"
    echo "Please run this from the project root directory."
    exit 1
fi

# Parse command line argument
MODE="${1:-full}"
echo -e "${BLUE}[INFO] Mode: ${MODE}${NC}"
echo ""

# ============================================
# PULL from Gitea
# ============================================

pull_from_gitea() {
    echo "----------------------------------------"
    echo -e "${BLUE}[1/3] Pulling from Gitea (origin)...${NC}"
    echo "----------------------------------------"

    if ! git fetch origin; then
        echo -e "${RED}[ERROR] Failed to fetch from Gitea!${NC}"
        echo "Check your network connection."
        exit 1
    fi

    echo "Current branch: $(git branch --show-current)"

    if git pull origin main; then
        echo -e "${GREEN}[SUCCESS] Pulled from Gitea.${NC}"
    else
        echo -e "${YELLOW}[WARNING] Pull failed - you may have local changes.${NC}"
        echo "Run 'git status' to check."
        return 1
    fi

    echo ""
}

# ============================================
# PUSH to GitHub
# ============================================

push_to_github() {
    echo "----------------------------------------"
    echo -e "${BLUE}[2/3] Pushing to GitHub...${NC}"
    echo "----------------------------------------"

    if git push github main; then
        echo -e "${GREEN}[SUCCESS] Pushed to GitHub.${NC}"
    else
        echo -e "${YELLOW}[WARNING] Push to GitHub failed!${NC}"
        echo "You may need to pull first or force push."
        echo "Try: git push github main --force-with-lease"
        return 1
    fi

    echo ""
}

# ============================================
# Verify Sync
# ============================================

verify_sync() {
    echo "----------------------------------------"
    echo -e "${BLUE}[3/3] Verifying sync...${NC}"
    echo "----------------------------------------"

    echo "Gitea (origin):"
    git log origin/main --oneline -1

    echo ""
    echo "GitHub:"
    git log github/main --oneline -1

    echo ""

    # Compare commits
    GITEA_HASH=$(git log origin/main --oneline -1 | awk '{print $1}')
    GITHUB_HASH=$(git log github/main --oneline -1 | awk '{print $1}')

    if [ "$GITEA_HASH" = "$GITHUB_HASH" ]; then
        echo -e "${GREEN}[SUCCESS] ✓ Gitea and GitHub are in sync!${NC}"
    else
        echo -e "${YELLOW}[WARNING] ✗ Gitea and GitHub are NOT in sync!${NC}"
        echo "  Gitea:  $GITEA_HASH"
        echo "  GitHub: $GITHUB_HASH"
    fi

    echo ""
}

# ============================================
# Main execution
# ============================================

case "$MODE" in
    pull)
        pull_from_gitea
        ;;
    push)
        push_to_github
        ;;
    verify)
        verify_sync
        ;;
    full)
        pull_from_gitea
        push_to_github
        verify_sync
        ;;
    *)
        echo -e "${RED}[ERROR] Unknown mode: $MODE${NC}"
        echo ""
        echo "Usage:"
        echo "  ./sync.sh         - Full sync (pull + push + verify)"
        echo "  ./sync.sh pull    - Pull from Gitea only"
        echo "  ./sync.sh push    - Push to GitHub only"
        echo "  ./sync.sh verify  - Verify sync status only"
        exit 1
        ;;
esac

echo "================================================"
echo "                   DONE"
echo "================================================"
echo ""

if [ "$MODE" = "full" ]; then
    echo "Summary:"
    echo " - Pulled from Gitea"
    echo " - Pushed to GitHub"
    echo " - Verified sync"
    echo ""
fi

echo "Usage:"
echo "  ./sync.sh         - Full sync (pull + push + verify)"
echo "  ./sync.sh pull    - Pull from Gitea only"
echo "  ./sync.sh push    - Push to GitHub only"
echo "  ./sync.sh verify  - Verify sync status only"
echo ""

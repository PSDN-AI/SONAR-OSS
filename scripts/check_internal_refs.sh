#!/usr/bin/env bash
# Internal-reference gate.
#
# SONAR-OSS is built by importing code from a private repository. This script
# fails if any tracked file references private infrastructure or internal-only
# modules that must never appear in the public tree. It runs as a blocking CI
# check on every PR (see .github/workflows/ci.yml) so the pre-public release
# checklist in the migration plan is enforced continuously, not just once at
# the end.
#
# False positives: prefer renaming the offending identifier in the imported
# code. If a match is genuinely legitimate public content, add a narrowly
# scoped path to ALLOWLIST below with a comment explaining why.
set -euo pipefail

cd "$(dirname "$0")/.."

# This script necessarily contains the patterns it scans for.
ALLOWLIST=(
  ":(exclude)scripts/check_internal_refs.sh"
)

# Internal-only module paths: the private service/ control plane and its
# prep CLI must never be imported or referenced in the public tree.
MODULE_PATTERNS=(
  'psdn_sonar[./]service'
  'psdn_sonar[./]cli_prep'
)

# Case-insensitive internal identifiers: internal data-validation backend,
# campaign/customer artifact names, internal Helm charts, internal AWS
# account ID.
INTERNAL_PATTERNS=(
  'numo'
  'kgen'
  'batch0'
  'psdn-charts'
  '294250598750'
)

fail=0

for pattern in "${MODULE_PATTERNS[@]}"; do
  if git grep -nIE "$pattern" -- . "${ALLOWLIST[@]}"; then
    echo "::error::Internal module reference matching '$pattern' found (see above)."
    fail=1
  fi
done

for pattern in "${INTERNAL_PATTERNS[@]}"; do
  if git grep -nIiE "$pattern" -- . "${ALLOWLIST[@]}"; then
    echo "::error::Internal reference matching '$pattern' found (see above)."
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo ""
  echo "Internal-reference gate FAILED. Remove or rename the flagged content;"
  echo "see docs/import-gate.md for the import policy."
  exit 1
fi

echo "Internal-reference gate passed: no private-infrastructure references found."

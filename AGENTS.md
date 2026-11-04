# Controllers release work

Use `main` for all further work on this project release: library fixes, licensing,
notices, packaging and scan configuration. Fetch and verify the latest remote `main`
before selecting a release candidate. Reuse this checkout; do not continue the old
scanner/licensing/release branches or create separate branches for release work.
Record the immutable `main` commit for each operation and retain old evidence against
its original commit. Other feature development is outside this release workflow.

Read [docs/release-workflow.md](docs/release-workflow.md) before changing release CI.
The pipeline's `CI_COMMIT_SHA` identifies both the source candidate and its scan
configuration. Do not introduce another source branch or a hard-coded old candidate.

Keep prior branches, tags, commits, reports and decisions as historical evidence.
Do not rewrite historical results or extend an old approval/finding disposition to
a new candidate without evidence that it covers that candidate.

The owner selected the NVIDIA License dated 22 March 2022 for NVIDIA-owned material.
This licensing patch does not create human approval or authorization to publish. Retain the Franka helper's Apache-2.0 terms and the UR models' BSD terms.
Follow `CLAUDE.md` for the library's architecture and generator contracts.

The owner requires separate authorization before pushing, publishing, submitting
or editing upstream tickets, contacting Legal/support, starting jobs/scans, changing
protected access or enabling automation. A manual CI job is not authorization to run
it. Use dashboard secure setup for credentials and never place secrets in source,
logs, reports or chat. Read-only upstream investigation and local static validation
can proceed. Do not change dashboard code.

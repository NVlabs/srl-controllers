# Controllers release workflow on main

`main` is the single branch for this release. Library changes,
NVIDIA License (22 March 2022) preparation, OSS scan profiles, malware scanning and diagnostics belong here.
The earlier scan, runtime and licensing branches are historical inputs; resume work
on `main` rather than continuing them. Fetch and verify the latest remote `main`
before selecting the immutable candidate; after each authorized push, update the
dashboard candidate and active workflow bindings to that verified commit.

The source and workflow identity for a run is its full `CI_COMMIT_SHA`. OSS bootstraps
and their collectors use that same SHA. Malware packaging verifies the candidate's
Git tree and the downstream job checks its archive hash and candidate identity.
Later branch movement does not change the source attributed to an existing run.

## Prepared capabilities

The combined configuration retains 14 existing OSS profiles, restores the reviewed
malware packaging/file-scan jobs from before the auth-only diagnostic, and keeps the
SSA diagnostic and scanner runtime check. It changes their branch and source binding,
not scanner versions, reports, dependency pins or finding dispositions.

Only explicit `web`/`api` pipelines on the protected `main` branch are accepted.
Exactly one of the five mode flags must be `1`; an OSS mode also needs a valid profile. Pushes,
schedules, merge requests, other branches and ambiguous selections create no pipeline.
Every selected job is manual, including malware archive preparation. Retries are zero.
`main` uses the consolidated isolated scan workflow; normal application CI remains in
the historical main-branch commits. No application publication, package promotion or release jobs are
included. The separately authorized manual CI image refresh publishes only its
commit-specific internal candidate tag; it never promotes `latest`.

The runtime-only job clears scanner credential variables before starting Python and
retains its no-scanner-credentials guard. This allows it to share the protected branch;
its child processes still use the existing restricted environment. It makes no scanner
authentication request and performs no Controllers build or scan.

## Future execution packet

This table documents prepared configuration. It does not authorize a pipeline or job.
Before execution, verify the remote branch is the reviewed commit and is protected,
reuse existing protected project variables, and obtain authorization for the exact
mode/profile, candidate and job effects. Never reuse consumed execution approvals.

Destination:
[Run pipeline](https://gitlab-master.nvidia.com/srl/cpp/controllers/-/pipelines/new?ref=main)

Branch: `main`. Set only the selected row's variables; leave the
other four mode flags unset. No environment selection or new credentials are required
by this consolidation. Existing variable delivery and scopes still require verification
before any authorized run.

| Mode | Pipeline variables | Manual jobs |
|---|---|---|
| OSS | `CONTROLLERS_OSS_SCAN=1`, `CONTROLLERS_OSS_PROFILE=<profile below>` | `controllers_oss_<profile>` |
| Malware | `CONTROLLERS_MALWARE_SCAN=1` | `controllers_malware_archive`, then `controllers_malware_file_scan` |
| SSA diagnostic | `CONTROLLERS_MALWARE_AUTH_DIAGNOSTIC=1` | `controllers_malware_auth_diagnostic` |
| Runtime check | `CONTROLLERS_OSS_RUNTIME_CHECK=1` | `controllers_oss_runtime_check` |
| CI image security refresh | `CONTROLLERS_SCAN_IMAGE_REFRESH=1` | `controllers_scan_image_refresh` |

Valid OSS profiles:

`core_cpp`, `core_python`, `tests_tools`, `python_ci`, `ur`, `flexiv_dual_1_9_1`,
`franka_0_9_2`, `franka_0_10_0`, `franka_0_15_3`, `franka_0_21_3`,
`rizon_1_9_1`, `rizon_1_9_2`, `rizon_1_9_3`, `rizon_1_9_4`.

These are available profiles, not a claim that every profile is required or complete.
Determine required coverage from the release configurations and official dashboard
guidance. All raw findings and candidate-specific limitations remain reviewable.

Completion requires the job's artifact identities and contents to match the selected
candidate, then an assessment of package capture, module coverage, findings and the
actual official nSpect/PLC activity. A green job alone is not release approval.

## Record continuity

Use existing nSpect `NSPECT-BE6H-WKN4` / `0.12.0`, OSRB `6666725`, export `6667806`,
the existing NVLabs destination and saved connections. Consolidation does not create
another registration, ticket, approval, credential or backend disposition.

Historical results and registrations for earlier SHAs remain historical. After the
consolidated `main` commit is available remotely, update authorized dashboard
candidate and active workflow associations to the verified remote commit. Preserve historical scan registrations and reports; changing
a registration may start a scan and requires its own disclosed authorization.
Do not reset approvals or rebind historical jobs to the new candidate.

The selected first-party terms are the NVIDIA License dated 22 March 2022
(`LicenseRef-NVIDIA-2022-03-22`). The distribution also retains Apache-2.0 and
BSD-3-Clause material as documented in `THIRD_PARTY_NOTICES.md`. This is a
source-available release for research or evaluation under the complete root
`LICENSE`; it is not an open-source license. Preserve existing approval records:
a license identifier change is not a request for re-approval, an approval itself,
or authorization to publish.
The local static checks for consolidation are not a new security scan or a human
ownership, code-audit, management, export or Legal attestation.

## CI image security refresh

This preparation adds one manual image-build mode. It is separate from OSS and
malware execution; it grants no consent to start a build or scan. Reuse the saved
connections and existing GitLab job token. No SSA credentials are passed to the
builder, Docker build, package tools or build context.

The job derives the reviewed CI image using
`docker/controllers-scan-security.dockerfile`, with a minimal build context
containing only that Dockerfile and `ci/refresh_scan_system_packages.sh`. The same
package checks run at the end of the full `docker/controllers.dockerfile` CI stage.
It upgrades installed Ubuntu noble/amd64 libc, Python 3.12 and Expat package
families, then uses complete Debian revisions and `dpkg --compare-versions` to
verify at least `2.39-0ubuntu8.9`, `3.12.3-1ubuntu0.15` and
`2.6.1-2ubuntu0.6`, respectively. These minima address the reviewed fixable High
rows; they do not clear unrelated or later findings.

The remote Docker route reuses the previous successful `build_ci_images` job's
Linux runner tags and daemon API fallback. It does not restore the inherited
application, release or promotion pipelines. It builds without cache, retains the
installed package inventory and source/image identities, and pushes only
`$CI_REGISTRY_IMAGE/ci:security-$CI_COMMIT_SHA`. An existing tag or an inconclusive
registry lookup stops the job for reconciliation. There is no automatic retry.

After an authorized successful build, independently read its actual registry
digest and retained package evidence. Prepare the reviewed scanner image-pin
change using that digest; never use an invented digest or unverified `latest`.
The current OSS job pins remain unchanged until that follow-up change is reviewed
and authorized. Commit/push, nSpect/dashboard candidate synchronization and any
replacement scan require the separately authorized execution packet. Keep prior
reports and finding decisions attributed to their original candidates. Image
building and candidate pinning do not establish vulnerability or PLC completion.

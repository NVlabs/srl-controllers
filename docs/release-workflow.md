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

All 14 OSS bootstrap jobs fetch the exact pipeline commit into an empty directory
using Git, then verify `HEAD`. They do not download or extract a source tar archive.
The malware job still constructs and verifies its input archive without extracting
it to the filesystem. Historical reports retain their original producer behavior.

The CI HTTPS bootstrap uses the reviewed `ci/secure_http.py`, embedded unchanged in
15 job definitions. Curl handles response framing; credentials enter its configuration
through stdin and responses stay in memory. Requests containing credentials cannot
redirect. Downloads may redirect only over HTTPS and retain their existing hash checks.
The helper limits transfer size/time and passes neither secrets nor diagnostics in
its command line or exceptions. It does not patch the system Python interpreter.

The isolated runtime check and all OSS profiles retain actual runtime security
evidence from `ci/check_scan_runtime.py`. Its network subprocess environment fixes
`LOCALDOMAIN=.` and `LANGUAGE=C`. It checks nscd absence, the actual architecture and
Python installation, and invocable root-owned privileged ELF loader paths. TinyXML2
assertion-symbol evidence is retained when that library is installed. These are
technical checks for the inspected CI context, not a claim about every user's OS
or a new PIC approval.

Both CUDA JUnit consumers use the bounded reader in
`tools/verify_cuda_python_tests.py`. Only UTF-8 XML is accepted, DTDs are rejected,
and parsing runs in a credential-free child with a five-second wall limit, Linux
CPU/address-space limits and node/depth bounds. The parent builds Elements from
generated JSON and never reparses XML. Fixed bundled URDF readers have a separate
input boundary; no general Expat patch or unrestricted XML-safety claim is made.

## Prepared capabilities

The combined configuration retains 14 existing OSS profiles, restores the reviewed
malware packaging/file-scan jobs from before the auth-only diagnostic, and keeps the
SSA diagnostic and scanner runtime check. It changes their branch and source binding,
not scanner versions, reports, dependency pins or finding dispositions.

Only explicit `web`/`api` pipelines on the protected `main` branch are accepted.
Exactly one of the six mode flags must be `1`; an OSS mode also needs a valid profile. Pushes,
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
other five mode flags unset. No environment selection or new credentials are required
by this consolidation. Existing variable delivery and scopes still require verification
before any authorized run.

| Mode | Pipeline variables | Manual jobs |
|---|---|---|
| OSS | `CONTROLLERS_OSS_SCAN=1`, `CONTROLLERS_OSS_PROFILE=<profile below>` | `controllers_oss_<profile>` |
| Malware | `CONTROLLERS_MALWARE_SCAN=1` | `controllers_malware_archive`, then `controllers_malware_file_scan` |
| SSA diagnostic | `CONTROLLERS_MALWARE_AUTH_DIAGNOSTIC=1` | `controllers_malware_auth_diagnostic` |
| Runtime check | `CONTROLLERS_OSS_RUNTIME_CHECK=1` | `controllers_oss_runtime_check` |
| CI image security refresh | `CONTROLLERS_SCAN_IMAGE_REFRESH=1` | `controllers_scan_image_refresh` |
| Existing OSS file locations | `CONTROLLERS_OSS_SCOPE_DIAGNOSTIC=1`, `CONTROLLERS_SCOPE_SOURCE_SHA=<existing source SHA>`, `CONTROLLERS_SCOPE_PIPELINE_ID=<existing scan pipeline>`, `CONTROLLERS_SCOPE_JOB_ID=<existing scan job>` | `controllers_oss_scope_diagnostic` |

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

## Core C++ inventory delivery

The manual `core_cpp` job requests the existing client's `update:report` scope.
The CLI scan keeps `--no-override`; after complete, unchanged SBOM and vulnerability
exports, the collector submits one inventory update through the pinned official
Pulse SDK. It records the exact source, backend project/version, request and task
ID before reporting the import as submitted. Submission is not completed ingestion.
The backend is retained. An uncertain response or an existing attempted receipt
requires readback, not automatic retry. No finding disposition or approval is added.

Executing this path requires separate authorization for native report changes:
it sets Pulse Override and disables native on-demand scans for that exact source.
Verify the resulting main commit is registered before starting the job. Confirm the
imported components/findings through native readback before any separately authorized
cleanup. This is only the `core_cpp` profile, not full release coverage; other profiles
keep their current behavior and are not automatically published over this inventory.

## Non-shipped package scope

The raw compiler-capture inventory includes system packages from the CI image.
Their presence in that inventory does not establish that their installations are
bundled in the source-only release. Keep raw exports and their original candidate
identities. Assess packages distributed by optional examples separately.

The existing-scan location diagnostic reads one explicitly selected, retained
`core_cpp` backend. Its source SHA and original pipeline/job are evidence selectors,
separate from the current commit running the reader. It reuses the existing protected
SSA connection and pinned Pulse SDK. The only business-data operations are GETs for
the existing project, version, components and component/origin matched-file collections.
Authentication and the SDK's normal audit telemetry still occur. It does not build,
start a scan, create a report, modify a finding, register a source or import inventory.
Its manual execution and any push require separate authorization.

Use complete package locations from those responses to prepare a root
`.nspect-allowlist.toml` under the official
[non-shipped directory procedure](https://nvidia.atlassian.net/wiki/spaces/PRODSEC/pages/2569266896/AllowList)
and its [directory/regex field reference](https://nvidia.atlassian.net/wiki/spaces/PRODSEC/pages/2569284526/nSpect+Allowlist+for+OSS+vulnerabilities+and+Secrets).
Use the [official validator](https://nspect.nvidia.com/tools/validate/allowlist).
An absent file keeps exclusions empty. The core C++ collector uses the pinned official
Pulse parser and passes applicable directory entries to the existing import helper;
the receipt retains their exact values and policy hash. Raw local SBOM/vulnerability
exports remain unfiltered; native filtered inventory must be checked after ingestion.
This location-only hook rejects vulnerability overrides and ownership reassignment.

The prepared policy contains seven program-scoped groups for the CI installations
of glibc, libc6, libc6-dev, libgcc-13-dev, libstdc++-13-dev, libgcc-s1 and
linux-libc-dev. Their package versions match the retained installed-package inventory.
The verified source archive at `37e73415140a68d67f8639f56aa640784eef9537`
contained 322 tracked source files and no native binaries or OS packages. This
establishes the scope of that source release, not the contents of future user builds.
Keep required third-party source and optional-example dependencies visible.

[Diagnostic job 553471076](https://gitlab-master.nvidia.com/srl/cpp/controllers/-/jobs/553471076)
read all seven origin collections completely: 250 file records with no blank paths.
The policy uses 253 exact regexes: the observed file locations, the archive-context and
SDK report forms of the one archived glibc location, and two package-location
aliases present in the original retained vulnerability export. It does not append
directory suffixes to files or exclude `/usr/include/`, `/usr/lib/` or `/lib/`.
Each regex escapes one literal location and ends with `$(?!\n)`. The pinned Pulse
matcher lowercases patterns and uses multiline matching; this end guard prevents
prefix/newline matches without using an escape whose meaning changes on lowercasing.
The archived file keeps its archive identity rather than adding a bare-path rule.
The collector passes the policy entries to the existing native import, preserves
unfiltered raw exports and records the policy digest. This does not retroactively
filter an existing report merely because the policy is committed.

The original scan remains attributed to source `37e7341`, pipeline `79636715`,
job `531134192`; the diagnostic ran reader configuration `8b6a952`. Its current
backend has nine components, whereas the retained original export has eight.
In particular, the old composite C++ library mapping is now split into GCC and
GNU Standard C++ Library. The read-only diagnostic did not make that change.

All locations for an excluded package must be accounted for. A blank/missing location
is unresolved, not a wildcard. Do not remove a package by name, delete the location or
exclude a broad system root to make its row disappear. Check proposed paths against
the other components, including genuinely distributed third-party source. A scope
policy does not establish vulnerability clearance, OSRB coverage or Legal approval.
Local validation must check every observed target location, all other component
locations and suffix/newline near-misses with the pinned official parser and matcher.
The historical glibc vulnerability export also contains a blank location. Preserve
it and report that historical coverage as incomplete: the complete matched-file
collection does not erase a blank in a different report. Official browser-validator
acceptance and native post-ingestion readback remain separate from local checks.
On 17 November 2026 UTC, the official browser validator returned
`Validations Successful` / `Content is having valid format` for the complete
17,896-byte policy with SHA-256
`4b324af9fb27e9808b0eeb82ea9da1a9a2a3a6d2c09369f77f46f110f663a9ec`.
Browser input readback matched that exact file. This establishes format acceptance;
the policy has not been applied to existing inventory or scan results.
Commit/push, source registration, new reports/imports and scans require their
separately authorized execution packet. The current source-only archive evidence
does not cover a future binary release.

## Record continuity

The Docker `release` target consumes a previously built local Debian package;
it does not fetch Controllers from the unsigned internal HTTP apt repository.
For a Linux amd64 CPack build in `_build`, the explicit local test command is:

```bash
docker build -f docker/controllers.dockerfile --target release \
  --build-arg DEB_PACKAGE=_build/packages/nvidia-srl-controllers_0.12.0_amd64.deb \
  --build-arg VERSION=0.12.0 .
```

Use the actual package path and its exact Debian version if they differ. The
image checks the package name and version before installation; dependency
resolution uses the base image's configured package sources. This command is
documentation, not permission to build or publish an image.

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
containing that Dockerfile and `ci/refresh_scan_system_packages.sh`.
The build context also includes `ci/refresh_certificate_trust.sh` and
`ci/check_certificate_trust.py` for the certificate-security update below. The same
package checks run at the end of the full `docker/controllers.dockerfile` CI stage.
It upgrades installed Ubuntu noble/amd64 libc, Python 3.12, Expat and zlib package
families, then uses complete Debian revisions and `dpkg --compare-versions` to
verify at least `2.39-0ubuntu8.9`, `3.12.3-1ubuntu0.17`,
`2.6.1-2ubuntu0.6` and `1:1.3.dfsg-3.1ubuntu2.2`, respectively. The zlib minimum
addresses CVE-2026-27171 using the Ubuntu noble fix from USN-8706-1. These minima
address the reviewed fixes; they do not clear unrelated or later findings.

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

## Certificate trust update

The internal image built from `fecb14bbe1dd0eeeceeb004935c3757b0beb2429` in
[job 530831637](https://gitlab-master.nvidia.com/srl/cpp/controllers/-/jobs/530831637)
was verified against the registry digest
`sha256:d2d9c10fb18b638925a34d42caf9ad47a2c8af7dcbaa7121920b60440e59906e`.
It contains Ubuntu zlib `1:1.3.dfsg-3.1ubuntu2.3`, above the reviewed fixed revision,
and retains the checked libc, Python, Expat and certificate-trust updates. The
workflow pins this observed digest. Image verification does not resolve the old
scan's findings or establish release approval; assess the resulting candidate.

The local preparation for CVE-2026-115557 updates Ubuntu's `ca-certificates` to
at least `20260816ubuntu0.24.04.1`, refreshes installed NSS packages, and checks
that Python's default CA store no longer trusts emSign ECC Root CA - G3. It also
uses hash-pinned pip 26.2.2 and certifi 2026.10.19 wheels. The full image applies
this update before Python dependency downloads; the isolated security image
refresh applies it without changing the reviewed compiler/toolchain baseline.

Every OSS bootstrap checks its system trust before reading SSA/job credentials.
Each new scanner/project venv upgrades its seeded pip from the reviewed wheel
before resolving dependencies. The scanner and runtime-check locks use the fixed
certifi bundle. The offline checker inspects the effective system, pip-vendored,
certifi and Java stores when those components are used, and fails if the affected
root remains trusted. It makes no TLS connection and changes no trust store.
Java is inspected after installation; it is explicitly recorded as absent in an
image that does not contain it. A stale Java-only store stops the build/check;
inspect and follow the [Ubuntu mitigation](https://ubuntu.com/blog/emudhra-ca-compromise-fixes-available)
before regenerating it, preserving any separately provisioned CA certificates.

The image build retains `certificate-trust.json` and the installed CA-package
version with its existing package inventory. These records must pass validation
before the internal candidate image is pushed. This is trust-path mitigation,
not a claim that the Ubuntu Python interpreter contains the upstream denylist.
Do not mark the existing finding Resolved or Not Affected from this source patch.

The installed checker is explicitly mode `0644` in both Dockerfiles, set by a
`RUN chmod` command compatible with the runner's legacy Docker builder. Do not use
BuildKit-only `COPY --chmod`. Build-context permissions must not make the checker
root-only. Before publication, a network-disabled
container runs the checker as the image's configured non-root CI user. Its exit
status and trust result must pass and match the build-time result. The receipt
retains that user and its trust evidence; a root-only build check is insufficient.

The current image digest remains pinned until an authorized build succeeds and
its actual registry digest and trust evidence are read back. The new preflight
intentionally rejects the old image. Commit/push, image build/publication, updating
the digest pin and replacement scans each need their separately reviewed execution
step. Preserve prior scans and approvals against their original candidates, then
assess the exact delta for reuse under current policy.

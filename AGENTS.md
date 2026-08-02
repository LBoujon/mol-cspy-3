# Codex repository instructions

Read `HANDOFF.md` before making changes. It records the current scientific
goal, the relevant commits, the Aloe workflow and the last known state of the
trans-stilbene calculations.

## Working conventions

- Communicate with Lucas in Spanish unless he asks otherwise.
- Do not connect to Aloe or Neon by SSH. Ask Lucas to run commands there and
  paste the output.
- Keep Slurm scripts minimal; do not add scheduler flags unless the calculation
  requires them.
- Do not store machine-specific executable paths in `cspy.toml`. External tools
  such as Critic2 must normally be discovered from `PATH` after `conda activate`
  or `module load`.
- Preserve scientific outputs. Do not delete or overwrite databases, WAL/SHM
  files, logs or restart files unless Lucas names the exact targets.
- Do not commit large CSP databases, SQLite sidecars, logs or temporary output.
  Only the curated ACETAC examples under `bases/` are intended to be versioned.
- Inspect `git status` and `git diff` before editing. The public GitLab remote is
  `origin`; Lucas's private GitHub remote is `private`.

## Verification

The project requires Python 3.11. Run focused tests while iterating and the
full suite before committing a completed change:

```bash
python -m pytest -q
git diff --check
```

The local workstation environment used for the handoff was
`/home/lucas/TFM/git/.conda-envs/mol-cspy`; that absolute path is not portable.
On Aloe, the editable environment is named `cspy-3.0`.

Treat timestamps and cluster progress in `HANDOFF.md` as a snapshot. Ask Lucas
for fresh Aloe output before drawing conclusions about a running job.

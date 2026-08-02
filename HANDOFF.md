# mol-CSPy project handoff

Last updated: 2026-08-02 (Europe/Madrid).

This file transfers the useful context from Codex CLI session
`019fc22f-7f38-7671-ba9e-fd88b589e5a0`. The raw transcript is intentionally
not committed because it contains machine paths, terminal output and unrelated
session metadata.

## Immediate objective

The current scientific objective is to validate a flexible crystal-structure
prediction protocol for trans-stilbene starting from a gas-phase optimized
geometry. Three torsions are sampled, then two independently selected molecular
conformations are used in a `Z'=2` CSP. The current production dataset is for
space group 14 (`P21/c`). Its duplicate clustering and completeness analysis
were still running on Aloe at the time of this handoff.

Lucas wants the code to work on Aloe and Neon without hard-coded executable
paths. Codex must not log into either cluster; Lucas will execute commands and
paste results.

## Git state

- Repository: `LBoujon/mol-cspy-3` (private GitHub mirror).
- Public upstream: `https://gitlab.com/mol-cspy/mol-cspy.git` (`origin`).
- Private remote in this clone: `git@github.com:LBoujon/mol-cspy-3.git`
  (`private`).
- Active branch: `feature/completeness-critic2-flex`.
- Public base: `8d6e0f2` (`origin/main`).
- Feature commits before this handoff:
  - `384ff56 feat(db): add completeness and Critic2 validation`
  - `b825a19 fix(flex): correct multidimensional HPC scans`
  - `e02cdb2 fix(flex): harden Z-prime conformer handling`

Commit `e02cdb2` ports three fixes that had previously been made manually only
in the Aloe clone:

1. Each conformational energy penalty is now added once before averaging over
   the molecules in the asymmetric unit.
2. Duplicate temporary filenames are removed safely.
3. Cleanup for seed `1464` no longer matches and deletes files for seed `14642`.

The branch was clean before generating `AGENTS.md` and this handoff. Do not
assume Aloe's existing editable clone is clean: it contains equivalent manual
changes. Inspect its `git diff` before pulling or replacing it.

To start from a fresh clone on a computer whose SSH config has the
`github-lboujon` alias:

```bash
git clone git@github-lboujon:LBoujon/mol-cspy-3.git
cd mol-cspy-3
git switch feature/completeness-critic2-flex
```

Without that alias, use the normal GitHub SSH URL.

## Implemented code

### Completeness and clustering

- Good--Turing sample coverage and Chao lower-bound estimates.
- Parametric bootstrap confidence intervals that reserve the estimated unseen
  probability mass; naive non-parametric resampling was rejected because it
  produces misleading intervals for singleton-dominated CSP samples.
- Trial-ordered convergence curves.
- Global and per-space-group results, including multiple input databases.
- Strict validation that every valid final structure has a cluster assignment.
- Cluster representative and multiplicity fields in table exports.
- Avoidance of an unused `output.db` during structure-search-only operations.

The main implementation and tests are in:

```text
cspy/db/completeness.py
cspy/db/clustering.py
cspy/db/dump.py
cspy/db/tests/test_completeness.py
cspy/db/tests/test_clustering.py
docs/source/commands/db.rst
```

### Critic2 and PXRD

- Structural candidates can be validated with GPWDF/GVCPWDF through Critic2.
- Critic2 comparisons use isolated temporary directories, per-candidate error
  records, timeouts, atomic checkpoints, restart support and optional fail-fast.
- GVCPWDF requires Critic2 built with NLopt; this is checked before a batch.
- Normal resolution order is CLI override, `CSPY_CRITIC2_EXECUTABLE`, then
  `critic2` from the active job's `PATH`.
- No Critic2 path belongs in `cspy.toml`. Conda activation or a cluster module
  should expose the executable automatically.
- Generated PXRD descriptors record their backend. Clustering supports explicit
  `platon`, explicit `pymatgen` and `auto` fallback.

The workstation has an unversioned Critic2+NLopt build under
`../.mol-cspy-tools/critic2-gvc/bin/critic2`. It is about 825 MB together with
its sources and must not be committed. A new computer or cluster must provide
Critic2 independently if GVCPWDF is needed.

### Flexible/HPC corrections

- `cspy-moldis --help` no longer initializes MPI.
- MPI orchestration imports are lazy, so serial tools and test collection work
  without an initialized MPI fabric.
- Multidimensional Slurm scans pass each `--scan_dofs` item as a separate CLI
  argument; the obsolete nonexistent `--scan` option was removed.
- An `n`-point internal-coordinate scan now produces exactly `n` points.
- Sobol coordinates span the complete requested angular interval rather than
  only the first grid step.
- A one-component joint conformational database is copied rather than
  pointlessly rejoined.
- `Z'>1` conformational energy averaging and concurrent temporary-file cleanup
  are fixed by `e02cdb2`.

## Verification snapshot

After `e02cdb2`, the workstation suite produced:

```text
130 passed, 54 skipped, 1 warning in 11.57s
```

The warning is a pymatgen `FutureWarning` concerning a deprecated `gcd` helper.
It does not fail the tests. The focused flexible tests produced `7 passed`.

Earlier integration checks also established:

- A real Critic2+NLopt calculation reproduced the committed ACETAC scores.
- Restart reused the structural RMSD and Critic2 checkpoint without requiring
  the executable again.
- Source and wheel builds included the new completeness and Critic2 modules.
- The Aloe clone, before the last local regression tests were added, produced
  `128 passed, 54 skipped` under Python 3.11.15.

## Environment and cluster conventions

The package requires Python `>=3.11,<3.12`. Aloe uses:

```bash
eval "$(/opt/software/conda/bin/conda shell.bash hook)"
conda activate cspy-3.0
```

That environment used an editable installation from
`/home/lucas/git/mol-cspy-3`; `python -m pip check` reported no broken
requirements. If recreating it, install the clone with its test dependencies
and verify the import path:

```bash
python -m pip install -e ".[test]"
python -c "import cspy; print(cspy.__version__); print(cspy.__file__)"
python -m pytest -q
```

External programs found on Aloe included `gdma`, `mulfit`, `neighcrys`,
`dmacrys`, `pmin`, Gaussian and `formchk`. Platon was not available. During CSP
this required a calculation-local configuration:

```toml
[descriptors]
pxrd = false
```

This only disables PXRD generation during CSP; pymatgen can create descriptors
later during clustering. Without it, the default Platon request caused every
otherwise valid worker result to fail because the executable was `None`.

For Gaussian on Aloe, merely symlinking `g09` was insufficient. Jobs must source
the node-appropriate Gaussian profile and create a temporary `g09` alias:

```bash
if [ -f /etc/sie_ladon ]; then
    NODETYPE=sie_ladon
else
    NODETYPE=sr630
fi

export g16root="/opt/software/g16A-${NODETYPE}"
source "$g16root/g16/bsd/g16.profile"
export GAUSS_SCRDIR="${SLURM_TMPDIR:-/tmp}"

TEMP_BIN=$(mktemp -d)
ln -s "$g16root/g16/g16" "$TEMP_BIN/g09"
export PATH="$TEMP_BIN:$PATH"
```

Remove only that known temporary directory at the end of the job. Keep Slurm
directives minimal, as requested by Lucas.

## Trans-stilbene conformational protocol

The optimized input on Aloe is `TSTILB_opt.xyz`, formula `C14H12`, with 26
atoms. It is not committed to this repository. The verified one-based torsions
for that exact atom order are:

| Coordinate | Definition | Initial value |
| --- | --- | ---: |
| right phenyl--vinyl | `14_1_2_3` | `-179.675 deg` |
| central `C=C` | `2_1_14_11` | `-179.994 deg` |
| left phenyl--vinyl | `1_14_11_10` | `0.174 deg` |

Do not reuse these indices for a differently ordered XYZ without recomputing
the angles.

The production conformational sampling created 256 Sobol samples and completed
Gaussian, GDMA and MULFIT for all of them:

```text
TSTILB_prod_flex.db: 256 distorted_molecules
TSTILB_prod-angles.dat: 256 lines
failed conformers: 0
```

The conformational energy span was `39.0286 kJ/mol` relative to the minimum.
Counts within candidate windows were:

| Window | Conformations retained |
| ---: | ---: |
| 5 kJ/mol | 42 |
| 10 kJ/mol | 99 |
| 15 kJ/mol | 165 |
| 22 kJ/mol | 228 |

The chosen CSP window was `10 kJ/mol`. Selection is uniform among admitted
conformations, not Boltzmann-weighted, so increasing this window has a large
effect. A one-conformer smoke test took about 486 seconds and completed
Gaussian, GDMA and MULFIT successfully.

## `Z'=2`, space-group-14 CSP on Aloe

The same flexible database was passed twice so the two independent molecules
could select conformations independently. The effective production command was:

```bash
mpiexec -n 256 \
    cspy-flex \
    ../flex/TSTILB_prod_flex.db \
    ../flex/TSTILB_prod_flex.db \
    --conf_energy_window 10 \
    -g 14 \
    -n 20000 \
    --adaptcell \
    --nudge 1
```

The job also needed `ulimit -n 8192` because MPICH/Hydra otherwise failed with
`Too many open files` at 256 ranks. `OMP_NUM_THREADS` and `MKL_NUM_THREADS` were
set to 1. No XYZ is needed in the CSP directory because the flexible database
contains geometries, energies, axes, charges and multipoles.

The run was intentionally cancelled with two structures remaining because the
last replacements were taking too long. Preserved result:

```text
valid final structures: 19,998
database: TSTILB_prod_flexx2-14.db
trial_structure rows: 79,992 (19,998 x initial + 3 minimization stages)
energy range: -101.24 to -54.05 kJ/mol
density range: 0.73 to 1.16 g/cm3
descriptors: 0 (PXRD intentionally disabled)
equivalent_to: 0 before clustering
```

The two missing targets are only 0.01% of the requested sample and do not need
to be regenerated. Observed candidate-level PMIN/DMACRYS timeouts, Buckingham
catastrophes, maximum-iteration failures and molecular-count mismatches were
discarded normally. Error classification still labels many failures as
`unknown`; improving that classification is a future maintenance task and does
not invalidate the saved final structures.

The warning claiming `Z'=1` when no XYZ was supplied was benign in this flow:
passing the flexible database twice constructed the two-molecule asymmetric
unit correctly.

## Active completeness analysis on Aloe

This is a last-known snapshot, not a live status. Do not SSH into Aloe. Ask
Lucas to run the checks below.

The original database was copied safely with SQLite backup into:

```text
TSTILB_z2_sg14_work2.db
```

The first `work.db` and `unique.db` were empty because SQLite silently created
an input at a wrong path. `work2.db` was verified as approximately 336 MB with
`79,992` rows and maximum minimization step 3.

The clustering job used one Slurm task with 56 CPUs (no `mpiexec`):

```bash
cspy-db cluster \
    TSTILB_z2_sg14_work2.db \
    -m pymatgen \
    -j "$SLURM_CPUS_PER_TASK" \
    -o TSTILB_z2_sg14_unique2.db \
    --completeness \
    --completeness-output completeness.csv \
    --completeness-convergence-output completeness-convergence.csv
```

At approximately 20:30 on 2026-08-02, `rmsds.dat` had grown to 27,354
comparisons. At 22,619 comparisons, 16 edges below RMSD 0.3 had been found,
forming 14 provisional duplicate families. This ratio is not the completeness;
completeness is calculated from final family multiplicities after clustering.

In `rmsds.dat`, `10000000000.0` is the `BIG_RMS=1e10` sentinel meaning pymatgen
did not find an acceptable overlay. It is not a physical RMSD or an error.

The absolute pair maximum for 19,998 structures is 199,950,003. The code filters
by energy and density before structural matching, so a rough estimate made from
the database spread was 10--15 million comparisons. This estimate was not
measured and should not be treated as a promised final count. At 56 CPUs the
rough runtime expectation was one to three days.

Pymatgen emits many harmless CIF warnings about missing formula keys and small
fractional-coordinate rounding. To prevent a huge log in future runs, the Slurm
script may include:

```bash
export PYTHONWARNINGS="ignore::UserWarning:pymatgen.core.structure"
```

Do not delete `rmsds.dat`; it is restart state. To obtain a fresh status from
Lucas:

```bash
sl
wc -l rmsds.dat
ls -lh completeness.err rmsds.dat
grep -E "Traceback|ERROR" completeness.err | tail
awk '$3 < 0.3 {n++} END {print n+0, "equivalencias encontradas"}' rmsds.dat
```

Expected final products are:

```text
TSTILB_z2_sg14_unique2.db
completeness.csv
completeness-convergence.csv
```

Interpret the result from Good--Turing coverage, its interval, Chao unseen
estimate and convergence shape together. Do not equate a high point estimate
alone with proof of physical completeness.

## Known future work

1. Obtain the final Aloe clustering/completeness outputs and interpret them.
2. Consider reducing repeated pymatgen warning output in code or Slurm scripts.
3. Improve classification of PMIN/DMACRYS failures currently grouped as
   `unknown`.
4. Validate the flexible protocol in additional space groups after the group-14
   analysis.
5. SAUCE may help for `Z' >= 3`, but this version of `cspy-flex` does not
   actually switch to `AUTCSP` when given the related option. Do not assume it
   is active without implementing and testing that integration.
6. Before updating the existing Aloe clone, reconcile its manual edits with
   commit `e02cdb2`; a fresh clone is safer than overwriting a dirty editable
   installation.

## Suggested first prompt on the other computer

```text
Lee AGENTS.md y HANDOFF.md completos. Comprueba git status, la rama y los tres
últimos commits. Ejecuta la suite de pruebas en Python 3.11 y resume el estado
sin conectarte por SSH a Aloe ni Neon. Después pídeme únicamente la salida
actual del clustering de Aloe que necesites para continuar.
```

# Hydro-mechanical models of EDZ formation and EDZ affected transport

## Project Summary

Endorse is a chaotic set of tools for solving real case problems using parametrized models.


## Structure
- `apps` - specific aplications with custom code, some of the code may later be moved into core library
  `apps/chodby_trans` - You are only allowed to modify files under this directory for the near future goals.
                        This project directory has its own environment in `apps/chodby_trans/venv`.
                        There is a working implementation of sensitivity analysis of a 3D transport problem with random fracture network.
                        Current approach uses Saltelli schema through OpenTurns library and simple MC sampling.
                        The goal is to apply MLMC to accelerate Sobol indices estimates.
                        
- `apps/chodby_trans/MLMC` - local only copy of current MLMC source for your reference
- `src/endorse` - core library
- `tests` - currently both unit tests and various numerical tests


## CODEX Ignore Folders

- `**/venv`
- `**/.tox`
- `**/.pytest_cache`
- `**/build`
- `**/dist`
- `**/*.egg-info`

## Workflow (with exception of AGENTS.md and python_coding.md all files in apps/chodby_trans)

- The user reviews changes in `git-cola`. Do not commit changes unless
  explicitly asked.
- Before editing, check the repository state and avoid overwriting unrelated
  user changes.
- At the beginning of work, check the request against `AGENTS.md`, `PLAN.md`,
  `README.md`, and relevant docs/tests.
- Do not ask for confirmation before making requested changes unless the
  required intent cannot be inferred from the repository context.
- Keep each change focused on one function, module concern, or coherent
  refactoring.
- Do not mix planning edits with code implementation unless explicitly
  requested.
- For larger edits, update `PLAN.md` with the intended steps and unresolved
  questions before implementation.
- Put unresolved project questions or inconsistencies in the last section of
  `PLAN.md` under `AGENT Questions And Remarks`.
- Use the `AGENT log` section in `PLAN.md` for concise completed-work records.
- Treat `AGENT` notes in source comments or documentation as direct
  instructions. When resolved, add a short `Resolved:` line after the note and
  let the user remove the note later.
- For documentation-only changes, tests are not required.
- For code changes, run targeted tests first, then broader verification when
  the change affects shared behavior.


## Agents Skills
Agents skills specific to `chodby_trans` application are defined in
`apps/chodby_trans/.agents/skills/inspect-mlmc-run/SKILL.md`.

## Subagents
Subagents specific to `chodby_trans` application are defined in
`apps/chodby_trans/.agents/subagents`.


## Coding Rules

Include and adapt: `python_coding.md`.

Chodby_trans -specific interpretation:
- All MD files have hard limit 120 chars per line.
- The code cannot be tested with a simplified model and the full model takes about half an hour to run. 
  So we need good logging for post moretem inspection.
- The edits should work correctly on a first run, do detailed review of the call path before edits instead of the run.
- Iteratively add type hints and doc strings to touched code.
- While the edits are done on local, I then run test on a parallel cluster.
  So Dask paralelization is essential for sensitivity analysis.
- Use `SA_USAGE.md` for overview of the MLMC library we are going to use.
  For details see installed MLMC in the chodby_trans/venv, or github source, this branch: https://github.com/GeoMop/MLMC/tree/JB_sa_goal
  

## Mandatory Finish Checklist

Before the final response, verify these items explicitly:

- `PLAN.md` has been reviewed for relevant current work.
- Any touched `AGENT` notes have following `Resolved:` lines.
- New unresolved questions or inconsistencies are recorded in `PLAN.md`.
- The final response mentions open `USER:` questions, missed requirements, and
  failed or skipped verification.

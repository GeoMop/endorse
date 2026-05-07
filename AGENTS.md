# Hydro-mechanical models of EDZ formation and EDZ affected transport

## Structure
- `apps` - specific aplications with custom code, some of the code may later be moved into core library
- `src/endorse` - core library
- `tests` - currently both unit tests and various numerical tests

## Coding style and rules
- less defensive in particular various custom code
- Only test the validity of inputs if you can fix if it is invalida and continue with valid structures.
  Otherwise fail early. 
- Do not test for indices and kays let them raise
- try to reduce number of indentions
- try to avoid conditions, use polymorphism, and 'projection' to canonic states and structures
- prefer usage of numpy/xarray vectorised code for numerical code
- adopt 'functional' style prefer pure functions, comprihentions, nonmodified classes
- use Path instead of os.path
- for any new code or touched code, prefer Path consistently; do not introduce new os.path usage unless there is a clear technical reason
- avoid code duplication

## Tests
- maintain tox run without errors (for any commit ready changes)
- use targeted pytest in `tests` to run specific test files

## Changes
- Do not combine more than single targeted (single function or consistent refactoring) change in a single commit.
- Do not ask me for confirmation of any changes, I will check tem in `git-cola`

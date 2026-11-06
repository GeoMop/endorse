---
name: python-implementation-review
description: >-
  Apply Python coding rules when implementing, refactoring, or reviewing Python code.
  Covers common coding practices and distinct expectations for research code and production libraries.
---

# Default rules for Python coding

Apply these rules during Python implementation and review. Research code is single-purpose code used internally;
production code includes libraries and software intended for general use. Review-only requests do not authorize edits.

## Common rules

### Common sense rules

- Never resolve test errors by adding try/except blocks.
- Use only obvious defaults; never guess default values.
- Apply strong defensive validation to user input, such as configuration files; reject invalid input early.
- Check required dictionary keys early only when needed later in a long calculation. Otherwise let `KeyError` propagate.
- Never substitute another configuration key after a `KeyError`; fail early instead.
- Use assertions for consistency checks.
- Never add runtime fallbacks or import shims for a broken or incomplete environment. Report missing dependencies or
  tools and fix the environment or tests, without code workarounds.
- Optional dependencies may be imported lazily in a clear helper at the feature boundary. On failure, name the missing
  dependency and the feature requiring it.
- Never call anything "self-explanatory" in documentation comments; explain its meaning with context.

### Best code, is no code!

- Suggest refactoring that reduces code size or complexity, even when it would break backward compatibility.
- Avoid duplicated code.
- Prefer high-level NumPy, pandas, or xarray operations over loops and native lists or dictionaries 
  for speed and brevity.
- Avoid branching, especially nested branching; use polymorphism and duck typing where appropriate.
- Avoid simple, one-use module or class functions. Prefer a local `def` or `lambda` for a small helper used in one
  method or expression. Use a private helper when reused or when separating it clearly simplifies the surrounding code.
- Do not let tests constrain early implementation design; consider radical refactoring and rewriting tests.

### Antipatterns

- Unpack tuples into named local variables instead of indexing them. 
  Prefer a data class for tuples longer than three items.

### Functional style

- Prefer pure functions.
- Prefer objects that remain unchanged after construction; methods calculate using stored data and passed arguments.

### Documentation

- Write docstrings for new public classes, functions, and methods.
- Document the goal of each pytest test function.
- Preserve existing comments, notes, and commented-out code unless removing the whole function or file, 
  or doing cleanup.
- Outside those exceptions, suggest comment removal and ask for confirmation. 
  Improve existing comments and documentation.

### Preferred libraries

- Use `logging`, including for debug output.
- Use `pathlib` for paths.
- Use `attrs` for data classes.
- Use static or class factory methods with `attrs` to construct objects from data different from their stored fields.

## Research project rules

- Favor less defensive code, no backward-compatibility requirement, and a simple input-to-output structure.

## Production code rules

- Allow greater complexity, use more defensive code, and preserve backward compatibility.

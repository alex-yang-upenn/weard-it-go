## Coding Guidelines

- Simplicity is king. Don't write code for niche edge cases, don't overgeneralize, and don't add things just to look "professional".
- Pick an assumption and go with it. Don't add options, flags, modes, or auto-detection when one good default will do.
- No bells and whistles. Keep functionality minimal. Don't add retries, fallbacks, health checks, or metrics unless something is actually broken without them.
- Close enough is good enough. If the exact feature is hard to build, build something nearby that's easy and say so.
- Modularity always. Use small, single-purpose Python files. Use classes where sensible, constructed in `__init__`/`main`. No module-level mutable globals.
- Avoid comments and docstrings. Let names carry the meaning. Comment only what a good name can't capture.
- Use long, descriptive names for variables, functions, classes, and files: `uniform_frame_sampler.py`, not `sampler.py`.
- Don't force sharing. Put code in `common/` only when two services really use it.

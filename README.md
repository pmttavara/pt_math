# pt_math.h
This is a set of **branch-free, table-free** implementations of the math functions in `<math.h>`.

All of the code is public domain - if you like, you can copy out the `sin` implementation if that's all you need.

These functions differ slightly from the C standard library *by design* - **to stay fast, they don't do range checking or follow C's precision requirements.** Think of these routines more as "decent approximations", for when you know the particular input domain and you have leeway with the output. Video games are a good use case.

### C compatibility
These functions stick to the C API, but the library doesn't implement all of `math.h` - notable exclusions are machine-related functions (e.g. `fma`, `nextafter`), precision-related functions (e.g. `expm1`), complex numbers, and `tgamma`/`lgamma`. The standard elementary functions that you would expect are included, as well as hyperbolic trig.

### Exceptions to branch-free rule
- `atan2` is highly discontinuous, so a truly branch-free implementation of this routine is likely impossible.

## Credits
I want this library to be easy to use, so no attribution is necessary.

I didn't invent all of these algorithms, but the original authors are lost to time.

- The `sin` algorithm was originally conceived by an anonymous user on a now-unavailable programming forum.
- The `log` snippet is of unknown origin, similarly to the fast inverse square root.

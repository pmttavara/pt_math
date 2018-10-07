# pt_math.h
This is a set of **branch-free, table-free** implementations of the math functions in `<math.h>`.

All of the code is "public domain"<sup>1</sup> - if you like, you can copy out the `sin` implementation if that's all you need.

These functions differ slightly from the C standard library *by design* - **to stay fast, they don't do range checking by default, or follow C's precision requirements.** Think of these routines more as "decent approximations", for when you know the particular input domain and you have leeway with the output. Video games are a good use case.

### C compatibility
These functions stick to the C API, but the library doesn't implement all of `math.h` - notable exclusions are machine-related functions (e.g. `fma`, `nextafter`), precision-related functions (e.g. `expm1`), complex numbers, and `tgamma`/`lgamma`. The standard elementary functions that you would expect are included, as well as hyperbolic trig.

### Functionality
The library uses a select few functions, fully implemented from scratch, in order to derive the implementation of others via identities. For example, `cos(x)` is implemented as `sin(x + π/2)`, and `tan(x)` is implemented as `sin(x)/cos(x)`. This simplifies many of the routines, since the potential precision loss from this approach is negligible within the given bounds of the library.

### Exceptions to branch-free rule
- `atan2` is highly discontinuous, so a truly branch-free implementation of this routine is likely impossible.

## License & Credits
I want this library to be easy to use, so no attribution is necessary.

<sup>1</sup>Actual public domain dedications are of unclear legality worldwide, so an equivalent permissive license (no-clause BSD) is used, available at the bottom of the file.

Algorithm credit:
- The `sin` algorithm was originally conceived by an anonymous user on (unfortunately) a now-unavailable programming forum.
- `round`, `sqrt`, `rsqrt`, and `log` are all of dubious origin, scattered publicly around the Internet.
- The rest are new, to my knowledge.

# pt_math.h
This is a set of **branch-free, table-free** implementations of the math functions in `<math.h>`.

All of the code is public domain - if you like, you can simply copy-paste the `sin` implementation if that's all you need.

These functions differ slightly from the C standard library *by design* - **to stay fast, they don't do range checking or follow C's precision requirements.** Think of these routines more as "decent approximations", for when you know the particular input domain and you have leeway with the output. Video games are a good example of an appropriate use case.

# API
These functions stick to the C API, but it doesn't implement all of `math.h` - notable exclusions are machine-related functions (e.g. `fma`, `nextafter`), precision-related functions (e.g. `expm1`), complex numbers, and `tgamma`/`lgamma`. But you've got your standard elementary functions and hyperbolic trig as well.

# Exceptions to branch-free rule
`atan2` likely compiles to branches; it's hopelessly discontinuous.
`dtoa` is a crappy implementation, included for completion.

# Credits
**There are no credits.** I want this library to be easy to use - which means public domain, and no attribution necessary.
To be clear, I didn't invent all of these algorithms - but conveniently, the original authors are lost to time.
The `sin` algorithm was originally conceived by an anonymous user on a now-unavailable programming forum, and the `log` snippet is of unknown origin similarly to the fast inverse square root.

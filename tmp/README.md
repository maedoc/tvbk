# fast kionex implementations

this is a set of implementations towards faster
sweeps for kionex,

- v0 is a reference impl w/ TVB
- v1 & v2 are simple C impls (but have bugs)
- v3 is a single thread, single sim ISPC impl
- v8 is a single thread, 8x sim ISPC impl
- v9 uses v8 with thread pool, for N cores x 8 SIMD
- v9sbi is a simple SBI workflow using v9 results
- v10 is an OpenCL impl drafted w/ Gemini, not totally verified

## usage

since testing against reference was important here, every version
uses the setup defined in v0.  if you change parameters etc, do it
in v0.  if you want to test correctness, run v0 then v3 then v8. if
it passes, then it should be ok.



# tvb_kernels

This is a library of computational kernels for TVB.

## scope

in order of priority

- [x] sparse delay coupling functions
- [ ] fused heun neural mass model step functions
- [x] random number generation
- [ ] neural ODE
- [ ] bold / tavg monitors
- [x] small matmul suitable for neural ODE

which will then be used internally by TVB.

## building

No stable versions are available, so users can install from
source with `pip install .` or to just get a wheel `pip wheel .`

For dev w/ editor using clangd, do
```
bear -- make -B test
```
 
## roadmap

- improve api (ownership, checking etc)
- variants
  - single
  - batches
  - spatialized parameters
- cuda ports for Jax, CuPy and Torch users 
- mex functions?
- scalable components
  - domains
  - projections
  - small vm to automate inner loop

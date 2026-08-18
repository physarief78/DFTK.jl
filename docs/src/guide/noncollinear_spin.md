# Noncollinear spin (experimental)

DFTK supports two-component Pauli spinors with `spin_polarization=:full`. The feature is
experimental and currently intended for ground-state LDA calculations without spin-orbit
coupling. See the executable [experimental noncollinear example](@ref experimental-noncollinear)
for a complete calculation.

## Representations

The three spin-related dimensions have distinct meanings:

| Quantity | Scalar / collinear modes | `spin_polarization=:full` |
|:--|:--|:--|
| k-point spin channels | 1 or 2 | 1 |
| wavefunction components | 1 | 2 |
| real density fields | 1 or 2 | 4: `(n, mx, my, mz)` |

A full-spin orbital at one k-point has shape `(2, n_G, n_bands)`. Use
[`component_view`](@ref), [`each_component`](@ref), or [`spinors`](@ref) to inspect its
spin components. [`as_matrix`](@ref) provides the zero-copy
`(2n_G, n_bands)` view used by dense linear algebra.

The full density is represented by real Pauli fields,

```math
\rho_\mathrm{mat}(\mathbf r) = \frac{1}{2}
\left[n(\mathbf r) I + \mathbf m(\mathbf r)\cdot\boldsymbol\sigma\right].
```

Use [`total_density`](@ref) for ``n``, [`magnetization`](@ref) for the three components of
``\mathbf m``, and [`density_matrices`](@ref) for a lazy pointwise matrix view. A local
matrix potential uses dual fields `(v, bx, by, bz)` and acts as
``v I + \mathbf b\cdot\boldsymbol\sigma``.

## External magnetic fields

[`MagneticField`](@ref) accepts a constant three-vector, a function of Cartesian
position, or an `(fft_size..., 3)` array. Its convention is

```math
H_Z = -\mathbf b(\mathbf r)\cdot\boldsymbol\sigma,
\qquad E_Z = -\int \mathbf b(\mathbf r)\cdot\mathbf m(\mathbf r)\,d\mathbf r.
```

## Current scope

Supported functionality includes kinetic, scalar local, nonlocal pseudopotential, Hartree,
locally collinear LDA exchange-correlation, external Zeeman fields, eigensolvers, SCF,
Kerker mixing, band calculations, density/orbital serialization, and spin projections.

The following deliberately raise an explicit error for full-spin models: GGA and meta-GGA
functionals, forces, stresses, phonons and response, exact exchange, Hubbard, direct
minimization, Newton SCF, current density, and vector-potential magnetic terms. Magnetic
space-group symmetry and spin-orbit coupling are not yet implemented; full-spin models use
spatial identity symmetry.

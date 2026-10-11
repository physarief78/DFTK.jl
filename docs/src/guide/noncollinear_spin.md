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

The term also supports `spin_polarization=:collinear`, where the field must point
along z. For example, `MagneticField([0.0, 0.0, b])` adds ``-b`` to the spin-up
potential and ``+b`` to the spin-down potential. Any nonzero transverse component
raises an `ArgumentError`; use `:full` for transverse fields. Uniform and spatially
varying z fields can be combined with the ordinary collinear functionals, including
PBE and Hubbard U.

The field parameter is an energy in Hartree, with ``b = \mu_B B`` for g = 2.
A field of 1 T corresponds to approximately 2.1272e-6 Ha; positive b favours
positive z magnetization. This term describes the spin Zeeman response without
orbital magnetism or spin-orbit coupling.

Without spin-orbit coupling, uniform fields supplied as a tuple or vector preserve
the model's spatial symmetries. Function and array fields conservatively disable
symmetry reduction. Full-spin models continue to use spatial identity symmetry.

Collinear response calculations and ForwardDiff through the SCF are available
without Hubbard U. For a field derivative, construct a basis with the dual scalar
type, for example by converting the lattice to `typeof(b)` in the differentiated
function. The field builder preserves dual values and strips them when rebuilding
the primal basis. With Hubbard U, use finite differences: a Hubbard response kernel
is not implemented.

## Current scope

Supported functionality includes kinetic, scalar local, nonlocal pseudopotential, Hartree,
locally collinear LDA exchange-correlation, external Zeeman fields, eigensolvers, SCF,
Kerker mixing, band calculations, density/orbital serialization, and spin projections.

The following deliberately raise an explicit error for full-spin models: GGA and meta-GGA
functionals, forces, stresses, phonons and response, exact exchange, Hubbard, direct
minimization, Newton SCF, current density, and vector-potential magnetic terms. Magnetic
space-group symmetry and spin-orbit coupling are not yet implemented; full-spin models use
spatial identity symmetry.

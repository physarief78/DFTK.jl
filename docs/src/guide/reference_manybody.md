# Dense response and excitons

These CPU reference routines implement literal band and reciprocal-space sums.
They are for small validation systems and explicit convergence studies. They
require a full k mesh, scalar orbitals and one MPI rank. Frequencies are Ha;
q vectors are reduced and must map the k mesh onto itself.

```julia
q = Vec3(0.25,0,0)  # on a 4×4×4 mesh
derivative = compute_occupation_derivative(bands.basis,bands.eigenvalues,bands.εF)
response = compute_rpa(bands,q,[0.0,0.1,0.2]; η=0.005,Ecut_response=1.0,
                       occupation_derivative=derivative)
loss = compute_eels(response)
```

`compute_χ0_dynamic` forms dimensionless pair densities by multiplying the FFT
of `conj(u_nk)*u_m,k+q` by `sqrt(Ω)`. Reciprocal relabelling at the Brillouin-zone
boundary follows exact exchange. The response is `1/Ω` times the Adler–Wiser
sum; spin degeneracy is already in the occupations. Explicit energy derivatives
are necessary in metallic static degenerate limits. At finite q, both intra-
and interband free-carrier screening follow the supplied occupations.

An optional `optical_direction` at q=0 analytically scales the head and wings by
q² and q, using the optical velocities. This is an **interband** optical limit;
use finite q for photocarrier plasmons and screening, or `compute_dielectric`
with an explicit Drude derivative for optical metallic response. The default
q=0 kernel omits its singular mesh point. That convention is exposed in the
returned `kernel`; it is not a converged singularity integral.

For BSE, supply a callable `screening(q)` returning static RPA at ω=η=0. It must
use a justified q=0 singularity treatment (for example the existing `VoxelAverage`
extension with `FastGaussQuadrature` for bulk), a converged screening cutoff and enough
empty bands. Use `SlabTruncatedCoulomb` for isolated-sheet screening, with its
normal and half-height cutoff set to the actual supercell. With
`FastGaussQuadrature`, `VoxelAverage(SlabTruncatedCoulomb(basis))` averages over
the two-dimensional reciprocal cell, subtracting the integrable `4πR/|q|`
singularity analytically. This is tested against an independent Duffy integral.
Converge its quadrature, q mesh and q-dependent screening before claiming slab
binding energies; a correct bare-kernel integral alone does not validate W or BSE.

`compute_bse` builds a Hermitian **singlet Tamm–Dancoff** matrix and diagonalizes
it densely. Positive occupation differences enter kernels and oscillator strengths
through square-root factors. Population inversion is rejected. This is a static
approximation to a constrained ensemble, not a full nonequilibrium BSE. A scissor
changes transition energies; rescaling velocities leaves dipoles unchanged.
`compute_bse_dielectric` includes both causal poles in the spectrum, without
claiming to include resonant/antiresonant coupling in the eigenproblem.

`compute_exciton_densities` returns electron and hole marginals. The conditional
fragment descriptor `compute_exciton_character` integrates a joint projector
with explicit masks; it does not multiply marginal populations. With partial
occupations it uses normalized population-weighted amplitudes of the static
response model. `wannier_mott` is only an isotropic 3D hydrogenic estimate.

The static response is checked against both complete-basis `compute_χ0` and
finite-q Sternheimer/`DielectricAdjoint`. The optical head agrees with the IPA
Kubo formula. BSE tests include the noninteracting limit, partial populations,
finite-q reciprocal relabelling and Bloch-phase covariance. These internal
checks do not substitute for converged Si, LiF, anatase, hBN and graphene
literature benchmarks. No such material spectrum is declared validated here.

Theory references: [Ismail-Beigi (slab truncation)](https://arxiv.org/abs/cond-mat/0603448),
[exciting BSE tutorial](https://exciting-code.org/uploads/exciting/tutorial_notebooks/excited_states_from_bse.html),
and [GPAW BSE theory](https://gpaw.readthedocs.io/documentation/bse/bse.html).

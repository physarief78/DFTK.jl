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

For BSE, supply a callable `screening(q)` returning static W at ω=η=0. At q=0,
averaging only the bare Coulomb head leaves bulk interband screening out of that
cell. `compute_rpa_screening_voxel` instead builds the long-wavelength head and
wings from velocities and averages **W**. It includes microscopic local fields
and an explicitly supplied carrier compressibility. For bulk insulators the
anisotropic `4π/(qᵀεq)` head is integrated by a surface reduction. For sheets the
analytic `4πR/|q|` singularity is subtracted before quadrature. Use the raw
`BareCoulomb()` or `SlabTruncatedCoulomb(basis)` kernel for this routine; it requires
`FastGaussQuadrature` and an even quadrature order.

```julia
kernel = BareCoulomb()  # or SlabTruncatedCoulomb(bands.basis)
screening(q) = iszero(q) ?
    compute_rpa_screening_voxel(bands; Ecut_response=1, kernel,
                                occupation_derivative=derivative) :
    compute_rpa(bands,q,[0.0]; η=0,Ecut_response=1,
                kernel=VoxelAverage(kernel),occupation_derivative=derivative)
```

This is a long-wavelength model inside the central cell. Converge its quadrature,
q mesh, empty bands and cutoff, especially relative to photocarrier Fermi
wavevectors. It does not reproduce full finite-q screening across a coarse cell.
The returned `dielectric_tensor` is defined only for insulating bulk. The cell
result supplies unscaled W for BSE and is not a pointwise EELS response.

The independent bare slab average `VoxelAverage(SlabTruncatedCoulomb(basis))`
uses the two-dimensional reciprocal cell and is checked against a Duffy integral.
Its normal and half-height cutoff must match the supercell. A correct kernel
integral alone does not establish a converged slab binding energy. For the role
of the small-q screened interaction see
[Rasmussen et al.](https://arxiv.org/abs/1511.00129).

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

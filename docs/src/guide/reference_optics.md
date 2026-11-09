# Reference optical response

The CPU reference routines use atomic units, full k grids (`symmetries=false`),
and converged empty bands. They support scalar and collinear Kohn–Sham models with
kinetic, local, norm-conserving nonlocal and Hubbard potentials. Exact exchange,
meta-GGA velocity terms, spinor orbitals and symmetry reduction are rejected.

```julia
bands = compute_bands(scfres, MonkhorstPack([8,8,8]);
                      n_bands=20, tol=1e-9, seed=UInt64(42))
velocity = compute_velocity(bands)
spectrum = compute_dielectric(bands, collect(0.0:0.001:0.4);
                              velocity, η=0.005, scissor=0.05, n_valence=4)
α = absorption_coefficient.(spectrum.ω, spectrum.ε[1,1,:])
```

The velocity is the ForwardDiff derivative of the existing kinetic and projector
formulas at fixed G set, density and Hubbard occupation matrices. The atomic
orbitals are Löwdin orthogonalized before selecting a Hubbard manifold, so the
derivative includes changes in the complete overlap matrix. A scissor rescales
intermanifold matrix elements by the ratio of shifted to unshifted transition
energies. It does not change the orbitals or occupations.

The dielectric function is a causal complex interband sum. Occupations include
DFTK's spin degeneracy. A Drude term is included **only** when an explicit
`occupation_derivative` array is supplied. Arbitrary occupations alone do not
specify a Fermi-surface derivative. Frequencies at zero are rejected when the
Drude weight is nonzero. Negative absorption is retained for population inversion.

`fit_tauc` requires an explicit fit window and exponent. Vertical independent-
particle spectra contain no phonon-assisted indirect absorption; an indirect
experimental Tauc gap is not a direct scissor calibration target. Slab dielectric
functions include the supercell vacuum and must be converted to a sheet response
or compared at a common cell height before drawing intensity conclusions.

These routines do not establish carrier lifetimes, PL yields or an illuminated
steady-state population. Converge Ecut, k grid, empty bands and broadening, and
benchmark the resulting spectra before using them in a scientific claim.

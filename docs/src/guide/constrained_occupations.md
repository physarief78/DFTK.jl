# Static light-state models

`QuasiFermi(Nv, n_eh)` fixes a valence and conduction population, with separate
chemical potentials determined by bisection. `n_eh` is electrons per cell, not
a photon flux or a predicted steady-state density. A positive **indirect** gap
must separate the two manifolds at every SCF step. The returned `scfres.εF`
is a `QuasiFermiLevels` holding `μ_h`, `μ_e` and `n_valence`.

```julia
light = self_consistent_field(basis;
    fermialg=QuasiFermi(24,0.1),
    nbandsalg=FixedBands(; n_bands_converge=40, n_bands_compute=40),
    mixing=KerkerMixing(), seed=UInt64(42))
```

The entropy uses the chemical potential of each manifold. At zero pairs the
manifolds are exactly full/empty. This equals ordinary insulating occupations
at zero temperature. At finite temperature a common Fermi level permits
intrinsic pairs, so exact equality is generally incompatible with the stated
zero-conduction-population constraint. Nonzero fractional pairs need positive
temperature. `compute_bands(light, ...)` re-solves the population constraints
on the new grid. `compute_occupation_derivative` holds both chemical potentials
fixed and supplies the derivative for optical Drude weights.

`FixedOccupations([f_k, ...])` accepts explicit occupations for ΔSCF. Request
exactly the same number of computed bands with `FixedBands`. At finite
temperature only Fermi–Dirac entropy is currently implemented for this mode.
Following band indices does not protect state character across crossings;
inspect the state during relaxation before interpreting a CT or trapped exciton.
Occupation vectors are local to an MPI rank. Changing the k mesh is rejected.

Both constrained modes support fixed-cell Hellmann–Feynman forces. Equilibrium
LDOS/χ0 mixing, stresses and equilibrium DFPT/phonons are rejected. The separate
sum-over-states dynamical response uses the explicitly supplied occupations;
`compute_χ0(ham)` by itself instead constructs an equilibrium reference.

For L2, choose `n_electrons=Nneutral+q`, positive temperature, and optionally add
`SheetCounterCharge(q; center=0.85, width=1.0)` to `extra_terms`. The fractional-
electron model also requires `disable_electrostatics_check=true`. Its electron
potential is negative for positive sheet charge. Its three energy pieces replace
the uniform background in the same zero-average electrostatic gauge. The sheet
is classical and fixed; no carrier exchange or voltage boundary condition is
implied. Keep it in the vacuum and converge its width and separation.

JLD2 stores the occupation algorithm and quasi-Fermi state. When restarting a
constrained calculation, explicitly pass the loaded `fermialg` and a compatible
`FixedBands` and mixing choice. A restart without them is an ordinary equilibrium
SCF and can erase the excitation.

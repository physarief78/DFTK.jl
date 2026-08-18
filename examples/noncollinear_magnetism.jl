# # [Experimental noncollinear spin](@id experimental-noncollinear)
#
# This example demonstrates the full-spin representation on a one-electron periodic cell.
# A transverse Zeeman field polarizes the electron along ``x``, so both spinor components
# are needed even though the model has only one k-point spin channel.
#
# !!! warning "Experimental feature"
#     `spin_polarization=:full` currently supports ground-state SCF with locally collinear
#     LDA exchange-correlation. Forces, stresses, response, GGA/meta-GGA, spin-orbit
#     coupling, and magnetic space-group symmetry are not yet implemented.

using DFTK
using LinearAlgebra

lattice = 5.0 .* Matrix{Float64}(I, 3, 3)
field = [0.05, 0.0, 0.0]

model = Model(
    lattice;
    n_electrons=1,
    terms=[Kinetic(), Hartree(), Xc([:lda_x]), MagneticField(field)],
    spin_polarization=:full,
    symmetries=false,
)
basis = PlaneWaveBasis(model; Ecut=1, kgrid=(1, 1, 1), fft_size=(9, 9, 9));

# A full-spin density stores `(n, mx, my, mz)`. We start from the exact uniform
# ``+x``-polarized density for this simple model.
n0 = fill(model.n_electrons / model.unit_cell_volume, basis.fft_size)
m0 = cat(n0, zero(n0), zero(n0); dims=Val(4))
ρ0 = ρ_from_total_and_magnetization(n0, m0);

scfres = self_consistent_field(
    basis;
    ρ=ρ0,
    tol=1e-9,
    maxiter=4,
    mixing=SimpleMixing(),
    solver=ScfDampingSolver(),
    eigensolver=diag_full,
    nbandsalg=FixedBands(; n_bands_converge=1, n_bands_compute=4),
);

# The density has four real fields, while each k-point orbital has the canonical
# `(component, G, band)` shape.
@show size(scfres.ρ)
@show size(first(scfres.ψ));

# Integrating the magnetization recovers one electron polarized along ``x``.
m = magnetization(basis, scfres.ρ)
integrated_magnetization = [sum(m[:, :, :, α]) * basis.dvol for α in 1:3]
@show integrated_magnetization;

# `spin_projections` gives ``(⟨σx⟩, ⟨σy⟩, ⟨σz⟩)`` for each band.
spin_projections(first(scfres.ψ))


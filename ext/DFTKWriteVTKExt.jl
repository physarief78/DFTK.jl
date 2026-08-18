module DFTKWriteVTKExt
using WriteVTK
using DFTK
using DFTK: blockify_ψ, gather_kpts, gather_kpts_block
using Printf
using MPI

function DFTK.save_scfres(::Val{:vts}, filename::AbstractString, scfres::NamedTuple;
                          save_ψ=false, extra_data=Dict{String,Any}(), kwargs...)
    # Initialise the grid and open the file on master
    vtkfile = nothing
    if mpi_master(scfres.basis.comm_kpts)
        grid = zeros(3, scfres.basis.fft_size...)
        for (idcs, r) in zip(CartesianIndices(scfres.basis.fft_size),
                             r_vectors_cart(scfres.basis))
            grid[:, Tuple(idcs)...] = r
        end
        vtkfile = WriteVTK.vtk_grid(filename, grid)
    end

    if save_ψ
        basis = gather_kpts(scfres.basis)
        ψblock_dist = blockify_ψ(scfres.basis, scfres.ψ).ψ
        ψblock = gather_kpts_block(scfres.basis, ψblock_dist)

        if mpi_master(scfres.basis.comm_kpts)
            C = n_components(basis.model)
            n_bands = C == 1 ? size(ψblock, 2) : size(ψblock, 3)
            for ik = 1:length(basis.kpoints), n = 1:n_bands
                kpt_n_G = length(G_vectors(basis, basis.kpoints[ik]))
                if C == 1
                    ψnk_real = ifft(basis, basis.kpoints[ik],
                                    ψblock[1:kpt_n_G, n, ik])
                    prefix = @sprintf "ψ_k%03i_n%03i" ik n
                    vtkfile["$(prefix)_real"] = real.(ψnk_real)
                    vtkfile["$(prefix)_imag"] = imag.(ψnk_real)
                else
                    for σ = 1:C
                        ψnk_real = ifft(basis, basis.kpoints[ik],
                                        ψblock[σ, 1:kpt_n_G, n, ik])
                        prefix = @sprintf "ψ_k%03i_n%03i_c%02i" ik n σ
                        vtkfile["$(prefix)_real"] = real.(ψnk_real)
                        vtkfile["$(prefix)_imag"] = imag.(ψnk_real)
                    end
                end
            end
        end
    end

    if mpi_master(scfres.basis.comm_kpts)
        vtkfile["ρtotal"] = total_density(scfres.ρ)
        if scfres.basis.model.spin_polarization == :full
            m = magnetization(scfres.basis, scfres.ρ)
            for (α, label) in enumerate(("x", "y", "z"))
                vtkfile["m$label"] = m[:, :, :, α]
            end
        else
            ρspin = spin_density(scfres.ρ)
            if !isnothing(ρspin)
                vtkfile["ρspin"] = ρspin
            end
        end
        for (key, value) in pairs(extra_data)
            vtkfile[key] = value
        end
        WriteVTK.vtk_save(vtkfile)
    end

    DFTK.mpi_barrier(scfres.basis.comm_kpts)
    nothing
end

end

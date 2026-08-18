@doc raw"""
Function for diagonalising each ``k``-Point blow of ham one step at a time.
Some logic for interpolating between ``k``-points is used if `interpolate_kpoints`
is true and if no guesses are given. `eigensolver` is the iterative eigensolver
that really does the work, operating on a single ``k``-Block.
`eigensolver` should support the API `eigensolver(A, X0; prec, tol, maxiter)`
`prec_type` should be a function that returns a preconditioner when called as `prec(ham, kpt)`
"""
function diagonalize_all_kblocks(eigensolver, ham::Hamiltonian, nev_per_kpoint::Int;
                                 ψguess=nothing,
                                 prec_type=PreconditionerTPA, interpolate_kpoints=true,
                                 tol=1e-6, miniter=1, maxiter=100, n_conv_check=nothing)
    kpoints = ham.basis.kpoints
    results = Vector{Any}(undef, length(kpoints))
    C = n_components(ham.basis.model)
    component_count = Val(C)

    for (ik, kpt) in enumerate(kpoints)
        n_Gk = length(G_vectors(ham.basis, kpt))
        block_size = C * n_Gk
        if block_size < nev_per_kpoint
            error("The size of the Hamiltonian block is $block_size, and you are asking for " *
                  "$nev_per_kpoint eigenvalues. Increase Ecut.")
        end
        # Get ψguessk
        if !isnothing(ψguess)
            ψguessk_matrix = as_matrix(ψguess[ik])
            if block_size != size(ψguessk_matrix, 1)
                error("Mismatch in dimension between guess ($(size(ψguessk_matrix, 1))) " *
                      "and Hamiltonian ($block_size)")
            end
            nev_guess = size(ψguessk_matrix, 2)
            if nev_guess > nev_per_kpoint
                ψguessk = ψguessk_matrix[:, 1:nev_per_kpoint]
            elseif nev_guess == nev_per_kpoint
                ψguessk = ψguessk_matrix
            else
                X0 = similar(ψguessk_matrix, block_size, nev_per_kpoint)
                X0[:, 1:nev_guess] = ψguessk_matrix
                X0[:, nev_guess+1:end] = randn(eltype(X0), block_size,
                                                nev_per_kpoint - nev_guess)
                ψguessk = ortho_qr(X0)
            end
        elseif interpolate_kpoints && ik > 1
            # use information from previous k-point
            ψguessk = interpolate_kpoint(results[ik - 1].X, ham.basis, kpoints[ik - 1],
                                         ham.basis, kpoints[ik], component_count)
        else
            ψguessk = as_matrix(random_orbitals(ham.basis, kpt, nev_per_kpoint))
        end
        @assert size(ψguessk) == (block_size, nev_per_kpoint)

        prec = nothing
        !isnothing(prec_type) && (prec = prec_type(ham[ik]))
        results[ik] = eigensolver(ham[ik], ψguessk;
                                  prec, tol, miniter, maxiter, n_conv_check)
    end

    # Transform results into a nicer datastructure
    # TODO It feels inconsistent to put λ onto the CPU here but none of the other objects.
    #      Better have this handled by the caller of diagonalize_all_kblocks.
    #      Note further that lobpcg_hyper by default puts the eigenvalues
    #      on the CPU ... even if the next line is removed.
    (; λ=[to_cpu(real.(res.λ)) for res in results],  # Always get onto the CPU
     X=[_canonical_orbitals(res.X, component_count,
                            length(G_vectors(ham.basis, kpt)))
        for (res, kpt) in zip(results, kpoints)],
     residual_norms=[res.residual_norms for res in results],
     n_iter=[res.n_iter for res in results],
     converged=all(res.converged for res in results),
     n_matvec=sum(res.n_matvec for res in results))
end

@doc raw"""
Function to select a subset of eigenpairs on each ``k``-Point. Works on the
Tuple returned by `diagonalize_all_kblocks`.
"""
function select_eigenpairs_all_kblocks(eigres, range)
    merge(eigres, (; λ=[λk[range] for λk in eigres.λ],
                   # Copy, such that the selection does not alias (and keep alive) the
                   # full diagonalisation blocks and stays a plain array.
                   X=[copy(_select_bands(Xk, range)) for Xk in eigres.X],
                   residual_norms=[resk[range] for resk in eigres.residual_norms]))
end

# The actual implementations using the above primitives. `lobpcg_hyper` is a thin
# wrapper around the LOBPCG solver from the standalone LOBPCGEigensolver.jl package
# (imported at the top of DFTK.jl); `diag_full` is a dense reference solver.
include("diag_full.jl")
include("diag_lobpcg_hyper.jl")

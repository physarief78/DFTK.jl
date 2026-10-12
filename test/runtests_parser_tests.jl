@testitem "Test argument parsing" tags=[:minimal, :dont_test_mpi] begin
    include(joinpath(@__DIR__, "runtests_parser.jl"))

    withenv("DFTK_TEST_ARGS" => nothing) do
        selected = parse_test_args(["noncollinear"])
        @test selected.base_tag == :all
        @test selected.included == [:noncollinear]
        @test :ncollinear ∉ selected.excluded
        @test :all in selected.excluded
        @test :gpu in selected.excluded

        excluded = parse_test_args(["nononcollinear"])
        @test isempty(excluded.included)
        @test :noncollinear in excluded.excluded

        ordinary = parse_test_args(["minimal", "noslow"])
        @test ordinary.included == [:minimal]
        @test :slow in ordinary.excluded

        combined = parse_test_args(["minimal", "noncollinear"])
        @test combined.included == [:minimal, :noncollinear]

        mpi = parse_test_args(["mpi", "noncollinear"])
        @test mpi.base_tag == :mpi
        @test mpi.included == [:noncollinear]
        @test :gpu in mpi.excluded
        @test :dont_test_mpi in mpi.excluded

        gpu = parse_test_args(["gpu", "noncollinear"])
        @test gpu.base_tag == :gpu
        @test gpu.included == [:noncollinear]
    end

    withenv("DFTK_TEST_ARGS" => "noncollinear-noslow") do
        selected = parse_test_args(String[])
        @test selected.included == [:noncollinear]
        @test :slow in selected.excluded

        explicit = parse_test_args(["minimal"])
        @test explicit.included == [:minimal]
        @test :slow ∉ explicit.excluded
        @test :noncollinear ∉ explicit.excluded
    end
end

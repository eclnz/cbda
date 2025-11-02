include("tfce2.jl")
using InteractiveUtils

# Use a small UnionFind for sanity checks as before
uf_small = UnionFind(10)
root_contrib_small = zeros(Float64, 10)
active_root_flag_small = falses(10)

# Use large 100x100x100 dataset for "big" tests
nx = 100
ny = 100
nz = 100
nvox = nx * ny * nz

active = falses(nvox)
neighbor_table = precompute_neighbors6(nx, ny, nz)
uf_large = UnionFind(nvox)
root_contrib_large = zeros(Float64, nvox)
active_root_flag_large = falses(nvox)
idx = div(nvox, 2)  # somewhere near the middle

tfce_out = zeros(Float64, nx, ny, nz)
t_stat = zeros(Float64, nx, ny, nz)

# Small tests
find!(uf_small, 1)
union_with_contrib!(uf_small, 1, 2, root_contrib_small, active_root_flag_small)
get_size(uf_small, 1)
neighbors6(63, 5, 5, 5)
precompute_neighbors6(5, 5, 5)

# Large dataset tests
merge_active_neighbors!(uf_large, active, idx, neighbor_table, root_contrib_large, active_root_flag_large)
tfce_3d_unionfind!(tfce_out, t_stat; E=0.5, H=2.0)

@time UnionFind(10)
@time find!(uf_small, 1)
@time union_with_contrib!(uf_small, 1, 2, root_contrib_small, active_root_flag_small)
@time get_size(uf_small, 1)
@time neighbors6(63, 5, 5, 5)
@time precompute_neighbors6(5, 5, 5)
@time merge_active_neighbors!(uf_large, active, idx, neighbor_table, root_contrib_large, active_root_flag_large)
@time tfce_3d_unionfind!(tfce_out, t_stat; E=0.5, H=2.0)

nothing
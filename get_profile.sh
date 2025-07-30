echo "##################################"
echo "PROFILE GEMM WITH PREFETCH"

ncu --set full -o gemm_profile uv run dense_gemm.py                    \
  --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
  --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
  --c_dtype Float16 --acc_dtype Float32                                \
  --a_major k --b_major k --c_major n                                  \


echo "##################################"
echo "PROFILE GEMM WITH COMPILER OPT"

ncu --set full -o gemm_profile_compiler_opt uv run dense_gemm_compiler_opt.py      \
  --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
  --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
  --c_dtype Float16 --acc_dtype Float32                                \
  --a_major k --b_major k --c_major n                                  \
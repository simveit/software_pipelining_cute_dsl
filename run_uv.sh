echo "RUN SIMPLE GEMM"

uv run dense_gemm_simple.py                                            \
  --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
  --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
  --c_dtype Float16 --acc_dtype Float32                                \
  --a_major k --b_major k --c_major n                                  \
  --warmup_iterations 100 --iterations 1000                            


echo "##################################"
echo "RUN GEMM WITH PREFETCH"

uv run dense_gemm.py                                                  \
  --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
  --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
  --c_dtype Float16 --acc_dtype Float32                                \
  --a_major k --b_major k --c_major n                                  \
  --warmup_iterations 100 --iterations 1000                            


echo "##################################"
echo "RUN GEMM WITH COMPILER OPT"

uv run dense_gemm_compiler_opt.py                                      \
  --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
  --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
  --c_dtype Float16 --acc_dtype Float32                                \
  --a_major k --b_major k --c_major n                                  \
  --warmup_iterations 100 --iterations 1000                            

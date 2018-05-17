__device__ inline float matget(float* mat, const int x, const int y, const int w) {
	return mat[x + y * w];
}

__device__ void get_matrix_values_for_vertex(float* mat, const int vi, const int mesh_width, float* vals, int* ids, float& a)
{
    const int vxi = vi % mesh_width;
    const int vyi = vi / mesh_width;
    
    const int x = vxi * 2;
    const int y = vyi * 2;
    
    const int mat_width = mesh_width * 2 - 1;
    
	if (vxi - 1 >= 0)
	{
		vals[0] = matget(mat, x - 1, y, mat_width);
		ids[0] = vi - 1;
	}
	else
	{
		vals[0] = 0.0;
		ids[0] = 0;
	}

	if (vyi - 1 >= 0)
	{
		vals[1] = matget(mat, x, y - 1, mat_width);
		ids[1] = vi - mesh_width;
	}
	else
	{
		vals[1] = 0.0;
		ids[1] = 0;
	}

	if (vxi + 1 < mesh_width && vyi - 1 >= 0)
	{
		vals[2] = matget(mat, x + 1, y - 1, mat_width);
		ids[2] = vi - mesh_width + 1;
	}
	else
	{
		vals[2] = 0.0;
		ids[2] = 0;
	}

	if (vxi + 1 < mesh_width)
	{
		vals[3] = matget(mat, x + 1, y, mat_width);
		ids[3] = vi + 1;
	}
	else
	{
		vals[3] = 0.0;
		ids[3] = 0;
	}

	if (vyi + 1 < mesh_width)
	{
		vals[4] = matget(mat, x, y + 1, mat_width);
		ids[4] = vi + mesh_width;
	}
	else
	{
		vals[4] = 0.0;
		ids[4] = 0;
	}

	if (vxi - 1 >= 0 && vyi + 1 < mesh_width)
	{
		vals[5] = matget(mat, x - 1, y + 1, mat_width);
		ids[5] = vi + mesh_width - 1;
	}
	else
	{
		vals[5] = 0.0;
		ids[5] = 0;
	}

	a = matget(mat, x, y, mat_width);
}

__device__ void gpu_inner(const int vi, const int mesh_width, float* Jtz_vec, float* JtJ_mat, float* h) {
	float xn = Jtz_vec[vi];
	float acc = 0.0f;

	float vals[6] { 0 };
	int ids[6] { 0 };

	float a;
	get_matrix_values_for_vertex(JtJ_mat, vi, mesh_width, vals, ids, a);

#pragma unroll
	for (int j = 0; j < 6; j++) {
		acc += vals[j] * h[ids[j]];
	}

	xn -= acc;

	const float w = 1.0f;
	h[vi] = (1.f - w) * h[vi] + w * xn / a;
}



__global__ void solve_kernel(const int xoffset, const int yoffset, const int mesh_width, float* Jtz_vec, float* JtJ_mat, float* h) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= (mesh_width / 2) * (mesh_width / 2)) {
		return;
	}

	int x = (idx % (mesh_width / 2)) * 2 + xoffset;
	int y = (idx / (mesh_width / 2)) * 2 + yoffset;
	int vi = x + y * mesh_width;

	if (x + y * mesh_width >= mesh_width * mesh_width)
		return;

	if ((x % 2 == xoffset) && (y % 2 == yoffset)) {
		gpu_inner(vi, mesh_width, Jtz_vec, JtJ_mat, h);
	}
}

template <typename T>
void Mesh<T>::parallel_gpu_solve(const int, const int, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>) {

    assert(false && "GPU wolver works only with float meshes");
}

template <> // Works only for float meshes
void Mesh<float>::parallel_gpu_solve(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<float, Eigen::Dynamic, 1>> h) {
  const float* Jtz_vec = Jtz[level].get_vec().data();
  const float* JtJ_mat = JtJ[level].get_mat().data();
  const int mesh_width = JtJ[level].get_width();
  const int mesh_width_squared = mesh_width*mesh_width;
  const int mat_width = JtJ[level].get_mat().rows();
  const int mat_width_squared = mat_width*mat_width;

	float *devJtz, *devJtJ, *devH;
	CUDA_CHECK(cudaMalloc(&devJtz, Jtz[level].get_vec().rows() * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devJtJ, mat_width_squared * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&devH, h.rows() * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(devJtz, Jtz_vec, Jtz[level].get_vec().rows() * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(devJtJ, JtJ_mat, mat_width_squared * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(devH, h.data(), h.rows() * sizeof(float), cudaMemcpyHostToDevice));

	for (int it = 0; it < iterations; it++) {
		solve_kernel<<<mesh_width_squared, 1>>>(0, 0, mesh_width, devJtz, devJtJ, devH);
		cudaDeviceSynchronize();
		solve_kernel<<<mesh_width_squared, 1>>>(1, 0, mesh_width, devJtz, devJtJ, devH);
		cudaDeviceSynchronize();
		solve_kernel<<<mesh_width_squared, 1>>>(0, 1, mesh_width, devJtz, devJtJ, devH);
		cudaDeviceSynchronize();
		solve_kernel<<<mesh_width_squared, 1>>>(1, 1, mesh_width, devJtz, devJtJ, devH);
		cudaDeviceSynchronize();
	}

	CUDA_CHECK(cudaMemcpy(h.data(), devH, h.rows() * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(devH));
	CUDA_CHECK(cudaFree(devJtz));
	CUDA_CHECK(cudaFree(devJtJ));
}

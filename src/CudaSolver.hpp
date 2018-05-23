//__device__ inline float matget(float* mat, const int x, const int y, const int w) {
//	return mat[x + y * w];
//}

__device__ void gpu_inner(const int vi, const int mesh_width, const JtzVector<float> Jtz_vec, const JtJMatrixGrid<float> JtJ_mat, float* h) {
	float xn = Jtz_vec.get(vi);
	float acc = 0.0f;

	float vals[6] { 0 };
	int ids[6] { 0 };

	float a;
	JtJ_mat.get_matrix_values_for_vertex(vi, vals, ids, a);

#pragma unroll
	for (int j = 0; j < 6; j++) {
		acc += vals[j] * h[ids[j]];
	}

	xn -= acc;

	const float w = 1.0f;
	h[vi] = (1.f - w) * h[vi] + w * xn / a;
}



__global__ void solve_kernel(const int xoffset, const int yoffset, const int mesh_width, const JtzVector<float> Jtz_vec, const JtJMatrixGrid<float> JtJ_mat, float* h) {
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

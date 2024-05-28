#include "algorithms/energetic/normalisation/energetic_normalisation.cuh"

__global__ void kernel_apply_forces_and_normalise(vector3* dev_points, vector3* dev_unit_vectors, int N, real_t distance, real_t precision)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N - 1)
	{
		int point_id = tid + 1;
		int unit_vector_id = tid;
		for (int i = 0; i < point_id; i++)
		{
			if (abs(point_id - i) > 1
				&& algorithms::model::get_distance(dev_points[i], dev_points[point_id]) < distance - precision)
			{
				vector3 force;
				force.x = dev_points[point_id].x - dev_points[i].x;
				force.y = dev_points[point_id].y - dev_points[i].y;
				force.z = dev_points[point_id].z - dev_points[i].z;

				real_t norm = algorithms::model::norm(force);
				if (norm < precision)
				{
					real_t tmp = dev_unit_vectors[unit_vector_id].x;
					dev_unit_vectors[unit_vector_id].x += dev_unit_vectors[unit_vector_id].z;
					dev_unit_vectors[unit_vector_id].z += dev_unit_vectors[unit_vector_id].y;
					dev_unit_vectors[unit_vector_id].y += tmp;
				}
				else
				{
					dev_unit_vectors[unit_vector_id].x += force.x / norm;
					dev_unit_vectors[unit_vector_id].y += force.y / norm;
					dev_unit_vectors[unit_vector_id].z += force.z / norm;
				}	
				norm = algorithms::model::norm(dev_unit_vectors[unit_vector_id]);
				dev_unit_vectors[unit_vector_id].x /= norm;
				dev_unit_vectors[unit_vector_id].y /= norm;
				dev_unit_vectors[unit_vector_id].z /= norm;
			}
		}
	}
}

bool algorithms::energetic::normalisation_method::main_loop(int N, int max_iterations)
{
	if (N < 1 || max_iterations < 1) return false;

	// generating random unit_vectors
	int number_of_blocks = (N + EN_BLOCK_SIZE - 1) / EN_BLOCK_SIZE;
	algorithms::randomization::kernel_generate_random_unit_vectors<<<number_of_blocks, EN_BLOCK_SIZE>>>(dev_unit_vectors, dev_states, N);
	cuda_check_terminate(cudaDeviceSynchronize());

	// determining points
	// thrust no operator matches error resolved here https://stackoverflow.com/questions/18123407/cuda-thrust-reduction-with-double2-arrays
	// eventually thrust does not implement operator+ for float3 or double3
	thrust::device_ptr<vector3> dev_unit_vectors_ptr = thrust::device_ptr<vector3>(dev_unit_vectors);
	thrust::device_ptr<vector3> dev_points_ptr = thrust::device_ptr<vector3>(dev_points);
	cuda_check_errors_status_terminate(thrust::exclusive_scan(dev_unit_vectors_ptr, dev_unit_vectors_ptr + N, dev_points_ptr, init_point, add));
	
	{
		vector3* start = new vector3[N];
		if (start)
		{
			cuda_check_terminate(cudaMemcpy(start, dev_points, sizeof(vector3) * N, cudaMemcpyDeviceToHost));
			std::cout << "Copied generated points" << std::endl;

			create_pdb_file(start, N, BEFORE_PDB_FILE_NAME);
			open_chimera(BEFORE_PDB_FILE_NAME);
			delete[] start;
		}
	}

	int iterations = 0;
	do 
	{
		// applying forces and normalising
		kernel_apply_forces_and_normalise << <number_of_blocks, EN_BLOCK_SIZE >> > (dev_points, dev_unit_vectors, N, DISTANCE, EN_PRECISION);
		cuda_check_terminate(cudaDeviceSynchronize());

		// determining new particles
		cuda_check_errors_status_terminate(thrust::exclusive_scan<thrust::device_ptr<vector3>>(dev_unit_vectors_ptr, dev_unit_vectors_ptr + N, dev_points_ptr, init_point, add));
		
		std::cout << iterations << std::endl;
	} while (!validator.validate(dev_points, N, DISTANCE, EN_PRECISION) && (iterations++ < max_iterations || max_iterations < 0));
	return iterations < max_iterations || max_iterations < 0;
}

bool algorithms::energetic::normalisation_method::run(vector3** result, void* p_void)
{
	parameters* p = (parameters*)p_void;
	
	if (result && *result && allocate_memory(p->N))
	{
		int number_of_blocks = (p->N + EN_BLOCK_SIZE - 1) / EN_BLOCK_SIZE;

		// setting seed
		algorithms::randomization::kernel_setup << <number_of_blocks, EN_BLOCK_SIZE >> > (dev_states, p->N, time(0), OFFSET);
		cuda_check_terminate(cudaDeviceSynchronize());

		// main loop
		while (!main_loop(p->N, EN_MAX_ITERATIONS));

		cuda_check_terminate(cudaMemcpy(*result, dev_points, sizeof(vector3) * p->N, cudaMemcpyDeviceToHost));

		release_memory();
		return true;
	}
	return false;
}

// make it more agile and release memory only if is mandatory to extend the size of the memory
bool algorithms::energetic::normalisation_method::allocate_memory(int N)
{
	// checking whether the input is correct
	if (N < 0) return false;

	bool allocation_failure = false;

	cuda_allocate((void**)&dev_unit_vectors, sizeof(vector3) * (N - 1), &allocation_failure);
	cuda_allocate((void**)&dev_points, sizeof(vector3) * N, &allocation_failure);
	cuda_allocate((void**)&dev_states, sizeof(curandState) * N, &allocation_failure);

	if (allocation_failure)
	{
		release_memory();
		return false;
	}

	// in case of success
	return true;
}

void algorithms::energetic::normalisation_method::release_memory()
{
	// freeing the memory if is allocated
	cuda_release((void**)&dev_unit_vectors);
	cuda_release((void**)&dev_points);
	cuda_release((void**)&dev_states);
}

algorithms::energetic::normalisation_method::normalisation_method(validators::abstract_validator& validator): validator{validator}
{
	this->dev_points = nullptr;
	this->dev_unit_vectors = nullptr;
	this->dev_states = nullptr;
}
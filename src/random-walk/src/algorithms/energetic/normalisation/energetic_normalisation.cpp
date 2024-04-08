#include "algorithms/energetic/normalisation/energetic_normalisation.cuh"

bool algorithms::energetic::normalisation_method::main_loop(int N, int max_iterations)
{
	if (N < 1 || max_iterations < 1) return false;

	// generating random unit_vectors
	int number_of_blocks = (N + EN_BLOCK_SIZE - 1) / EN_BLOCK_SIZE;
	algorithms::randomization::kernel_generate_random_unit_vectors(dev_unit_vectors, dev_states, N);
	
	// determining particles
	thrust::device_ptr<model::particle> dev_unit_vectors_ptr = thrust::device_ptr<model::particle>(dev_unit_vectors);
	thrust::device_ptr<model::particle> dev_points_ptr = thrust::device_ptr<model::particle>(dev_points);
	cuda_check_errors_status_terminate(thrust::exclusive_scan(dev_unit_vectors_ptr, dev_unit_vectors_ptr + N, dev_points_ptr));

	int iterations = 0;
	do 
	{
		// applying forces and normalising
		
		// determining new particles
		cuda_check_errors_status_terminate(thrust::exclusive_scan(dev_unit_vectors_ptr, dev_unit_vectors_ptr + N, dev_points_ptr));

	} while (!validator.validate(dev_points, N, DISTANCE, EN_PRECISION) && iterations++ < max_iterations);
	return iterations < max_iterations;
}

bool algorithms::energetic::normalisation_method::run(algorithms::model::particle** result, int N)
{
	if (allocate_memory(N))
	{
		int number_of_blocks = (N + EN_BLOCK_SIZE - 1) / EN_BLOCK_SIZE;

		// setting seed
		algorithms::randomization::kernel_setup << <number_of_blocks, EN_BLOCK_SIZE >> > (dev_states, N, SEED, OFFSET);
		cuda_check_terminate(cudaDeviceSynchronize());

		while (!main_loop(N, EN_MAX_ITERATIONS));
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

	// allocation of unit vectors array
	if (!cuda_check_continue(cudaMalloc(&dev_unit_vectors, sizeof(model::particle) * N)))
	{
		dev_unit_vectors = nullptr;
		allocation_failure = true;
	}

	// allocation of points array
	if (!allocation_failure && !cuda_check_continue(cudaMalloc(&dev_points, sizeof(model::particle) * N)))
	{
		dev_points = nullptr;
		allocation_failure = true;
	}

	if (!allocation_failure && !cuda_check_continue(cudaMalloc(&dev_points, sizeof(curandState) * N)))
	{
		dev_states = nullptr;
		allocation_failure = true;
	}

	if (allocation_failure)
	{
		release_memory();
	}

	// in case of success
	return true;
}

// TODO: move it somewhere so that it could be used to freeing memory
void algorithms::energetic::normalisation_method::cuda_release(void** dev_ptr)
{
	if (dev_ptr)
	{
		cuda_check_terminate(cudaFree(dev_ptr));
		dev_ptr = nullptr;
	}
}

void algorithms::energetic::normalisation_method::release_memory()
{
	// freeing the memory if is allocated
	cuda_release((void**)&dev_unit_vectors);
	cuda_release((void**)&dev_points);
	cuda_release((void**)&dev_states);
}

bool algorithms::energetic::normalisation_method::generate_random_unit_vectors(int N)
{


	return false;
}

algorithms::energetic::normalisation_method::normalisation_method(validators::abstract_validator& validator): validator{validator}
{
	this->dev_points = nullptr;
	this->dev_unit_vectors = nullptr;
}
#include "algorithms/energetic/normalisation/energetic_normalisation.cuh"

bool algorithms::energetic::normalisation_method::main_loop(int N, int max_iterations)
{
	// generating random unit_vectors
	
	// determining particles

	int iterations = 0;
	do 
	{
		// applying forces and normalising
		
		// determining new particles
	} while (!validator.validate(dev_points, N, DISTANCE, EN_PRECISION) && iterations++ < max_iterations);
	return iterations < max_iterations;
}

bool algorithms::energetic::normalisation_method::run(algorithms::model::particle** result, int N)
{
	if (allocate_memory(N))
	{
		while (!main_loop(N, EN_MAX_ITERATIONS));
		release_memory();
		return true;
	}
	return false;
}

bool algorithms::energetic::normalisation_method::allocate_memory(int N)
{
	// checking whether the input is correct
	if (N < 0) return false;

	// allocation of unit vectors array
	if (!cuda_check_continue(cudaMalloc(&dev_unit_vectors, sizeof(model::particle) * N)))
	{
		dev_unit_vectors = nullptr;
		return false;
	}

	// allocation of points array
	if (!cuda_check_continue(cudaMalloc(&dev_points, sizeof(model::particle) * N)))
	{
		// releasing the memory of unit vectors array in case of cudaMalloc failure
		cuda_check_terminate(cudaFree(dev_unit_vectors));
		dev_unit_vectors = nullptr;
		dev_points = nullptr;
		return false;
	}

	// in case of success
	return true;
}

void algorithms::energetic::normalisation_method::release_memory()
{
	// releasing the memory if is allocated
	if (dev_unit_vectors)
	{
		cuda_check_terminate(cudaFree(dev_unit_vectors));
		dev_unit_vectors = nullptr;
	}
	if (dev_points)
	{
		cuda_check_terminate(cudaFree(dev_points));
		dev_points = nullptr;
	}
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
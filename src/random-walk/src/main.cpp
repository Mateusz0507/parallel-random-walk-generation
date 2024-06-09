#include "main.h"


int main(int argc, char** argv)
{
	parameters p;

	if (!read(argc, argv, p))
		return 1;

	auto validator = algorithms::validators::single_check_validator::single_check_validator();
	vector3* result = new vector3[p.N];

	std::chrono::steady_clock::time_point start_time = std::chrono::high_resolution_clock::now();
	
	{
		algorithms::genetic::genetic_improved_method::parameters params;
		params.N = p.N = 500;
		params.generation_size = p.generation_size = 40;
		params.mutation_ratio = p.mutation_ratio = 0.04;
		algorithms::genetic::genetic_method method;
		method.run(&result, &params);
	}

	/*if (std::string(p.method) == "naive")
	{
		auto method = algorithms::energetic::naive_method::naive_method(validator);

		algorithms::energetic::naive_method::parameters naive_parameters;
		naive_parameters.N = p.N;
		naive_parameters.directional_level = p.directional_level;
		naive_parameters.segments_number = p.segments_number;

		method.run(&result, &naive_parameters);
	}
	else if (std::string(p.method) == "normalization")
	{
		auto method = algorithms::energetic::normalisation_method(validator);

		algorithms::energetic::normalisation_method::parameters normalization_parameters;
		normalization_parameters.N = p.N;
		normalization_parameters.directional_level = p.directional_level;
		normalization_parameters.segments_number = p.segments_number;

		method.run(&result, &normalization_parameters);
	}
	else if (std::string(p.method) == "genetic")
	{
		algorithms::genetic::genetic_method method;

		algorithms::genetic::genetic_method::parameters genetic_parameters;
		genetic_parameters.N = p.N;
		genetic_parameters.mutation_ratio = p.mutation_ratio;
		genetic_parameters.generation_size = p.generation_size;

		method.run(&result, &genetic_parameters);
	}*/

	std::chrono::steady_clock::time_point end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

	if (create_pdb_file(result, p.N, AFTER_PDB_FILE_NAME))
		open_chimera(AFTER_PDB_FILE_NAME);

	add_test_to_csv(p, duration.count());

	delete[] result;
}

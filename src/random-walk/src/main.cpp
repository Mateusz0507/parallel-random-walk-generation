#include "main.h"
#include "chimera/chimera.h"


int main(int argc, char** argv)
{
	parameters p;

	algorithms::genetic::genetic_method::parameters params;
	params.N = 100;
	params.mutation_ratio = 0.05;
	params.generation_size = 2;

	vector3* result = new vector3[params.N];

	algorithms::genetic::genetic_method method;
	method.run(&result, &params);

	/*if (!read(argc, argv, p))
		return 1;

	auto validator = algorithms::validators::single_check_validator::single_check_validator();
	vector3* result = new vector3[p.N];

	if (std::string(p.method) == "naive")
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

		method.run(&result, &normalization_parameters);
	}

	if (create_pdb_file(result, p.N, "walk"))
		open_chimera("walk");
	delete[] result;*/
}

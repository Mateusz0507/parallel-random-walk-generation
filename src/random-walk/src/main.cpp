#include "main.h"
#include "chimera/chimera.h"


int main(int argc, char** argv)
{
	parameters p;

	if (!read(argc, argv, p))
		return 1;

	auto validator = algorithms::energetic::validators::single_check_validator::single_check_validator();
	vector3* result = new vector3[p.N];

	if (std::string(p.method) == "naive")
	{
		auto method = algorithms::energetic::naive_method::naive_method(validator);
		method.run(&result, p.N);
	}
	else if (std::string(p.method) == "normalization")
	{
		auto method = algorithms::energetic::normalisation_method(validator);
		method.run(&result, p.N);
	}

	if (create_pdb_file(result, p.N, "walk"));
		open_chimera("walk");
	delete[] result;
}

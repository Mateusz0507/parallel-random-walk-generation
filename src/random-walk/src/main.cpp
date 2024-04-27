#include "main.h"
#include "chimera/chimera.h"


int main(int argc, char** argv)
{
	parameters p;
	if (read(argc, argv, p))
	{
		// p.length = 10000;
		if (p.method == 0)
		{
			auto validator = algorithms::energetic::validators::single_check_validator::single_check_validator();
			auto method = algorithms::energetic::naive_method::naive_method(validator);
			// auto method = algorithms::energetic::normalisation_method(validator);
			
			vector3* result = new vector3[p.length];
			method.run(&result, p.length);
			if (create_pdb_file(result, p.length, "walk"));
				open_chimera("walk");
			delete[] result;
		}
	}
}

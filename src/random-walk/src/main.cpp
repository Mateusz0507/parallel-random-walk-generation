#include "main.h"
#include "chimera/chimera.h"

#include "algorithms/energetic/naive/energetic_naive.cuh"
#include "algorithms/energetic/validators/single_check_validator.cuh"


bool open_chimera()
{
	std::string command = CHIMERA_PATH + " " + FILE_PATH;
	system(command.c_str());
	return true;
}


int main(int argc, char** argv)
{
	parameters p;
	if (read(argc, argv, p))
	{
		if (p.method == 0)
		{
			algorithms::energetic::validators::single_check_validator validator = algorithms::energetic::validators::single_check_validator::single_check_validator();
			algorithms::energetic::naive_method method = algorithms::energetic::naive_method::naive_method(validator);
			algorithms::model::particle* result = new algorithms::model::particle[p.length];
			method.run(&result, p.length);
			if(create_file(result, p.length))
				open_chimera();
		}
	}
}

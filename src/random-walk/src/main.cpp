#include "main.h"

#include "algorithms/energetic/naive/energetic_naive.cuh"
#include "algorithms/energetic/validators/single_check_validator.cuh"


int main(int argc, char** argv)
{
	parameters p;
	if (read(argc, argv, p))
	{
		if (p.method == 0)
		{
			algorithms::energetic::validators::single_check_validator validator = algorithms::energetic::validators::single_check_validator::single_check_validator();
			algorithms::energetic::naive_method method = algorithms::energetic::naive_method::naive_method(validator);
			algorithms::model::particle* particles = static_cast<algorithms::model::particle*>(malloc(p.length * sizeof(algorithms::model::particle)));
			method.run(&particles, p.length);

			std::cout << "End of main" << std::endl;
		}
	}
}
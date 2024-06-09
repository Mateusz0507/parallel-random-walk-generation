#include "program_parametrization.h"


void program_parametrization::print_usage(const char* name)
{
    std::cerr << "Usage " << name << ":" << std::endl
    << "[-m/--method]=[naive/normalization/genetic] (default:naive)" << std::endl
    << "[-N/--N]=[int] (default:100)" << std::endl
    << "[-d/--directional-level]=[int] (default:0)" << std::endl
    << "[-s/--segments--number]=[int] (default:1)" << std::endl
    << "Parameters for genetic method only:" << std::endl
    << "   --mutation-ratio=[float] (default:0.05)" << std::endl
    << "   --generation-size=[int] (default:10)" << std::endl;
}

bool program_parametrization::read(int argc, char** argv, parameters& p)
{
	const char* name = argv[0];

    for (int i = 1; i < argc; ++i) {
        char* value = nullptr;
        char* parameter = strtok_s(argv[i], "=", &value);

        if (parameter == nullptr || strlen(parameter) == 0 || value == nullptr || strlen(value) == 0)
        {
            print_usage(name);
            return false;
        }

        if (std::string(parameter) == "-m" || std::string(parameter) == "--method") {
            if (std::string(value) != "naive" && std::string(value) != "normalization" && std::string(value) != "genetic")
            {
                print_usage(name);
                return false;
            }
            p.method = value;
        }
        else if (std::string(parameter) == "-N" || std::string(parameter) == "--N") {
            p.N = atoi(value);
        }
        else if (std::string(parameter) == "-d" || std::string(parameter) == "--directional-level") {
            p.directional_level = atoi(value);
        }
        else if (std::string(parameter) == "-s" || std::string(parameter) == "--segments-number") {
            p.segments_number = atoi(value);
        }
        else if (std::string(p.method) == "genetic")
        {
            if (std::string(parameter) == "--mutation-ratio") {
                p.mutation_ratio = atof(value);
            }
            else if (std::string(parameter) == "--generation-size") {
                p.generation_size = atoi(value);
            }
            else
            {
                print_usage(name);
                return false;
            }
        }
        else
        {
            print_usage(name);
            return false;
        }
    }

    return true;
}

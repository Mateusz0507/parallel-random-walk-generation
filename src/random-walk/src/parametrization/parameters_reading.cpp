#include "program_parametrization.h"

bool program_parametrization::read(int argc, char** argv, parameters& p)
{
	switch (argc)
	{
	case DEFAULT_PARAMS_COUNT:
		if (argv[1])
		{
			try
			{
				p.length = stoi(argv[1]);
			}
			catch (const std::exception e)
			{
				error(e.what());
			}
		}
		else
		{
			error("nullptr passed to stoi");
		}

		break;
	case 1:
		p.length = 100;
		p.method = 0;
		return true;
		break;
	default:
		error("Wrong parameters passed!\n");
	}
	return false;
}
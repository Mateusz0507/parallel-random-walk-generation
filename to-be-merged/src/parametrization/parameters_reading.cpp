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
				ERROR(e.what());
			}
		}
		else
		{
			ERROR("nullptr passed to stoi");
		}

		break;
	default:
		ERROR("Wrong parameters passed!\n");
	}
	return false;
}
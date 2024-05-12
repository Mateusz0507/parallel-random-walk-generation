#include "program_parametrization.h"
#include <vector>
#include <string>
#include <sstream>
#include <regex>

program_data parameters_reader::read(int argc, const char** argv)
{
    program_data data;
    data.is_invokable = false;
    if (!validateGNUStandard(argc, argv)) 
    {
        return data;
    }
    
    std::vector<std::string> arguments;
    for (int i = 1; i < argc; i++)
    {
        arguments.push_back(argv[i]);
    }
    
    auto name_it = std::find_if(arguments.cbegin(), arguments.cend(), [](const std::string& arg) {
        return arg.find("--name", 0) == 0;
        });

    if (name_it == arguments.cend())
    {
        return data;
    }




}

bool parameters_reader::validateGNUStandard(int argc, const char** argv)
{
    std::regex regex("^--\w[-\w]+(=\w+|)$");
    for (int i = 1; i < argc; i++) {
        if (!std::regex_match(argv[i], regex))
            return false;
    }
    return true;
}

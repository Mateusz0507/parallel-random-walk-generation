#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

#include "constants.h"
#include "parametrization/program_parametrization.h"
#include "chimera/chimera.h"


bool file_exists(const std::string& file_path);
std::string current_date_time();
std::string format_duration(long long milliseconds);
void add_test_to_csv(program_parametrization::parameters p, long long duration_ms);

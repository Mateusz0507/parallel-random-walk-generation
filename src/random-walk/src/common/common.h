// file that defines global MACROS and global includes
#pragma once

#include <cstdlib>
#include <iostream>

using namespace std;

#define error(msg) perror("Error occurred."), fprintf(stderr, "[%s] %d, %s", __FILE__, __LINE__, msg), exit(EXIT_FAILURE)

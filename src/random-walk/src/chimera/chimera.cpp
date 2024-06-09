#include "chimera.h"


bool open_chimera(const std::string file_name)
{
	std::string command = CHIMERA_PATH + " " + executable_path() + PDB_FILES_FOLDER_PATH_RELATIVE_TO_EXE + file_name + ".pdb";

	STARTUPINFO si;
	PROCESS_INFORMATION pi;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));
	if (!CreateProcess(NULL, (LPSTR)command.c_str(), NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
		std::cerr << "Failed to start Chimera" << std::endl;
		return false;
	}

	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
	return true;
}

std::string executable_path() {
#ifdef _WIN32
	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	return std::string(buffer);
#elif __linux__
	char buffer[PATH_MAX];
	readlink("/proc/self/exe", buffer, sizeof(buffer));
	return std::string(buffer);
#else
#error Unsupported platform
#endif
}

std::string format_number(float number, int size)
{
    std::ostringstream oss;
	int digits_before_decimal_point = std::abs(number) < 10.0 ? 1 : static_cast<int>(std::log10(std::abs(number))) + 1;
	int precision = size - digits_before_decimal_point - 2; // -2 is for minus and decimal point
    oss << std::fixed << std::setw(size) << std::setprecision(precision) << number;
    return oss.str();
}

bool create_pdb_file(vector3* points, const int N, const std::string file_name)
{
	if (N < 3) {
		std::cerr << "Number of particles is less than 3!" << std::endl;
		return false;
	}

	std::string file_path = executable_path() + PDB_FILES_FOLDER_PATH_RELATIVE_TO_EXE + file_name + ".pdb";
	std::ofstream file(file_path);

	if (!file) {
		std::cerr << "Failed to open the file for writing!" << std::endl;
		return false;
	}

	// Save particles
	for (int i = 0; i < N; i++)
	{
		file <<
			std::left <<
			std::setw(6) << "ATOM" <<
			std::right <<
			std::setw(5) << i + 1 <<
			" " <<
			std::setw(4) << "B  " <<
			std::setw(1) << "" <<
			std::setw(3) << "BEA" <<
			" " <<
			std::setw(1) << "A" <<
			std::setw(4) << "0" <<
			std::setw(1) << "" <<
			"   " <<
			std::setprecision(3) <<
			format_number(points[i].x, 8) <<
			format_number(points[i].y, 8) <<
			format_number(points[i].z, 8) <<
			std::setw(6) << "0.00" <<
			std::setw(6) << "0.00" <<
			"          " <<
			std::setw(2) << "B" <<
			std::setw(2) << "" <<
			std::endl;
	}

	// Save connections
	int i = 1;
	file << std::right << "CONECT" << std::setw(5) << i << std::setw(5) << i + 1 << std::endl;
	for (i = 2; i < N; i++)
	{
		file << std::right << "CONECT" << std::setw(5) << i << std::setw(5) << i - 1 << std::setw(5) << i + 1 << std::endl;
	}
	file << std::right << "CONECT" << std::setw(5) << i << std::setw(5) << i - 1 << std::endl;

	file.close();
	std::cout << "File created successfully." << std::endl;
	return true;
}

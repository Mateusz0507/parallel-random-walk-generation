#include "chimera.h"


bool open_chimera(const std::string file_name)
{
	std::string command = CHIMERA_PATH + " " + PDB_FILES_FOLDER_PATH + "/" + file_name + ".pdb";;

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

bool create_pdb_file(vector3* points, const int N, const std::string file_name)
{
	if (N < 3) {
		std::cerr << "Number of particles is less than 3!" << std::endl;
		return false;
	}

	std::string file_path = PDB_FILES_FOLDER_PATH + "/" + file_name + ".pdb";
	std::ofstream file(file_path);

	if (!file) {
		std::cerr << "Failed to open the file for writing!" << std::endl;
		return false;
	}

	// Save particles
	for (int i = 0; i < N; i++)
	{
		file << std::right <<
			"ATOM" << std::setw(7) <<
			i + 1 << std::setw(3) <<
			"B" << std::setw(6) <<
			"BEA" << std::setw(2) <<
			"A" << std::setw(4) <<
			"0" << std::setprecision(3) << std::setw(12) <<
			points[i].x << std::setw(8) <<
			points[i].y << std::setw(8) <<
			points[i].z << std::setw(6) <<
			"0.00" << std::setw(6) <<
			"0.00" << std::setw(12) <<
			"B" << std::endl;
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

#include "tests.h";


bool file_exists(const std::string& file_path)
{
	std::ifstream file(file_path);
	return file.is_open();
}

std::string current_date_time()
{
	auto now = std::chrono::system_clock::now();
	std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
	std::tm now_tm = *std::localtime(&now_time_t);

	std::ostringstream oss;
	oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");

	return oss.str();
}

std::string format_duration(long long milliseconds)
{
	const long long ms_in_sec = 1000;
	const long long ms_in_min = ms_in_sec * 60;
	const long long ms_in_hour = ms_in_min * 60;

	long long hours = milliseconds / ms_in_hour;
	milliseconds %= ms_in_hour;
	long long minutes = milliseconds / ms_in_min;
	milliseconds %= ms_in_min;
	long long seconds = milliseconds / ms_in_sec;
	milliseconds %= ms_in_sec;

	std::ostringstream oss;
	if (hours > 0)
		oss << hours << "h";
	if (minutes > 0 || hours > 0)
		oss << minutes << "m";
	if (seconds > 0 || minutes > 0 || hours > 0)
		oss << seconds << "s";

	oss << milliseconds << "ms";

	return oss.str();
}

void add_test_to_csv(program_parametrization::parameters p, long long duration_ms)
{
	std::string file_path = executable_path() + CSV_FILE_PATH_RELATIVE_TO_EXE + CSV_FILE_NAME + ".csv";
	bool is_file_created = file_exists(file_path);

	std::ofstream file(file_path, std::ios::app);

	// Create first row if file doesn't exist
	if (!is_file_created)
		file << "date,N,time (ms),time,method,directional level,segments number,mutation ratio, generation size" << std::endl;


	file <<
		current_date_time() << ',' <<
		p.N << ',' <<
		duration_ms << ',' <<
		format_duration(duration_ms) << ',' <<
		p.method << ',' <<
		p.directional_level << ',';

	if (p.directional_level > 0)
		file << p.segments_number << ',';
	else
		file << "-,";

	if (std::string(p.method) == "genetic")
		file << p.mutation_ratio << ',' << p.generation_size << std::endl;
	else
		file << "-,-" << std::endl;

	file.close();
	std::cout << "Test added successfully." << std::endl;
}

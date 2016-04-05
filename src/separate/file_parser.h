#include "main.h"

bool isDirectory(char* path);
void read_csv_graph(const char* file_dir);
map<string, int> read_csv_dict(const char* file_dir);
void output_hidden(const char* input_file_dir, const char* out_file_dir);
void output_2d(const char* file_dir, vector< vector<double> > arr, int n1, int n2);
void output_1d(const char* file_dir, vector<double> arr, int n);


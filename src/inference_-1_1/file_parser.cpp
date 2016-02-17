#include "file_parser.h"

vector<string> &split(const string &s, char delim, vector<string> &elems) {
  stringstream ss(s);
  string item;
  while (getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

vector<string> split(const string &s, char delim) {
  vector<string> elems;
  split(s, delim, elems);
  return elems;
}

/** 
 * read graph in csv format
 * graph file: x,y ( x -> y )
 */
void read_csv_graph(const char* file_dir) {
  string line;
  string file_prefix = file_dir;
  N = 0; 
  for (int t = 0; t < T; t++) {
    ifstream my_file;
    stringstream ss;
    ss << t;
    string file_suffix = ss.str();
    string file_name = file_prefix + file_suffix + ".csv";
    my_file.open(file_name.c_str());

    // read N
    if (my_file) {
      while (getline(my_file, line)) {
	vector<string> vec_s = split(line, ',');
	int x = atoi(vec_s.at(0).c_str());
	int y = atoi(vec_s.at(1).c_str());
	N = max(N, x); N = max(N, y); 
      }
      my_file.close();
    }
  }
  N++;
  cout << "N = " << N << endl;

  users = vector< vector<int> >(T, vector<int>(0));
  G = vector< vector< vector<int> > >(T, vector< vector<int> >(N, vector<int>(N)));
  for (int t = 0; t < T; t++) {
    ifstream my_file;
    stringstream ss;
    ss << t;
    string file_suffix = ss.str();
    string file_name = file_prefix + file_suffix + ".csv";

    // read G and users
    map<int, int> map_t;
    my_file.open(file_name.c_str());
    if (my_file) {
      while (getline(my_file, line)) {
	vector<string> vec_s = split(line, ',');
	int x = atoi(vec_s.at(0).c_str());
	int y = atoi(vec_s.at(1).c_str());
	G.at(t).at(x).at(y) = 1;
	map_t[x] = 0; map_t[y] = 0;
      }
      for (map<int, int>::iterator it = map_t.begin(); it != map_t.end(); it++) {
	int i = it->first;
	users.at(t).push_back(it->first);
      }
      my_file.close();
    }
  }

}

/**
 * read id dictionary in csv format
 * dictionary file: <old_id>,<new_id>
 */
map<string, int> read_csv_dict(const char* file_dir) {
  map<string, int> id_map;
  ifstream my_file;
  string line;
  my_file.open(file_dir);
  if (my_file) {
    while (getline(my_file, line)) {
      vector<string> vec_s = split(line, ',');
      string x = vec_s.at(0);
      int y = atoi(vec_s.at(1).c_str());
      id_map[x] = y;
    }
    my_file.close();
  }
  return id_map;
}

/**
 * output X to a file
 * format: <id> <space> <k1, k2, ... >
 */
void output_hidden(const char* file_dir) {
  // TODO
}



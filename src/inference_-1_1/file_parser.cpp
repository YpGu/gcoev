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
  G = vector< struct sub_graph >(T);

  // read N
  N = 0; 
  for (int t = start_T; t < T; t++) {
    ifstream my_file;
    stringstream ss;
    ss << t;
    string file_suffix = ss.str();
    string file_name = file_prefix + file_suffix + ".csv";
    if (verbose)
      cout << file_name << endl;
    my_file.open(file_name.c_str());

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

  // read user map (for each t)
  for (int t = start_T; t < T; t++) {
    cout << "t = " << t << endl;
    ifstream my_file;
    stringstream ss;
    ss << t;
    string file_suffix = ss.str();
    string file_name = file_prefix + file_suffix + ".csv";
    if (verbose) cout << file_name << endl;

    my_file.open(file_name.c_str());
    if (my_file) {
      while (getline(my_file, line)) {
	vector<string> vec_s = split(line, ',');
	int x = atoi(vec_s.at(0).c_str());
	int y = atoi(vec_s.at(1).c_str());
	int weight = atoi(vec_s.at(2).c_str());
	G[t].u_map[x] = 0; G[t].u_map[y] = 0;
      }
      // assign new_id & update users_timestamps
      int new_id = 0;
      for (map<int, int>::iterator it = G[t].u_map.begin(); it != G[t].u_map.end(); it++) {
	int old_id = it->first;
	G[t].u_map[old_id] = new_id;
	G[t].u_invert_map[new_id] = old_id;
	if (users_time.find(old_id) == users_time.end()) {
	  users_time[old_id] = vector<int>(0);
	}
	users_time[old_id].push_back(t);	// user 'old_id' appears in time 't'
	new_id++;
      }
      // create graph
      G[t].n_users = new_id;
      G[t].graph = vector< vector<int> >(G[t].n_users, vector<int>(G[t].n_users));

      my_file.close();
    }
  }
  cout << "reading 1 done" << endl;

  // read G 
  for (int t = start_T; t < T; t++) {
    cout << "t = " << t << endl;
    ifstream my_file;
    stringstream ss;
    ss << t;
    string file_suffix = ss.str();
    string file_name = file_prefix + file_suffix + ".csv";
    if (verbose) cout << file_name << endl;

    my_file.open(file_name.c_str());
    if (my_file) {
      map<int, int> gmap = G[t].u_map;
      while (getline(my_file, line)) {
	vector<string> vec_s = split(line, ',');
	int x = gmap.find(atoi(vec_s[0].c_str())) -> second;
	int y = gmap.find(atoi(vec_s[1].c_str())) -> second;
	int weight = atoi(vec_s.at(2).c_str());
//	G[t].graph[x][y] = weight;
//	G[t].graph[y][x] = weight;	// for undirected graph
//	G[t].graph[x][y] = 1;
//	G[t].graph[y][x] = 1;	// for undirected graph
	G[t].graph[x][y] = (int)log(weight+1);
	G[t].graph[y][x] = (int)log(weight+1);	// for undirected graph
      }
      my_file.close();
    }
  }
  cout << "reading 2 done" << endl;
//  cin >> gu;

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
 * file_dir example: "./save/"
 */
void output_hidden(const char* file_dir) {
  map<int, int> id_map;
  if (true) {
    ifstream my_file;
    string line;
    my_file.open("../../data/dict/user_id_map.dat");
    if (my_file) {
      while (getline(my_file, line)) {
	vector<string> vec_s = split(line, ',');
	int x = atoi(vec_s[0].c_str());	  // old_id
	int y = atoi(vec_s[1].c_str());	  // original_id
	id_map[x] = y;
      }
      my_file.close();
    } else {
      cout << "user_id_map file does not exist!" << endl;
    }
  }

  string file_prefix = file_dir;
  for (int t = start_T; t < T; t++) {
    ofstream my_file;
    stringstream ss;
    ss << t;
    string file_suffix = ss.str();
    string file_name = file_prefix + file_suffix + ".txt";
    if (verbose) cout << file_name << endl;

    my_file.open(file_name.c_str());
    if (my_file) {
      int n = G[t].n_users;
      for (int i = 0; i < n; i++) {
	int old_id = G[t].u_invert_map[i];
	int original_id = id_map[old_id];
	stringstream ssp1; ssp1 << original_id;
	string newline = ssp1.str();
	for (int k = 0; k < K; k++) {
	  stringstream ssp2; ssp2 << G[t].X[i][k];
	  string s_xk = ssp2.str();
	  newline = newline + " " + s_xk;
	}
	newline = newline + "\n";
	my_file << newline;
      }

      my_file.close();
    }
  }
}



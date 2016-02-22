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

  /* read N
  int N = 0; 
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
  */

  // read user map (for each t)
  for (int t = start_T; t < T; t++) {
    cout << "t = " << t << endl;
    ifstream my_file; stringstream ss; ss << t; string file_suffix = ss.str();
    string file_name = file_prefix + file_suffix + ".csv";
    if (verbose) cout << file_name << endl;

    my_file.open(file_name.c_str());
    if (my_file) {
      while (getline(my_file, line)) {
	vector<string> vec_s = split(line, ',');
	int x = atoi(vec_s.at(0).c_str());
	int y = atoi(vec_s.at(1).c_str());
	G[t].u_map[x] = 0; G[t].u_map[y] = 0;
	vector<string>().swap(vec_s); vec_s.clear();
      }
      /* assign new_id & update users_timestamps */
      int new_id = 0;
      for (map<int, int>::iterator it = G[t].u_map.begin(); it != G[t].u_map.end(); it++) {
	int old_id = it->first;
	G[t].u_map[old_id] = new_id;
	G[t].u_invert_map[new_id] = old_id;
	new_id++;
      }
      /* create graph */
      G[t].n_users = new_id;
      G[t].graph = vector< vector<int> >(G[t].n_users, vector<int>(G[t].n_users));
      G[t].encoded_all = vector<int>(0);

      my_file.close();
    }
  }
  cout << "reading 1 done" << endl;

  /* read G & initialize G */
  for (int t = start_T; t < T; t++) {
    cout << "t = " << t << endl;
    ifstream my_file; stringstream ss; ss << t;
    string file_suffix = ss.str();
    string file_name = file_prefix + file_suffix + ".csv";
    if (verbose) cout << file_name << endl;

    my_file.open(file_name.c_str());
    if (my_file) {
      int n_t = G[t].n_users;
      map<int, int> gmap = G[t].u_map;
      while (getline(my_file, line)) {
	vector<string> vec_s = split(line, ',');
	int x = gmap.find(atoi(vec_s[0].c_str())) -> second;
	int y = gmap.find(atoi(vec_s[1].c_str())) -> second;
	int weight = atoi(vec_s[2].c_str());
	if (weight < 1000) continue;
//	G[t].graph[x][y] = weight;
//	G[t].graph[y][x] = weight;	// for undirected graph
	G[t].graph[x][y] = 1;
	G[t].graph[y][x] = 1;	// for undirected graph
//	G[t].graph[x][y] = (int)log(weight+1);
//	G[t].graph[y][x] = (int)log(weight+1);	// for undirected graph
	G[t].encoded_all.push_back(x * n_t + y);
	vector<string>().swap(vec_s); vec_s.clear();
      }
      /* reading done, sample non-existing links */
      vector<int> all_users = vector<int>();
      for (int i = 0; i < n_t; i++)
	all_users.push_back(i);
      for (int i = 0; i < n_t; i++) {
	/* existing links */
	vector<int> pos_users = vector<int>();
	for (int j = 0; j < n_t; j++) if (i != j && G[t].graph[i][j] > 0) {
	  pos_users.push_back(j);
	}

	// test
//	cout << "--- i = " << i << ", size = " << pos_users.size() << endl; 

	vector<int> diff = vector<int>(n_t);
	/* TODO: check */
	vector<int>::iterator it = set_difference(all_users.begin(), all_users.end(), 
	    pos_users.begin(), pos_users.end(), diff.begin());
	diff.resize(it - diff.begin());
	if (diff.size() >= pos_users.size()) {
	  random_shuffle(diff.begin(), diff.end());
	  for (int a = 0; a < pos_users.size(); a++) {
	    int j = diff[a];
	    int encode = i * n_t + j;
	    G[t].encoded_all.push_back(encode);
	  }
	} else {
	  for (int a = 0; a < diff.size(); a++) {
	    int j = diff[a];
	    int encode = i * n_t + j;
	    G[t].encoded_all.push_back(encode);
	  }
	}

	/* free */
	vector<int>().swap(pos_users); pos_users.clear();
	vector<int>().swap(diff); diff.clear();
      }
//      int gu; cin >> gu;
      /* free */
      vector<int>().swap(all_users); all_users.clear();

      /* initialize hX */
      G[t].hgX = vector< vector<double> >(n_t, vector<double>(K));
      for (int i = 0; i < n_t; i++) {
	for (int k = 0; k < K; k++) {
	  G[t].hgX[i][k] = numeric_limits<double>::min();
	}
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
      vector<string>().swap(vec_s); vec_s.clear();
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
	vector<string>().swap(vec_s); vec_s.clear();
      }
      my_file.close();
    } else {
      cout << "user_id_map file does not exist!" << endl;
    }
  }

  string file_prefix = file_dir;
  for (int t = start_T; t < T; t++) {
    ofstream my_file; stringstream ss; ss << t;
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


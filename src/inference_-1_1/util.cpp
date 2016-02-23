#include "util.h"

int N, K = 1;
int T = 114;	// real data
//int T = 3;	// toy example
int ITER = 100;
int start_T = 105;
bool verbose = true;

vector<struct sub_graph> G;		// T*1
map< int, vector<int> > users_time;	// N*1; records all the time stamps each user appears in



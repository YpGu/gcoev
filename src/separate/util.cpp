#include "util.h"

int K = 1;
int T = 114;	// real data
//int T = 3;	// toy example
int ITER = 30;
int start_T = 110;
bool verbose = true;

double stepsize = 0.1;
double sigma = 50;
double lambda = 0.5;

vector<struct sub_graph> G;		// T*1
map< int, vector<int> > users_time;	// N*1; records all the time stamps each user appears in



#include "util.h"

int K = 1;
int T = 114;	// co-voting dataset
int start_T = 0;
int ITER = 20;
bool verbose = true;

double stepsize = 1;			// initial stepsize
double sigma = 3;			// 1/(2*sigma*sigma) is the regularization coefficient
//double lambda = 0.5;			// (1-lambda) * self + lambda * ave_neighbor
double lambda = 0;			// (1-lambda) * self + lambda * ave_neighbor

vector<struct sub_graph> G;		// T*1


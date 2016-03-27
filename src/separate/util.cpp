#include "util.h"

int K = 1;
int T = 105;	// co-voting dataset
int start_T = 100;
int ITER = 20;
bool verbose = false;

double stepsize = 0.05;			// initial stepsize (for batch gradient descent)
double delta = 1;			// 1/(2*delta*delta) is the regularization coefficient
double lambda = 0.5;			// (1-lambda) * self + lambda * ave_neighbor
//double lambda = 0;			// (1-lambda) * self + lambda * ave_neighbor
//double lambda = 1;			// (1-lambda) * self + lambda * ave_neighbor

vector<double> alpha_s;			// T*1:  p(z_{it} = 1) = alpha[t]; p(z_{it} = 0) = 1-alpha[t];

vector<struct sub_graph> G;		// T*1


#include "util.h"

int K = 1;
//int T = 114;	// co-voting dataset
int T = 95;	// co-voting dataset
int start_T = 85;
int TOT_ITER = 50;
int M_ITER = 200;
bool verbose = false;

double stepsize = 0.005;			// initial stepsize (for batch gradient descent)
//double delta = 0.2;			// 1/(2*delta*delta) is the regularization coefficient

double delta = 1.0;			// 1/(2*delta*delta) is the regularization coefficient

//double lambda = 0.5;			// (1-lambda) * self + lambda * ave_neighbor
double lambda = 0;			// (1-lambda) * self + lambda * ave_neighbor
//double lambda = 1;			// (1-lambda) * self + lambda * ave_neighbor

vector<double> alpha_s;			// T*1: p(z_{it} = 1) = alpha[t]; p(z_{it} = 0) = 1-alpha[t];
vector<double> likel;			// T*1: stores log likelihood 
vector< vector<double> > v;		// T*n: latent assignment 

vector<struct sub_graph> G;		// T*1


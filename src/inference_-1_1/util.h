#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cmath>
#include <limits>

using namespace std;

extern int N;	  // total number of users
extern int E;	  // total number of edges(?)
extern int K;	  // number of latent dimension
extern int T;	  // total number of timestamps
extern int ITER;  // total number of iterations
extern vector< vector< vector<int> > > G;		      // T * N * N
extern vector< vector< vector<double> > > X;		      // T * N * K
extern vector< vector< vector<double> > > Sigma;
extern vector< vector< vector<double*> > > log_pt;	      // joint p(r,s)
extern vector< vector< vector<double*> > > log_pt_tik;	      // marginal p(r)
extern vector< vector<int> > users;

extern vector< vector< vector<double*> > > log_pt;
extern vector< vector< vector<double*> > > log_pt_tik; 


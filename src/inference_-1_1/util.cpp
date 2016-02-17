#include "util.h"

int N, E, K = 1, T = 3;
int ITER = 100;
vector< vector< vector<int> > > G;	      // T * N * N
vector< vector< vector<double> > > X;	      // T * N * K
vector< vector< vector<double> > > Sigma;     // T * N * N
vector< vector< vector<double*> > > log_pt;	  // joint p(r,s)
vector< vector< vector<double*> > > log_pt_tik;   // marginal p(r)
vector< vector<int> > users;		      // T * ?



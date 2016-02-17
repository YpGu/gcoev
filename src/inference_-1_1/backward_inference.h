#include "compute_logl.h"
#include "util.h"

void backward();
/*
    vector< vector< vector<int> > > G,		      // T * N * N
    vector< vector< vector<double> > > X,	      // T * N * K
    vector< vector< vector<double> > > Sigma,
    vector< vector< vector<double*> > > log_pt,	      // joint p(r,s)
    vector< vector< vector<double*> > > log_pt_tik,   // marginal p(r)
    vector< vector<int> > users
  );
*/


#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cmath>
#include <limits>

using namespace std;

extern int N;	      // total number of users
extern int K;	      // number of latent dimension
extern int T;	      // total number of timestamps
extern int start_T;   // starting time stamp
extern int ITER;      // total number of iterations
extern bool verbose;  // do you want to see more outputs?

/* graph information for each time stamp */
struct sub_graph {
  int n_users;
  map<int, int> u_map;		  // global_id -> local_id
  map<int, int> u_invert_map;	  // local_id -> global_id (only used for reverse lookup in final results)
  vector< vector<int> > graph;	  // graph: n*n
  vector< vector<double> > Sigma; // sigmoid(X_i, X_j) for all (i,j)'s: n*n
  vector< vector<double> > X;	  // X: n*K
  vector< vector<double*> > log_pt;	  // log_pt: n*K; joint p(r,s)
  vector< vector<double*> > log_pt_tik;	  // log_pt_tik: n*K; marginal p(r)
};

extern vector<struct sub_graph> G;	      // T*1
extern map< int, vector<int> > users_time;    // N*1; records all the time stamps each user appears in
					      // (in chronological order) 

#endif


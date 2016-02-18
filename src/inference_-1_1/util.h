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
extern int E;	      // total number of edges(?)
extern int K;	      // number of latent dimension
extern int T;	      // total number of timestamps
extern int start_T;   // starting time stamp
extern int ITER;      // total number of iterations
extern bool verbose;  // more outputs?

struct sub_graph {
  int n_users;
  map<int, int> u_map;		  // global_id -> local_id
  map<int, int> u_invert_map;	  // local_id -> global_id (only used for reverse lookup in final results)
  vector< vector<int> > graph;	  // graph
  vector< vector<double> > Sigma; // sigmoid(X_i, X_j) for all (i,j)'s
  vector< vector<double> > X;	  // X: n*K
  vector< vector<double*> > log_pt;	  // log_pt: n*K; joint p(r,s)
  vector< vector<double*> > log_pt_tik;	  // log_pt_tik: n*K; marginal p(r)
};

extern vector<struct sub_graph> G;	  // T * n * n

#endif


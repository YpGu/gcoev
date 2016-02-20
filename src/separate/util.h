#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <cstdlib>
#include <cmath>
#include <limits>

using namespace std;

extern int K;	      // number of latent dimension
extern int T;	      // total number of timestamps
extern int start_T;   // starting time stamp
extern int ITER;      // total number of iterations
extern bool verbose;  // do you want to see more outputs?

extern double stepsize;
extern double sigma;	  // variance
extern double lambda;	  // (1-lambda) * self + lambda * ave_neighbor

/* graph information for each time stamp */
struct sub_graph {
  int n_users;

  /* dictionary */
  map<int, int> u_map;		  // global_id -> local_id
  map<int, int> u_invert_map;	  // local_id -> global_id (used for reverse lookup)

  /* graph */
  vector< vector<int> > graph;	  // graph: n*n
  vector<int> encoded_all;	  // S+ and S-, encoded as i*n_users+j

  /* parameters */
  vector< vector<double> > X;	  // X: n*K
  vector< vector<double> > ave;	  // ave of user i's neighbors: n*K

  /* historic gradients (for adaptive stepsize) */
  vector< vector<double> > hgX;
};

extern vector<struct sub_graph> G;	      // T*1

#endif


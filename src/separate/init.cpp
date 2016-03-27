#include "init.h"

void init() {
  cout << "start init..." << endl;
  double scope = 1;
  for (int t = start_T; t < T; t++) {
    int n = G[t].n_users;

    // X
    G[t].X = vector< vector<double> >(n, vector<double>(K));
    for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
      double r = rand() / (double)RAND_MAX;
      G[t].X[i][k] = 2 * scope * (r-0.5);
//      if (r < 0.5) G[t].X[i][k] = 1;
//      else G[t].X[i][k] = -1;
    }

    // ave
    G[t].ave = vector< vector<double> >(n, vector<double>(K));
    for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
      G[t].ave[i][k] = 0;
    }
  }
  /* designed init
  for (int t = start_T; t < T; t++) {
    int n = G[t].n_users;
    G[t].X = vector< vector<int> >(n, vector<int>(K));
    for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
      double r = rand() / (double)RAND_MAX;
      if (i < 3) G[t].X[i][k] = 1;
      else G[t].X[i][k] = -1;
    }
  }
  */

  /* init alpha_s (prior) */
  alpha_s = vector<double>(T);
  for (int t = start_T; t < T; t++) {
    alpha_s[t] = 0.5;
  }

  cout << "init done" << endl;
}


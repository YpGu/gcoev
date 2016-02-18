#include "init.h"

void init() {
  cout << "start init..." << endl;
  for (int t = start_T; t < T; t++) {
    int n = G[t].n_users;

    // X
    G[t].X = vector< vector<double> >(n, vector<double>(K));
    for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
      double r = rand() / (double)RAND_MAX;
      if (r < 0.5) G[t].X[i][k] = 1;
      else G[t].X[i][k] = -1;
    }

    // Sigma
    G[t].Sigma = vector< vector<double> >(n, vector<double>(n));
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
      if (i != j) {
	vector<double> xi = G[t].X[i];
	vector<double> xj = G[t].X[j];
        G[t].Sigma[i][j] = sigmoid(xi, xj);
      }
    }

    // log_pt
    G[t].log_pt = vector< vector<double*> >(n, vector<double*>(K));
    for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
      G[t].log_pt[i][k] = new double[2*2];
      for (int sr = 0; sr < 2*2; sr++) G[t].log_pt[i][k][sr] = 0;
    }

    // log_pt_tik
    G[t].log_pt_tik = vector< vector<double*> >(n, vector<double*>(K));
    for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
      G[t].log_pt_tik[i][k] = new double[2];
      for (int sr = 0; sr < 2; sr++) G[t].log_pt_tik[i][k][sr] = 0;
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

  cout << "init done" << endl;
}


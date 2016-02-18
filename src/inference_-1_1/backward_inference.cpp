#include "backward_inference.h"

void backward() {
  cout << "\tbackward called" << endl;
  double* likelihood_t = new double[T];

  // B-step
  for (int k = 0; k < K; k++) {
    // t = T-1
    int n_T = G[T-1].n_users;
    for (int i = 0; i < n_T; i++) {
      double p0 = exp(G[T-1].log_pt_tik[i][k][0]);
      double random = rand() / (double)RAND_MAX;
cout << "i = " << i << ", t = " << T-1 << ", k = " << k << " p0 = " << p0 << endl;
      if (random < p0) { 
	G[T-1].X[i][k] = -1;
      } else {
	G[T-1].X[i][k] = 1;
      }
    }

    // t != T-1
    for (int t = T-2; t >= start_T; t--) {
      int n = G[t].n_users;
      for (int i = 0; i < n; i++) {
	double log_row_sum = log_sum_exp(G[t+1].log_pt[i][k], G[t+1].X[i][k], 2*2, 2);
	int cord = ((int)G[t+1].X[i][k] == 1) ? 1 : 0;
	double p0 = exp(G[t+1].log_pt[i][k][2*0+cord] - log_row_sum);
cout << "i = " << i << ", t = " << t << ", k = " << k << " p0 = " << p0 << endl;
	double random = rand() / (double)RAND_MAX;
	if (random < p0) {
	  G[t].X[i][k] = -1;
	} else {
	  G[t].X[i][k] = 1;
	}
      }
    }
  }

  // update Sigma
  for (int t = start_T; t < T; t++) {
    int n = G[t].n_users;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
      if (i != j) {
	vector<double> xi = G[t].X[i];
	vector<double> xj = G[t].X[j];
	G[t].Sigma[i][j] = sigmoid(xi, xj);
      }
    }
  }

}

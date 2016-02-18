#include "forward_inference.h"

void forward() {
  cout << "\tforward called" << endl;
  double* likelihood_t = new double[T];
  // calculate likelihood
  for (int t = start_T; t < T; t++) {
    likelihood_t[t] = compute_logl(t);
    cout << "t = " << t << ", likelihood = " << likelihood_t[t] << endl;
  }

  for (int k = 0; k < K; k++) {
    // t = start_T
    int n0 = G[start_T].n_users;
    for (int i = 0; i < n0; i++) {
      for (int r = 0; r < 2; r++) {
	// s = 0
	double ll_1 = likelihood_t[start_T];
	G[start_T].log_pt[i][k][2*r+0] = ll_1;
	// s = 1
	ll_1 = update_logl(start_T, i, k, 1, likelihood_t[start_T]);
	G[start_T].log_pt[i][k][2*r+1] = ll_1;
      }
      // normalize
      double log_prob_sum = log_sum_exp(G[start_T].log_pt[i][k], 0, 4, 1);
      for (int r = 0; r < 2; r++) for (int s = 0; s < 2; s++) G[start_T].log_pt[i][k][2*r+s] -= log_prob_sum;

      /* for tmp use */
//      for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {
//	cout << "a = " << a << ", b = " << b << ", log_pt = " << log_pt.at(start_T).at(i).at(k)[2*a+b] << ". ";
//      }

      // dynamic programming: update pi_tik (for next round)
      for (int s = 0; s < 2; s++) {
	double log_row_sum = log_sum_exp(G[start_T].log_pt[i][k], s, 2*1+s+1, 2);    // sum for each column
	G[start_T].log_pt_tik[i][k][s] = log_row_sum;
      }
    }

    // t > start_T
    for (int t = start_T + 1; t < T; t++) {
      cout << "\n t = " << t << endl;
      int n = G[t].n_users;
cout << n << endl;
      for (int i = 0; i < n; i++) {	// for each person
cout << i << ' ';
	for (int s = 0; s < 2; s++) {
	  double ll_1 = update_logl(t, i, k, s, likelihood_t[t]);
	  for (int r = 0; r < 2; r++) {
	    double log_pi = G[t-1].log_pt_tik[i][k][r];
	    double log_q = compute_logq(t, r, s, i, k);
	    G[t].log_pt[i][k][2*r+s] = log_pi + log_q + ll_1;
	  }
	}
	// normalize
	double log_prob_sum = log_sum_exp(G[t].log_pt[i][k], 0, 4, 1);
	for (int r = 0; r < 2; r++) for (int s = 0; s < 2; s++) G[t].log_pt[i][k][2*r+s] -= log_prob_sum;
	// dynamic programming: update pi_tik (for next round)
	for (int s = 0; s < 2; s++) {
	  double log_row_sum = log_sum_exp(G[t].log_pt[i][k], s, 2*1+s+1, 2);
	  G[t].log_pt_tik[i][k][s] = log_row_sum;
	}
cout << "done ";
      }
    }
  }

}

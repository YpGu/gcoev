#include "forward_inference.h"
#include "main.h"

void forward() {
  cout << "\tforward called" << endl;
  double* likelihood_t = new double[T];
  for (int t = 0; t < T; t++) for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) {
    log_pt.at(t).at(n).at(k) = new double[2*2];	      // joint p(r,s)
    log_pt_tik.at(t).at(n).at(k) = new double[2];     // marginal p(r)
  }
  // calculate likelihood
  for (int t = 0; t < T; t++) {
    likelihood_t[t] = compute_logl(t);
    cout << "likelihood = " << likelihood_t[t] << endl;
  }

  for (int k = 0; k < K; k++) {
    // t = 0
    vector<int> users_0 = users.at(0);
    for (vector<int>::iterator it = users_0.begin(); it != users_0.end(); it++) {   
      int i = *it;
      for (int r = 0; r < 2; r++) {
	// s = 0
	double ll_1 = likelihood_t[0];
	log_pt.at(0).at(i).at(k)[2*r+0] = ll_1;
	// s = 1
	ll_1 = update_logl(0, i, k, 1, likelihood_t[0]);
	log_pt.at(0).at(i).at(k)[2*r+1] = ll_1;
      }
      // normalize
      double log_prob_sum = log_sum_exp(log_pt.at(0).at(i).at(k), 0, 4, 1);
      for (int r = 0; r < 2; r++) for (int s = 0; s < 2; s++) log_pt.at(0).at(i).at(k)[2*r+s] -= log_prob_sum;

      /* for tmp use */
      for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {
//	cout << "a = " << a << ", b = " << b << ", log_pt = " << log_pt.at(0).at(i).at(k)[2*a+b] << ". ";
      }

      // dynamic programming: update pi_tik (for next round)
      for (int s = 0; s < 2; s++) {
	double log_row_sum = log_sum_exp(log_pt.at(0).at(i).at(k), s, 2*1+s+1, 2);    // sum for each column
	log_pt_tik.at(0).at(i).at(k)[s] = log_row_sum;
      }
    }

    // t != 0
    for (int t = 1; t < T; t++) {
      vector<int> users_t = users.at(t);
      for (vector<int>::iterator it = users_t.begin(); it != users_t.end(); it++) {   // for each person
	int i = *it;
	for (int s = 0; s < 2; s++) {
	  double ll_1 = update_logl(t, i, k, s, likelihood_t[t]);
	  for (int r = 0; r < 2; r++) {
	    double log_pi = log_pt_tik.at(t-1).at(i).at(k)[r];
	    double log_q = compute_logq(t, r, s, i, k);
	    log_pt.at(t).at(i).at(k)[2*r+s] = log_pi + log_q + ll_1;
	  }
	}
	// normalize
	double log_prob_sum = log_sum_exp(log_pt.at(t).at(i).at(k), 0, 4, 1);
	for (int r = 0; r < 2; r++) for (int s = 0; s < 2; s++) log_pt.at(t).at(i).at(k)[2*r+s] -= log_prob_sum;
	// dynamic programming: update pi_tik (for next round)
	for (int s = 0; s < 2; s++) {
	  double log_row_sum = log_sum_exp(log_pt.at(t).at(i).at(k), s, 2*1+s+1, 2);
	  log_pt_tik.at(t).at(i).at(k)[s] = log_row_sum;
	}
      }
    }
  }

}

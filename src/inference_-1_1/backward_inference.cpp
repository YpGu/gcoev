#include "backward_inference.h"

void backward() {
  cout << "\tbackward called" << endl;
  double* likelihood_t = new double[T];

  // B-step
  for (int k = 0; k < K; k++) {
    // t = T-1
    vector<int> users_T = users.at(T-1);
    for (vector<int>::iterator it = users_T.begin(); it != users_T.end(); it++) {
      int i = *it;
      double p0 = exp(log_pt_tik.at(T-1).at(i).at(k)[0]);
      double random = rand() / (double)RAND_MAX;
cout << "i = " << i << ", t = " << T-1 << ", k = " << k << " p0 = " << p0 << endl;
      if (random < p0) { 
	X.at(T-1).at(i)[k] = -1; 
      } else {
	X.at(T-1).at(i)[k] = 1;
      }
    }

    // t != T-1
    for (int t = T-2; t >= 0; t--) {
      vector<int> users_t = users.at(t);
      for (vector<int>::iterator it = users_t.begin(); it != users_t.end(); it++) {   // for each person
	int i = *it;
	double log_row_sum = log_sum_exp(log_pt.at(t+1).at(i).at(k), X.at(t+1).at(i).at(k), 2*2, 2);
	int cord = ((int)X.at(t+1).at(i).at(k) == 1) ? 1 : 0;
	double p0 = exp(log_pt.at(t+1).at(i).at(k)[2*0+cord] - log_row_sum);
cout << "i = " << i << ", t = " << t << ", k = " << k << " p0 = " << p0 << endl;
	double random = rand() / (double)RAND_MAX;
	if (random < p0) {
	  X.at(t).at(i)[k] = -1;
	} else {
	  X.at(t).at(i)[k] = 1;
	}
      }
    }
  }

  // update Sigma
  for (int t = 0; t < T; t++) for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
    if (i != j) 
      Sigma.at(t).at(i).at(j) = sigmoid(X.at(t).at(i), X.at(t).at(j));
  }

}

/* 
 * compute log likelihood
 */

#include "compute_logl.h"

double log_sigma(vector<double> arr1, vector<double> arr2) {
  double res = 0;
  for (int i = 0; i < arr1.size(); i++) res += arr1.at(i) * arr2.at(i);
  res = log(1/(1+exp(-res)) + numeric_limits<double>::min());
  if (res != res || res > 0) {
    cout << "log_sigma " << res << endl;
    int gu; cin >> gu;
  }
  return res;
}

double sigmoid(vector<double> arr1, vector<double> arr2) {
  double res = 0;
  for (int i = 0; i < arr1.size(); i++) res += arr1.at(i) * arr2.at(i);
  res = 1/(1+exp(-res)) + numeric_limits<double>::min();
  return res;
}

double sigmoid(double x) {
  return 1.0/(1+exp(-x));
}

/*
 * input: l_1, l_2, ...
 *    (l_i < 0, \forall i)
 * output: log ( exp(l_1) + exp(l_2) + ... )
 */
double log_sum_exp(double* arr, int start, int end, int jump = 1) {
  double res = 0.0;
  for (int i = start; i < end; i += jump) {
    res += exp(arr[i]);
  }
  res = log(res);
//  res = log(res + numeric_limits<double>::min());
  return res;
}

/*
 * Sigma: sigmoid (X_i \cdot X_j)
 */
double compute_logl(int t) {
  double res = 0;
  vector<int>::iterator it1, it2;
  vector<int> users_t = users.at(t);
  for (it1 = users_t.begin(); it1 != users_t.end(); it1++) {
    for (it2 = users_t.begin(); it2 != users_t.end(); it2++) {
      int i = *it1, j = *it2;
      if (i == j) continue;
      double prob = G.at(t).at(i).at(j) * log(Sigma.at(t).at(i).at(j) + numeric_limits<double>::min()) 
	  + (1-G.at(t).at(i).at(j)) * log(1 - Sigma.at(t).at(i).at(j) + numeric_limits<double>::min());
      res += prob;
      /*
      if (prob > 0) {
	cout << "prob = " << prob << endl;
	cout << "sigma = " << Sigma.at(t).at(i).at(j) << endl;
	cout << "G = " << G.at(t).at(i).at(j) << endl;
	int gu; cin >> gu;
      }
      if (res != res) {
	cout << "nan " << prob << endl;
	cout << "sigma = " << Sigma.at(t).at(i).at(j) << endl;
	int gu; cin >> gu;
      }
      */
    }
  }
  return res;
}

/** 
 * calculate the log likelihood when X[t][i][k] changes to xik
 * time complexity: O(N)
 */
double update_logl(int t, int i, int k, double xik, double logl) {
  double new_x = X.at(t).at(i).at(k), res = logl;
  vector<int>::iterator it1;
  if (new_x != xik) {
    vector<int> users_t = users.at(t);
    for (it1 = users_t.begin(); it1 != users_t.end(); it1++) {
      int j = *it1;
      if (j == i) continue;
      double s = 0;   // changed (new X)
      for (int kk = 0; kk < K; kk++) {
	if (kk != k) {
	  s += X.at(t).at(i)[kk] * X.at(t).at(j)[kk];
	} else {
	  s += xik * X.at(t).at(j)[kk];
	}
      }
      s = sigmoid(s);

      // j -> i
      res -= (G.at(t).at(j).at(i) * log(Sigma.at(t).at(j).at(i)) + (1-G.at(t).at(j).at(i)) * log(1-Sigma.at(t).at(j).at(i)));
      res += (G.at(t).at(j).at(i) * log(s + numeric_limits<double>::min()) 
	  + (1-G.at(t).at(j).at(i)) * log(1 - s + numeric_limits<double>::min()));
      // i -> j
      res -= (G.at(t).at(i).at(j) * log(Sigma.at(t).at(i).at(j)) + (1-G.at(t).at(i).at(j)) * log(1-Sigma.at(t).at(i).at(j)));
      res += (G.at(t).at(i).at(j) * log(s + numeric_limits<double>::min()) 
	  + (1-G.at(t).at(i).at(j)) * log(1 - s + numeric_limits<double>::min()));
    }
  }

  return res;
}

double compute_logq(int t, int r, int s, int i, int k) {
  double lambda = 0.7;
  double offset = 0;	    // TODO: specific w/ K; very important 
  double ave = 0; int num = 0;
  vector<int> users_t = users.at(t);
  for (vector<int>::iterator it = users_t.begin(); it != users_t.end(); it++) {
    int j = *it;
    if (G.at(t-1).at(i).at(j) == 1) {	// i -> j
      ave += X.at(t-1).at(j).at(k); num++;
    }
  }
  if (num != 0) ave /= num;
  double mu = (1-lambda) * r + lambda * ave;
  double pho = 1.0 / (1 + exp(-(mu - offset)));
  double res = -s * log(pho + numeric_limits<double>::min()) 
    - (1-s) * log(1 - pho + numeric_limits<double>::min());
  return res;
}
 


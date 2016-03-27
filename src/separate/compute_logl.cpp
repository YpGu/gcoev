/* 
 * compute log likelihood
 */

#include "compute_logl.h"

double log_sigma(vector<double> arr1, vector<double> arr2) {
  double res = 0;
  for (int i = 0; i < arr1.size(); i++) res += arr1[i] * arr2[i];
  res = log(1/(1+exp(-res)) + numeric_limits<double>::min());
  if (res != res || res > 0) {
    cout << "log_sigma " << res << endl;
    int gu; cin >> gu;
  }
  return res;
}

double sigmoid(vector<double> arr1, vector<double> arr2) {
  double res = 0;
  for (int i = 0; i < arr1.size(); i++) res += arr1[i] * arr2[i];
  res = 1/(1+exp(-res));
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
  double n_max = -numeric_limits<double>::max();
  int i_max = 0;
  for (int i = start; i < end; i += jump) {
    if (arr[i] > n_max) {
      n_max = arr[i];
      i_max = i;
    }
  }
  double res = 1.0;
  for (int i = start; i < end; i += jump) {
    if (i != i_max) 
      res += exp(arr[i] - n_max);
  }
  res = n_max + log(res);
  return res;
}

double compute_logl(int t) {
  double res = 0;
  /* all n*n users
  vector<int>::iterator it1, it2;
  int n = G[t].n_users;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      if (G[t].graph[i][j] > 0) {
	double prob = G[t].graph[i][j] * log(sigmoid(G[t].X[i], G[t].X[j]) + numeric_limits<double>::min());
	res += prob;
      } else {
	double prob = log(1 - sigmoid(G[t].X[i], G[t].X[j]) + numeric_limits<double>::min());
	res += prob;
      }
    }
  }
  return res;
  */
   
  int t_n = G[t].n_users;
  for (vector<int>::iterator it = G[t].encoded_all.begin(); it != G[t].encoded_all.end(); it++) {
    int e = *it; int i = e/t_n; int j = e%t_n;
    if (G[t].graph[i][j] > 0) {
      double ss = sigmoid(G[t].X[i], G[t].X[j]);
      double prob = G[t].graph[i][j] * log(ss + numeric_limits<double>::min());
      res += prob;
    } else {
      double ss = sigmoid(G[t].X[i], G[t].X[j]);
      double prob = log(1 - ss + numeric_limits<double>::min());
      res += prob;
    }
  }
  double res1 = res;
  for (int i = 0; i < t_n; i++) for (int k = 0; k < K; k++) {
    res -= 1/(2*delta*delta) * G[t].X[i][k] * G[t].X[i][k];
  }
  if (verbose) cout << "\tregularization = " << (res1-res) << endl;
  return res;
}

double compute_logl_tentative(int t, vector< vector<double> > values) {
  double res = 0;
  int t_n = G[t].n_users;
  for (vector<int>::iterator it = G[t].encoded_all.begin(); it != G[t].encoded_all.end(); it++) {
    int e = *it; int i = e/t_n; int j = e%t_n;
    if (G[t].graph[i][j] > 0) {
      double ss = sigmoid(values[i], values[j]);
      double prob = G[t].graph[i][j] * log(ss + numeric_limits<double>::min());
      res += prob;
    } else {
      double ss = sigmoid(values[i], values[j]);
      double prob = log(1 - ss + numeric_limits<double>::min());
      res += prob;
    }
  }
  double res1 = res;
  for (int i = 0; i < t_n; i++) for (int k = 0; k < K; k++) {
    res -= 1/(2*delta*delta) * values[i][k] * values[i][k];
  }
  if (verbose) cout << "\tregularization = " << (res1-res) << endl;
  return res;
}



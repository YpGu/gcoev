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
  int n = G[t].n_users;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      if (G[t].graph[i][j] == 0) {
	double prob = log(1 - G[t].Sigma[i][j] + numeric_limits<double>::min());
	res += prob;
      } else {
	double prob = G[t].graph[i][j] * log(G[t].Sigma[i][j] + numeric_limits<double>::min());
	res += prob;
      }
      /* check abnormality
      if (prob > 0) {
	cout << "prob = " << prob << endl;
	cout << "sigma = " << Sigma.at(t).at(i).at(j) << endl;
	cout << "G = " << G[t].graph[i][j] << endl;
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
  double new_x = G[t].X[i][k], res = logl;
  int n = G[t].n_users;
  if (new_x != xik) {
    for (int j = 0; j < n; j++) {
      if (j == i) continue;
      double s = 0;   // changed (new X)
      for (int kk = 0; kk < K; kk++) {
	if (kk != k) {
	  s += G[t].X[i][kk] * G[t].X[j][kk];
	} else {
	  s += xik * G[t].X[j][kk];
	}
      }
      s = sigmoid(s);

      // j -> i
      if (G[t].graph[j][i] == 0) {
	res -= log(1 - G[t].Sigma[j][i]);
	res += log(1 - s + numeric_limits<double>::min());
      } else {
	res -= G[t].graph[j][i] * log(G[t].Sigma[j][i]);
	res += G[t].graph[j][i] * log(s + numeric_limits<double>::min());
      }

      // i -> j
      if (G[t].graph[i][j] == 0) {
	res -= log(1 - G[t].Sigma[i][j]);
	res += log(1 - s + numeric_limits<double>::min());
      } else {
	res -= G[t].graph[i][j] * log(G[t].Sigma[i][j]);
	res += G[t].graph[i][j] * log(s + numeric_limits<double>::min()); 
      }

    }
  }

  return res;
}

double compute_logq(int t, int r, int s, int i, int k) {
  double lambda = 0.7;
  double offset = 0;	    // TODO: specific w/ K; very important 
  double ave = 0; int num = 0;
  int n = G[t].n_users;
  for (int j = 0; j < n; j++) {
    if (G[t-1].graph[i][j] != 0) {	// i -> j
      ave += G[t-1].X[j][k] * G[t-1].graph[i][j];
      num += G[t-1].graph[i][j];
    }
  }
  if (num != 0) ave /= num;
  double mu = (1-lambda) * r + lambda * ave;
  double pho = 1.0 / (1 + exp(-(mu - offset)));
  double res = -s * log(pho + numeric_limits<double>::min()) 
    - (1-s) * log(1 - pho + numeric_limits<double>::min());
  return res;
}
 


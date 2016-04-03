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


/* 
 * compute_logl:
 * compute the log-likelihood at given time t
 */
double compute_logl(int t) {
  double res = 0;
  int t_n = G[t].n_users;
 
  /* link part */
  for (vector<int>::iterator it = G[t].encoded_all.begin(); it != G[t].encoded_all.end(); it++) {
    int e = *it; int i = e/t_n; int j = e%t_n;
    double ss = sigmoid(G[t].X[i], G[t].X[j]);
    if (G[t].graph[i][j] > 0) {
      double prob = G[t].graph[i][j] * log(ss + numeric_limits<double>::min());
      res += prob;
    } else {
      double prob = log(1 - ss + numeric_limits<double>::min());
      res += prob;
    }
  }

  /* regularization (prior) part */
  double res1 = res;
  for (int i = 0; i < t_n; i++) if (G[t].has_predecessor[i]) {
    int old_i = G[t-1].u_map[G[t].u_invert_map[i]];
    for (int k = 0; k < K; k++) {
      double mu = (1-lambda) * G[t-1].X[old_i][k] + lambda * G[t-1].ave[old_i][k];
      res -= 1.0/(delta*delta) * (G[t].X[i][k] - mu) * (G[t].X[i][k] - mu);
    }
  }
  if (verbose) cout << "\tregularization = " << (res1-res) << endl;

  return res;
}


/* 
 * compute_logl_lower:
 * compute the lower bound of log-likelihood
 * used in M-step
 * input:
 *  t: time 
 *  v: the fuzzy assignment of clusters (probabilities)
 * output:
 *  the lower bound of log-likelihood, given (latent) v
 */
double compute_logl_lower(int t, vector<double> v) {
  double res = 0;
  int t_n = G[t].n_users;
 
  /* link part */
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

  /* regularization (prior) part */
  double res1 = res;
  for (int i = 0; i < t_n; i++) if (G[t].has_predecessor[i]) {
    int old_i = G[t-1].u_map[G[t].u_invert_map[i]];
    for (int k = 0; k < K; k++) {
      res -= 1.0/(delta*delta) * (
	  (1-v[i]) * (G[t].X[i][k] - G[t-1].X[old_i][k]) * (G[t].X[i][k] - G[t-1].X[old_i][k])
	  + v[i] * (G[t].X[i][k] - G[t-1].ave[old_i][k]) * (G[t].X[i][k] - G[t-1].ave[old_i][k]) );
    }
  }
  if (verbose) cout << "\tregularization = " << (res1-res) << endl;

  return res;
}


/*
 * Compute the log-likelihood, based on G[t] = values
 */
double compute_logl_tentative(int t, vector< vector<double> > values) {
  double res = 0;
  int t_n = G[t].n_users;

  /* link part */
  for (vector<int>::iterator it = G[t].encoded_all.begin(); it != G[t].encoded_all.end(); it++) {
    int e = *it; int i = e/t_n; int j = e%t_n;
    double ss = sigmoid(values[i], values[j]);
    if (G[t].graph[i][j] > 0) {
      double prob = G[t].graph[i][j] * log(ss + numeric_limits<double>::min());
      res += prob;
    } else {
      double prob = log(1 - ss + numeric_limits<double>::min());
      res += prob;
    }
  }
  /* regularization (prior) part */
  double res1 = res;
  for (int i = 0; i < t_n; i++) if (G[t].has_predecessor[i]) {
    int old_i = G[t-1].u_map[G[t].u_invert_map[i]];
    for (int k = 0; k < K; k++) {
      double mu = (1-lambda) * G[t-1].X[old_i][k] + lambda * G[t-1].ave[old_i][k];
      res -= 1.0/(delta*delta) * (values[i][k] - mu) * (values[i][k] - mu);
    }
  }
  if (verbose) cout << "\tregularization = " << (res1-res) << endl;

  return res;
}


/*
 * compute_logl_lower_tentative:
 * Compute the lower bound of log-likelihood,
 * given the cluster assignment estimated in E-step.
 * The objective function is based on G[t] = values.
 */
double compute_logl_lower_tentative(int t, vector< vector<double> > values, vector<double> v) {
  double res = 0;
  int t_n = G[t].n_users;

  /* link part */
  for (vector<int>::iterator it = G[t].encoded_all.begin(); it != G[t].encoded_all.end(); it++) {
    int e = *it; int i = e/t_n; int j = e%t_n;
    double ss = sigmoid(values[i], values[j]);
    if (G[t].graph[i][j] > 0) {
      double prob = G[t].graph[i][j] * log(ss + numeric_limits<double>::min());
      res += prob;
    } else {
      double prob = log(1 - ss + numeric_limits<double>::min());
      res += prob;
    }
  }
  /* regularization (prior) part */
  double res1 = res;
  for (int i = 0; i < t_n; i++) if (G[t].has_predecessor[i]) {
    int old_i = G[t-1].u_map[G[t].u_invert_map[i]];
    for (int k = 0; k < K; k++) {
      res -= 1.0/(delta*delta) * (
	  (1-v[i]) * (values[i][k] - G[t-1].X[old_i][k]) * (values[i][k] - G[t-1].X[old_i][k])
	  + v[i] * (values[i][k] - G[t-1].ave[old_i][k]) * (values[i][k] - G[t-1].ave[old_i][k]) );
    }
  }
  if (verbose) cout << "\tregularization = " << (res1-res) << endl;

  return res;
}



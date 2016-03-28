#include "train.h"

/**
 * train(): train the latent features for each time stamp t
 * t: current timestamp
 * stepsize: initial stepsize for SGD
 * delta: variance of Gaussian dist. 
 *    1/(2*delta*delta) is the regularization coefficient
 * lambda: tradeoff between self and neighbors (0 <= lambda <= 1)
 */
void train(int t, double stepsize, double delta, double lambda) {
  bool check_grad = false;
  for (int n_iter = 0; n_iter < M_ITER; n_iter++) {
    cout << "*** iteration " << n_iter << " ***" << endl;
    if (n_iter % 10 == 0) 
      check_grad = true;
    double grad_norm = 0;
    random_shuffle(G[t].encoded_all.begin(), G[t].encoded_all.end());
    int t_n = G[t].n_users;
    for (vector<int>::iterator it = G[t].encoded_all.begin(); it != G[t].encoded_all.end(); it++) {
      int e = *it; int i = e/t_n; int j = e%t_n;
      /* gradient ascent */
      if (G[t].graph[i][j] > 0) {     /* 1. positive link */
	double ss = sigmoid(G[t].X[i], G[t].X[j]);
	if (!G[t].has_predecessor[i]) {	  /* 1.1 for the first time */
	  for (int k = 0; k < K; k++) {
//	    double grad_xik = G[t].graph[i][j] * (1 - ss) * G[t].X[j][k] - 1.0/(delta*delta) * G[t].X[i][k];
//	    double grad_xjk = G[t].graph[i][j] * (1 - ss) * G[t].X[i][k] - 1.0/(delta*delta) * G[t].X[j][k]; 
  	    double grad_xik = G[t].graph[i][j] * (1 - ss) * G[t].X[j][k];
  	    double grad_xjk = G[t].graph[i][j] * (1 - ss) * G[t].X[i][k];
	    G[t].hgX[i][k] += grad_xik * grad_xik;
	    G[t].hgX[j][k] += grad_xjk * grad_xjk;
	    G[t].X[i][k] += grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	    G[t].X[j][k] += grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	    if (check_grad) {
	      double v1 = grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	      double v2 = grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	      grad_norm += (v1 * v1 + v2 * v2);
	    }
	  }
	} else {			  /* 1.2 not the first time */
	  int old_i = G[t-1].u_map[G[t].u_invert_map[i]];
	  int old_j = G[t-1].u_map[G[t].u_invert_map[j]];
	  for (int k = 0; k < K; k++) {
	    double grad_xik = G[t].graph[i][j] * (1 - ss) * G[t].X[j][k] 
	      - 1.0/(delta*delta) * (G[t].X[i][k] - (1-lambda) * G[t-1].X[old_i][k] - lambda * G[t-1].ave[old_i][k]);
	    double grad_xjk = G[t].graph[i][j] * (1 - ss) * G[t].X[i][k]
	      - 1.0/(delta*delta) * (G[t].X[j][k] - (1-lambda) * G[t-1].X[old_j][k] - lambda * G[t-1].ave[old_j][k]);
	    G[t].hgX[i][k] += grad_xik * grad_xik;
	    G[t].hgX[j][k] += grad_xjk * grad_xjk;
	    G[t].X[i][k] += grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	    G[t].X[j][k] += grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	    if (check_grad) {
	      double v1 = grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	      double v2 = grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	      grad_norm += (v1 * v1 + v2 * v2);
	    }
	  }
	}
      } else {			      /* 2. negative link */
	double ss = sigmoid(G[t].X[i], G[t].X[j]);
	if (!G[t].has_predecessor[i]) {	  /* 2.1 for the first time */
	  for (int k = 0; k < K; k++) {
//	    double grad_xik = -ss * G[t].X[j][k] - 1.0/(delta*delta) * G[t].X[i][k];
//	    double grad_xjk = -ss * G[t].X[i][k] - 1.0/(delta*delta) * G[t].X[j][k];
  	    double grad_xik = -ss * G[t].X[j][k];
  	    double grad_xjk = -ss * G[t].X[i][k];
	    G[t].hgX[i][k] += grad_xik * grad_xik;
	    G[t].hgX[j][k] += grad_xjk * grad_xjk;
	    G[t].X[i][k] += grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	    G[t].X[j][k] += grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	    if (check_grad) {
	      double v1 = grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	      double v2 = grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	      grad_norm += (v1 * v1 + v2 * v2);
	    }
	  }
	} else {			  /* 2.2 not the first time */
	  int old_i = G[t-1].u_map[G[t].u_invert_map[i]];
	  int old_j = G[t-1].u_map[G[t].u_invert_map[j]];
	  for (int k = 0; k < K; k++) {
	    double grad_xik = -ss * G[t].X[j][k] 
	      - 1.0/(delta*delta) * (G[t].X[i][k] - (1-lambda) * G[t-1].X[old_i][k] - lambda * G[t-1].ave[old_i][k]);
	    double grad_xjk = -ss * G[t].X[i][k]
	      - 1.0/(delta*delta) * (G[t].X[j][k] - (1-lambda) * G[t-1].X[old_j][k] - lambda * G[t-1].ave[old_j][k]);
	    G[t].hgX[i][k] += grad_xik * grad_xik;
	    G[t].hgX[j][k] += grad_xjk * grad_xjk;
	    G[t].X[i][k] += grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	    G[t].X[j][k] += grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	    if (check_grad) {
	      double v1 = grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	      double v2 = grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	      grad_norm += (v1 * v1 + v2 * v2);
	    }
	  }
	}
      }
    }
    /* next iteration of stochastic gradient ascent */
    if (n_iter % 10 == 0) {
      double llt = compute_logl(t);
      cout << "log likelihood at time " << t << " = " << llt << endl;
    }
    if (check_grad) {
      check_grad = false;
      if (verbose) cout << "l2 norm of gradient = " << grad_norm << endl;
    }
  }

  /*
  else {	 // not the first time 
    for (int n_iter = 0; n_iter < ITER; n_iter++) {
      cout << "*** iteration " << n_iter << " ***" << endl;
      if (n_iter % 10 == 0) 
	check_grad = true;
      double grad_norm = 0;
      random_shuffle(G[t].encoded_all.begin(), G[t].encoded_all.end());
      int t_n = G[t].n_users;
      for (vector<int>::iterator it = G[t].encoded_all.begin(); it != G[t].encoded_all.end(); it++) {
	int e = *it; int i = e/t_n; int j = e%t_n;
	int old_i = G[t-1].u_map[G[t].u_invert_map[i]];
	int old_j = G[t-1].u_map[G[t].u_invert_map[j]];
	// gradient ascent 
	if (G[t].graph[i][j] > 0) {
	  double ss = sigmoid(G[t].X[i], G[t].X[j]);
	  for (int k = 0; k < K; k++) {
	    double grad_xik = G[t].graph[i][j] * (1 - ss) * G[t].X[j][k] 
	      - 1.0/(delta*delta) * (G[t].X[i][k] - (1-lambda) * G[t-1].X[old_i][k] - lambda * G[t-1].ave[old_i][k]);
	    double grad_xjk = G[t].graph[i][j] * (1 - ss) * G[t].X[i][k]
	      - 1.0/(delta*delta) * (G[t].X[j][k] - (1-lambda) * G[t-1].X[old_j][k] - lambda * G[t-1].ave[old_j][k]);
	    G[t].hgX[i][k] += grad_xik * grad_xik;
	    G[t].hgX[j][k] += grad_xjk * grad_xjk;
	    G[t].X[i][k] += grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	    G[t].X[j][k] += grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	    if (check_grad) {
	      double v1 = grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	      double v2 = grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	      grad_norm += (v1 * v1 + v2 * v2);
	    }
	  }
	} else {
	  double ss = sigmoid(G[t].X[i], G[t].X[j]);
	  for (int k = 0; k < K; k++) {
	    double grad_xik = -ss * G[t].X[j][k] 
	      - 1.0/(delta*delta) * (G[t].X[i][k] - (1-lambda) * G[t-1].X[old_i][k] - lambda * G[t-1].ave[old_i][k]);
	    double grad_xjk = -ss * G[t].X[i][k]
	      - 1.0/(delta*delta) * (G[t].X[j][k] - (1-lambda) * G[t-1].X[old_j][k] - lambda * G[t-1].ave[old_j][k]);
	    G[t].hgX[i][k] += grad_xik * grad_xik;
	    G[t].hgX[j][k] += grad_xjk * grad_xjk;
	    G[t].X[i][k] += grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	    G[t].X[j][k] += grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	    if (check_grad) {
	      double v1 = grad_xik * stepsize / sqrt(G[t].hgX[i][k]);
	      double v2 = grad_xjk * stepsize / sqrt(G[t].hgX[j][k]);
	      grad_norm += (v1 * v1 + v2 * v2);
	    }
	  }
	}
      }
      // next iteration of stochastic gradient ascent 
      if (n_iter % 10 == 0) {
	double llt = compute_logl(t);
	cout << "log likelihood at time " << t << " = " << llt << endl;
      }
      if (check_grad) {
	check_grad = false;
	if (verbose) cout << "l2 norm of gradient = " << grad_norm << endl;
      }
    }
  }
  */

  /* update ave[i][k] for graph at time t */
  vector<int> degree = vector<int>(G[t].n_users);
  for (int i = 0; i < G[t].n_users; i++) {
    for (int k = 0; k < K; k++) {
      G[t].ave[i][k] = 0;
    }
    degree[i] = 0;
  }
  for (int i = 0; i < G[t].n_users; i++) {
    for (int j = 0; j < G[t].n_users; j++) {
      if (G[t].graph[i][j] > 0) for (int k = 0; k < K; k++) {
	G[t].ave[i][k] += G[t].X[j][k];
	G[t].ave[j][k] += G[t].X[i][k];
	degree[i] += 1; degree[j] += 1;
      }
    }
  }
  for (int i = 0; i < G[t].n_users; i++) {
    if (degree[i] != 0) for (int k = 0; k < K; k++) {
      G[t].ave[i][k] /= (double)degree[i];
    }
  }

  /* free */
  vector<int>().swap(degree);
  degree.clear();

}


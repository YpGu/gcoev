#include "train_gd.h"

/**
 * train(): train the latent features for each time stamp t
 * t: current timestamp
 * stepsize: initial stepsize for GD
 * delta: variance of Gaussian dist. 
 *    1/(2*delta*delta) is the regularization coefficient
 * lambda: tradeoff between self and neighbors (0 <= lambda <= 1)
 */
void train_gd(int t, double stepsize, double delta, double lambda) {
  bool check_grad = false;
  double old_obj = compute_logl(t), new_obj;
  for (int n_iter = 0; n_iter < M_ITER; n_iter++) {
    if (verbose) cout << "\n*** iteration " << n_iter << " ***" << endl;
    int t_n = G[t].n_users;
    vector< vector<double> > grad = vector< vector<double> >(t_n, vector<double>(K));
    if (n_iter % 1 == 0) check_grad = true;
    double grad_norm = 0;
    /* iterate all links */
    for (vector<int>::iterator it = G[t].encoded_all.begin(); it != G[t].encoded_all.end(); it++) {
      int e = *it; int i = e/t_n; int j = e%t_n;
      double ss = sigmoid(G[t].X[i], G[t].X[j]);
      /* gradient ascent */
      if (G[t].graph[i][j] > 0) {     /* 1. positive link */
	for (int k = 0; k < K; k++) {
	  double grad_xik = G[t].graph[i][j] * (1 - ss) * G[t].X[j][k];
	  double grad_xjk = G[t].graph[i][j] * (1 - ss) * G[t].X[i][k];
	  grad[i][k] += grad_xik;
	  grad[j][k] += grad_xjk;
	}
      } else {			      /* 2. negative link */
	for (int k = 0; k < K; k++) {
	  double grad_xik = -ss * G[t].X[j][k];
	  double grad_xjk = -ss * G[t].X[i][k];
	  grad[i][k] += grad_xik;
	  grad[j][k] += grad_xjk;
	}
      }
    }

    /* iterate all users (regularization) */
    for (int i = 0; i < t_n; i++) if (G[t].has_predecessor[i]) {
      int old_i = G[t-1].u_map[G[t].u_invert_map[i]];
      for (int k = 0; k < K; k++) {
	double grad_xik = - 1.0/(delta*delta) * (G[t].X[i][k] - (1-lambda) * G[t-1].X[old_i][k] - lambda * G[t-1].ave[old_i][k]);
	grad[i][k] += grad_xik;
      }
    }

    /* check gradient */
    if (check_grad) for (int i = 0; i < G[t].n_users; i++) for (int k = 0; k < K; k++) {
      grad_norm += grad[i][k] * grad[i][k];
    }

    /* line search */
    double cur_stepsize = stepsize;
    double com_obj = old_obj;
    for (int iter = 0; iter < 20; iter++) {
      vector< vector<double> > tmp_value = vector< vector<double> >(t_n, vector<double>(K));
      for (int i = 0; i < t_n; i++) for (int k = 0; k < K; k++) {
	tmp_value[i][k] = G[t].X[i][k] + cur_stepsize * grad[i][k];
      }
      double llt = compute_logl_tentative(t, tmp_value);
      if (verbose) cout << "line search: objective = " << llt << endl;
      if (llt > old_obj) {
	old_obj = llt;
	/* update parameters */
	for (int i = 0; i < t_n; i++) for (int k = 0; k < K; k++) {
	  G[t].X[i][k] = tmp_value[i][k];
	}
	vector< vector<double> >().swap(tmp_value);
	tmp_value.clear();
	break;
      }
      vector< vector<double> >().swap(tmp_value);
      tmp_value.clear();
      cur_stepsize *= 0.5;
    }

    /* next iteration of stochastic gradient ascent */
    if (n_iter % 1 == 0) {
      new_obj = compute_logl(t);
      cout << "log likelihood at time " << t << " (iter " << n_iter << ") = " << new_obj << endl;
    }
    if (check_grad) {
      check_grad = false;
      output_2d("./tmp/grad1", grad, G[t].n_users, K);
      if (verbose) cout << "l2 norm of gradient = " << sqrt(grad_norm) << endl;
    }

    /* free gradients */
    vector< vector<double> >().swap(grad);
    grad.clear();

    if (fabs((old_obj - com_obj) / com_obj) < 1e-6 && n_iter > 3) 
      break;
  }

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

  likel[t] = new_obj;

}


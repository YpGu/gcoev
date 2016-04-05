#include "train_gd.h"


/**
 * init_gd:
 *  in order to avoid the flipping issue, the latent features at t
 *  are initialized to be related to those at t-1
 */
void init_gd(int t) {
  int t_n = G[t].n_users;
  for (int i = 0; i < t_n; i++) {
    if (G[t].has_predecessor[i]) {  /* not the first time */
      int old_i = G[t-1].u_map[G[t].u_invert_map[i]];
      for (int k = 0; k < K; k++) {
	G[t].X[i][k] = 0.5 * G[t-1].X[old_i][k] + 0.5 * G[t-1].ave[old_i][k];
      }
    }
  }

  return;
}


/**
 * update_param_gd:
 *  update model parameter using line search,
 *  based on the gradient computed in the previous step
 * returns the objective after update
 */
double update_param_gd(int t, vector< vector<double> > grad, vector<double> v) {
  int t_n = G[t].n_users;
  double cur_stepsize = stepsize;
  double old_obj = compute_logl_lower(t, v);
  for (int iter = 0; iter < 20; iter++) {
    vector< vector<double> > tmp_value = vector< vector<double> >(t_n, vector<double>(K));
    for (int i = 0; i < t_n; i++) for (int k = 0; k < K; k++) {
      tmp_value[i][k] = G[t].X[i][k] + cur_stepsize * grad[i][k];
    }
    double llt = compute_logl_lower_tentative(t, tmp_value, v);
    if (verbose) cout << "\tline search: objective = " << llt << endl;
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

  return old_obj;
}


/** 
 * compute_grad_gd:
 *  given the latent variable estimated from the E-step,
 *  compute the gradients w.r.t. model parameters 
 */
vector< vector<double> > compute_grad_gd(int t, double lambda) {
  int t_n = G[t].n_users;
  vector< vector<double> > grad = vector< vector<double> >(t_n, vector<double>(K));

  /* iterate all links */
  for (vector<int>::iterator it = G[t].encoded_all.begin(); it != G[t].encoded_all.end(); it++) {
    int e = *it; int i = e/t_n; int j = e%t_n;
    /* gradient ascent */
    double ss = sigmoid(G[t].X[i], G[t].X[j]);
    if (G[t].graph[i][j] > 0) {     /* 1. positive link */
      for (int k = 0; k < K; k++) {
	double grad_xik = G[t].graph[i][j] * (1 - ss) * G[t].X[j][k];
	double grad_xjk = G[t].graph[i][j] * (1 - ss) * G[t].X[i][k];
	grad[i][k] += grad_xik;
	grad[j][k] += grad_xjk;
      }
    } else {			    /* 2. negative link */
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
      double grad_xik = -1.0/(delta*delta) * ( 
	  (1-lambda) * (G[t].X[i][k] - G[t-1].X[old_i][k])
	  + lambda* (G[t].X[i][k] - G[t-1].ave[old_i][k]) );
      grad[i][k] += grad_xik;
    }
  }

  return grad;
}


/**
 * train_gd: train the latent features for each time stamp t
 * t: current timestamp
 * stepsize: initial stepsize for GD
 * delta: variance of Gaussian dist. 
 *    1/(2*delta*delta) is the regularization coefficient
 * lambda: tradeoff between self and neighbors (0 <= lambda <= 1)
 */
void train_gd(int t, double stepsize, double delta, double lambda) {
  bool check_grad = false;
  double new_obj = -1, old_obj = -1;

  for (int n_iter = 0; n_iter < M_ITER; n_iter++) {
    if (verbose) cout << "\t*** iteration " << n_iter << " ***" << endl;

    /* calculate grad */
    vector< vector<double> > grad = compute_grad_gd(t, lambda);

    /* update parameters */
    new_obj = update_param_gd(t, grad, v[t]);
    if (n_iter % 10 == 0) 
      cout << "\tlog likelihood at time " << t << " (iter " << n_iter << ") = " << new_obj << endl;

    /* check gradient */
    if (check_grad) {
      double grad_norm = 0;
      for (int i = 0; i < G[t].n_users; i++) for (int k = 0; k < K; k++) {
	grad_norm += grad[i][k] * grad[i][k];
      }
      check_grad = false;
      output_2d("./tmp/grad1", grad, G[t].n_users, K);
      if (verbose) cout << "l2 norm of gradient = " << sqrt(grad_norm) << endl;
    }

    /* free gradients */
    vector< vector<double> >().swap(grad);
    grad.clear();

    if (fabs((new_obj - old_obj) / old_obj) < 1e-6 && n_iter > 1) 
      break;
    old_obj = new_obj;
  }

  likel[t] = new_obj;

  return;
}



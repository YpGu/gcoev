#include "train_gd.h"


/* E-step */
void e_step(int t) {
  int t_n = G[t].n_users;
  double alpha1 = 0, alpha2 = 0;
  double alpha_t = alpha_s[t];
  for (int i = 0; i < t_n; i++) {
    if (G[t].has_predecessor[i]) {  /* not the first time */
      int old_i = G[t-1].u_map[G[t].u_invert_map[i]];

      double ip1 = 0, ip2 = 0;
      for (int k = 0; k < K; k++) ip1 += (G[t].X[i][k] - G[t-1].X[old_i][k]) * (G[t].X[i][k] - G[t-1].X[old_i][k]);
      double v1 = (1-alpha_t) * exp(-ip1 / (2*delta*delta));
      for (int k = 0; k < K; k++) ip2 += (G[t].X[i][k] - G[t-1].ave[old_i][k]) * (G[t].X[i][k] - G[t-1].ave[old_i][k]);
      double v2 = alpha_t * exp(-ip2 / (2*delta*delta));
      double norm = v1 + v2;
      if (norm != 0) { v1 /= norm; v2 /= norm; }

      v[t][i] = v2;
      alpha1 += v1; alpha2 += v2;
    } else {
      v[t][i] = 0.5;
    }
  }
  if (alpha2 != 0) {
    alpha_s[t] = alpha2 / (alpha1 + alpha2);
  } else if (alpha1 != 0) {
    alpha_s[t] = 0;
  } else {
    alpha_s[t] = 0.5;
  }

}


/**
 * update_param: 
 *  update model parameter using line search,
 *  based on the gradient computed in the previous step
 * returns the objective after update
 */
double update_param(int t, vector< vector<double> > grad, vector<double> v) {
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
 * compute_grad: 
 *  given the latent variable estimated from the E-step,
 *  compute the gradients w.r.t. model parameters 
 */
vector< vector<double> > compute_grad(int t) {
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
	  (1-v[t][i]) * (G[t].X[i][k] - G[t-1].X[old_i][k])
	  + v[t][i] * (G[t].X[i][k] - G[t-1].ave[old_i][k]) );
      grad[i][k] += grad_xik;
    }
  }

  return grad;
}

/*
 * flip:
 *  try flipping the sign of model parameters;
 *  flip the sign if a better objective can be achieved
 */
void flip(int t, vector< vector<double> > v) {
  int t_n = G[t].n_users;
  vector< vector<double> > tmp_value = vector< vector<double> >(t_n, vector<double>(K));
  for (int i = 0; i < t_n; i++) for (int k = 0; k < K; k++) {
    tmp_value[i][k] = -G[t].X[i][k];
  }
  double obj1 = compute_logl_lower(t, v[t]);
  double obj2 = compute_logl_lower_tentative(t, tmp_value, v[t]);
  if (obj2 > obj1) {
    for (int i = 0; i < t_n; i++) for (int k = 0; k < K; k++) {
      G[t].X[i][k] = -G[t].X[i][k];
    }
    cout << "fliped!" << endl;
  }
  vector< vector<double> >().swap(tmp_value);
  tmp_value.clear();

  return;
}


/**
 * train_em: update the latent features for each time stamp t
 * t: current timestamp
 * stepsize: initial stepsize for GD
 * delta: variance of Gaussian dist. 
 *    1/(2*delta*delta) is the regularization coefficient
 */
void train_em(int t, double stepsize, double delta) {
  bool check_grad = false;
  double new_obj = -1, old_obj = -1, com_obj = -1;

  for (int o_iter = 0; o_iter < TOT_ITER; o_iter++) {
    if (verbose) cout << "\n*** outer iteration " << o_iter << " ***" << endl;

    if (o_iter == 1) {
      flip(t, v);
    }

    /* E-step */
    e_step(t);
    cout << "\talpha = " << alpha_s[t] << endl;

    /* M-step */
    for (int n_iter = 0; n_iter < M_ITER; n_iter++) {
      if (verbose) cout << "\t*** iteration " << n_iter << " ***" << endl;

      /* calculate grad */
      vector< vector<double> > grad = compute_grad(t);

      /* update parameters */
      new_obj = update_param(t, grad, v[t]);
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

      if (fabs((new_obj - com_obj) / com_obj) < 1e-6 && n_iter > 1) 
        break;
      com_obj = new_obj;
    }

    if (fabs((old_obj - new_obj) / old_obj) < 1e-6 && o_iter > 1) 
      break;
    old_obj = new_obj;
  }
  cout << "final t = " << t << ", alpha = " << alpha_s[t] << endl;
  likel[t] = new_obj;
//  int gu; cin >> gu;

}


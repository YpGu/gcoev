#include "init.h"

void init() {
  X = vector< vector< vector<double> > >(T, vector< vector<double> >(N, vector<double>(K)));
  cout << "start init..." << endl;
  for (int t = 0; t < T; t++) for (int i = 0; i < N; i++) for (int k = 0; k < K; k++) {
    double r = rand() / (double)RAND_MAX;
    if (r < 0.5) X.at(t).at(i).at(k) = 1;
    else X.at(t).at(i).at(k) = -1;
  }
//  /* designed init
  for (int t = 0; t < T; t++) for (int i = 0; i < N; i++) for (int k = 0; k < K; k++) {
    if (i < 3) X.at(t).at(i).at(k) = 1;
    else X.at(t).at(i).at(k) = -1;
  }
//  */
  cout << "init done" << endl;

  Sigma = vector< vector< vector<double> > >(T, vector< vector<double> >(N, vector<double>(N)));
  for (int t = 0; t < T; t++) for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
    if (i != j) 
      Sigma.at(t).at(i).at(j) = sigmoid(X.at(t).at(i), X.at(t).at(j));
  }

  log_pt = vector< vector< vector<double*> > >(T, vector< vector<double*> >(N, vector<double*>(K)));
  for (int t = 0; t < T; t++) for (int i = 0; i < N; i++) for (int k = 0; k < K; k++) {
    log_pt.at(t).at(i).at(k) = new double[2*2];
    for (int sr = 0; sr < 2*2; sr++) log_pt.at(t).at(i).at(k)[sr] = 0;
  }
  log_pt_tik = vector< vector< vector<double*> > >(T, vector< vector<double*> >(N, vector<double*>(K)));
  for (int t = 0; t < T; t++) for (int i = 0; i < N; i++) for (int k = 0; k < K; k++) {
    log_pt_tik.at(t).at(i).at(k) = new double[2];
    for (int sr = 0; sr < 2; sr++) log_pt_tik.at(t).at(i).at(k)[sr] = 0;
  }
}


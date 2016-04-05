#include "init.h"


/** config:
 *  get configurations for the model
 */
bool config(int argc, char* argv[]) {
  if (argc != 5) {
    cout << "Usage: ./a.out <option> <delta_t> <lambda> <output_dir>" << endl;
    cout << "Example: ./a.out 1 10 0.5 ./save/lfpm/0.5/" << endl;
    cout << "\toption = 1: no em" << endl;
    cout << "\toption = 2: with em" << endl;
    cout << "\tdelta_t: variance" << endl;
    cout << "\tlambda: tradeoff parameter between self and neighbor" << endl;
    cout << endl;
    return false;
  }
  option = atoi(argv[1]);
  if (option != 1 && option != 2) {
    cout << "Option must be either 1 (no em) or 2 (with em)!\n" << endl;
    return false;
  }
  delta = atof(argv[2]);
  lambda = atof(argv[3]);
  if (lambda > 1 || lambda < 0) {
    cout << "lambda must be between 0 and 1!\n" << endl;
    return false;
  }
  outdir = argv[4];
  if (!isDirectory(outdir)) {
    cout << "Directory does not exist!\n" << endl;
    return false;
  }
  cout << "Save result under directory " << outdir << endl;
  
  return true;
}


/** init:
 *  initialize parameters
 */
void init() {
  cout << "start init..." << endl;

  srand(0);
//  srand(time(NULL));

  v = vector< vector<double> >(T, vector<double>());
  double scope = 1;
  for (int t = start_T; t < T; t++) {
    int n = G[t].n_users;
    v[t] = vector<double>(n);

    // X
    G[t].X = vector< vector<double> >(n, vector<double>(K));
    for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
      double r = rand() / (double)RAND_MAX;
      G[t].X[i][k] = 2 * scope * (r-0.5);
    }

    // ave
    G[t].ave = vector< vector<double> >(n, vector<double>(K));
    for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
      G[t].ave[i][k] = 0;
    }
  }
  /* designed init
  for (int t = start_T; t < T; t++) {
    int n = G[t].n_users;
    G[t].X = vector< vector<int> >(n, vector<int>(K));
    for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
      double r = rand() / (double)RAND_MAX;
      if (i < 3) G[t].X[i][k] = 1;
      else G[t].X[i][k] = -1;
    }
  }
  */

  /* init alpha_s (prior) */
  alpha_s = vector<double>(T);
  for (int t = start_T; t < T; t++) {
    alpha_s[t] = 0.5;
  }
  
  /* init likelihood */
  likel = vector<double>(T);
  for (int t = start_T; t < T; t++) {
    likel[t] = 0;
  }

  cout << "init done" << endl;
}


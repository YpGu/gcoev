#include "main.h"
#include "util.h"
#include "init.h"
#include "file_parser.h"
#include "train.h"
#include "train_gd.h"
#include "train_em.h"

int main() {

  srand(0);
//  srand(time(NULL));

  /* init N, K, T */
  read_csv_graph(&"../../data/graph/"[0u]);
  init();

  /* run */
  for (int t = start_T; t < T; t++) {
    cout << "t = " << t << endl;
//    train(t, stepsize, delta, lambda);
//    train_gd(t, stepsize, delta, lambda);
    train_em(t, stepsize, delta);
  }
  cout << "training done!" << endl;
  output_hidden(&"./save/"[0u]);

}



#include "main.h"
#include "util.h"
#include "init.h"
#include "file_parser.h"
#include "train.h"
#include "train_gd.h"

int main() {

  // init N, K, T
  read_csv_graph(&"../../data/graph/"[0u]);
  init();

  // run
  for (int t = start_T; t < T; t++) {
    cout << "t = " << t << endl;
//    train(t, stepsize, sigma, lambda);
    train_gd(t, stepsize, sigma, lambda);
  }
  cout << "training done!" << endl;
  output_hidden(&"./save/"[0u]);

}



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
  read_csv_graph(&"../../data/jsim_graph_selected/"[0u]);
  init();
//  int option = 1;   // no em
  int option = 2;   // em

  /* train */
  if (option == 1) {
    for (int t = start_T; t < T; t++) {
      cout << "t = " << t << endl;
      train_gd(t, stepsize, delta, lambda);
    }
  }
  else if (option == 2) {
    for (int t = start_T; t < T; t++) {
      cout << "t = " << t << endl;
      train_em(t, stepsize, delta);
    }
  }
  cout << "training done!" << endl;

  /* save */
  if (option == 1) {
    output_hidden(&"../../data/dict/user_id_map.dat"[0u], &"./save/baseline_0/"[0u]);
    output_1d(&"./save/baseline_0/likelihood.txt"[0u], likel, T);
  }
  else if (option == 2) {
    output_hidden(&"../../data/dict/user_id_map.dat"[0u], &"./save/lfpm/"[0u]);
    output_1d(&"./save/lfpm/alpha.txt"[0u], alpha_s, T);
    output_1d(&"./save/lfpm/likelihood.txt"[0u], likel, T);
  }

}



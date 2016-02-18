#include "main.h"
#include "util.h"
#include "init.h"
#include "file_parser.h"
#include "forward_inference.h"
#include "backward_inference.h"
#include "compute_logl.h"

int main() {

  // init N, K, T
//  read_csv_graph(&"./test/"[0u]);
  read_csv_graph(&"../../data/graph/"[0u]);
  init();
  // check X
  /*
  for (int t = start_T; t < T; t++) {
    cout << "---------------------" << endl;
    int n = G[t].n_users;
    for (int i = 0; i < n; i++) {
      cout << "X(" << t << ")(" << i << "): ";
      for (int k = 0; k < K; k++) {
	cout << G[t].X[i][k] << " ";
      }
      cout << endl;
    }
    cout << "---------------------" << endl;
  }
  */

  // run
  for (int iter = 0; iter < ITER; iter++) {
    cout << "Iteration " << iter << endl;
    // F-step
    forward();

    // B-step
    backward();

    // output
    stringstream ss; ss << iter;
    output_hidden(&("./save/" + ss.str() + "/")[0u]);

    /*
    // check X
    for (int t = start_T; t < T; t++) {
      cout << "---------------------" << endl;
      int n = G[t].n_users;
      for (int i = 0; i < n; i++) {
	int old_id = G[t].u_invert_map[i];
	cout << "X(" << t << ")(" << old_id << "): ";
	for (int k = 0; k < K; k++) {
	  cout << G[t].X[i][k] << " ";
	}
	cout << endl;
      }
      cout << "---------------------" << endl;
    }
    */
  }

}



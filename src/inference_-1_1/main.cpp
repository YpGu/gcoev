#include "main.h"
#include "util.h"
#include "init.h"
#include "file_parser.h"
#include "forward_inference.h"
#include "backward_inference.h"
#include "compute_logl.h"

int main() {
  // init N, K, T
  read_csv_graph(&"./test/"[0u]);
  init();
  // check X
  for (int t = 0; t < T; t++) {
    cout << "---------------------" << endl;
    for (int i = 0; i < N; i++) {
      cout << "X(" << t << ")(" << i << "): ";
      for (int k = 0; k < K; k++) {
	cout << X.at(t).at(i).at(k) << " ";
      }
      cout << endl;
    }
    cout << "---------------------" << endl;
  }

  // run
  for (int iter = 0; iter < ITER; iter++) {
    cout << "Iteration " << iter << endl;
    // F-step
    forward();

    // B-step
    backward();

    // check X
    for (int t = 0; t < T; t++) {
      cout << "---------------------" << endl;
      for (int i = 0; i < N; i++) {
	cout << "X(" << t << ")(" << i << "): ";
	for (int k = 0; k < K; k++) {
	  cout << X.at(t).at(i).at(k) << " ";
	}
	cout << endl;
      }
      cout << "---------------------" << endl;
    }

  }
}

#include <iostream>
#include <vector>
#include <cmath>
#include "compute_logl.h"
#include "file_parser.h"
#include "util.h"

using namespace std;

void train_gd(int t, double stepsize, double delta, double lambda);
void init_gd(int t);


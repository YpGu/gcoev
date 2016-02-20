/* 
 * compute log likelihood
 */

#include "util.h"

double log_sigma(vector<double> arr1, vector<double> arr2);
double sigmoid(vector<double> arr1, vector<double> arr2);
double compute_logl(int time);
double log_sum_exp(double* arr, int start, int end, int jump);
 

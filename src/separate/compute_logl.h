/* 
 * compute log likelihood
 */

#include "util.h"

double log_sigma(vector<double> arr1, vector<double> arr2);
double sigmoid(vector<double> arr1, vector<double> arr2);
double compute_logl(int time);
double compute_logl_lower(int t, vector<double> v);
double compute_logl_tentative(int t, vector< vector<double> > values);
double compute_logl_lower_tentative(int t, vector< vector<double> > values, vector<double> v);
double log_sum_exp(double* arr, int start, int end, int jump);
 

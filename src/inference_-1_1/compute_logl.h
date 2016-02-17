/* 
 * compute log likelihood
 */

#include "util.h"

double log_sigma(vector<double> arr1, vector<double> arr2);
double sigmoid(vector<double> arr1, vector<double> arr2);
//double compute_logl(vector<vector<int> > G, vector<vector<double> > X, vector<vector<double> > Sigma, vector<int> users, int time);
double compute_logl(int time);
//double update_logl(vector<vector<int> > G, vector<vector<double> > X, vector<vector<double> > Sigma, vector<int> users, int time, 
//    int i, int k, double xik, double logl);
double update_logl(int time, int i, int k, double xik, double logl);
//double compute_logq(vector<vector<int> > G, vector<vector<double> > X, vector<vector<double> > Sigma, vector<int> users, 
//    int time, int r, int s, int i, int k);
double compute_logq(int time, int r, int s, int i, int k);
double log_sum_exp(double* arr, int start, int end, int jump);
 

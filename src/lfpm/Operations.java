import java.util.*;
import Jama.*;

public class Operations {
  /* 
   * translate
   * copy array/matrix according to the id of time <t>
   * option: 
   *	true if use random prior
   *	false if use zero prior
   */
  public static Matrix 
  translate(Matrix arr1, Map<Integer, Integer> map1, Map<Integer, Integer> map2, boolean option) {
    int n2 = map2.size();
    if (arr1.getColumnDimension() == arr1.getRowDimension()) {
      /* arr1 is a square matrix */
      double[][] arr2 = new double[n2][n2];
      if (option) {
	/* random prior */ // TODO: what about covariance matrix?
	for (int i = 0; i < n2; i++) for (int j = 0; j < n2; j++) {
	  arr2[i][j] = Main.scale * (Main.rand.nextDouble() - 0.5);
	}
      }
      /* override some elements */
      for (Map.Entry<Integer, Integer> e: map1.entrySet()) {
	int global_id_x = e.getKey();
	if (map2.containsKey(global_id_x)) {
	  int local_id2_x = map2.get(global_id_x);
	  int local_id1_x = map1.get(global_id_x);
	  for (Map.Entry<Integer, Integer> f: map1.entrySet()) {
	    int global_id_y = f.getKey();
	    if (map2.containsKey(global_id_y)) {
	      int local_id2_y = map2.get(global_id_y);
	      int local_id1_y = map1.get(global_id_y);
	      arr2[local_id2_x][local_id2_y] = arr1.get(local_id1_x, local_id1_y);
	    }
	  }
	}
      }
      return new Matrix(arr2);
    } else {
      /* arr1 is an array */
      double[][] arr2 = new double[n2][1];
      if (option) {
	/* random prior */
	for (int i = 0; i < n2; i++) {
	  arr2[i][0] = Main.scale * (Main.rand.nextDouble() - 0.5);
	}
      }
      /* override some elements */
      for (Map.Entry<Integer, Integer> e: map1.entrySet()) {
	int global_id = e.getKey();
	if (map2.containsKey(global_id)) {
	  int local_id2 = map2.get(global_id);
	  int local_id1 = map1.get(global_id);
	  arr2[local_id2][0] = arr1.get(local_id1, 0);
	}
      }
      return new Matrix(arr2);
    }
  }


  public static double[]
  translate(double[] arr1, Map<Integer, Integer> map1, Map<Integer, Integer> map2) {
    /* arr1 is an array */
    int n2 = map2.size();
    double[] arr2 = new double[n2];

    /* override some elements */
    for (Map.Entry<Integer, Integer> e: map1.entrySet()) {
      int global_id = e.getKey();
      if (map2.containsKey(global_id)) {
	int local_id2 = map2.get(global_id);
	int local_id1 = map1.get(global_id);
	arr2[local_id2] = arr1[local_id1];
      }
    }
    return arr2;
  }

  public static double[][]
  translate(double[][] mat1, Map<Integer, Integer> map1, Map<Integer, Integer> map2) {
    int n2 = map2.size();

    if (mat1.length == mat1[0].length) {
      double[][] mat2 = new double[n2][n2];
      /* override some elements */
      for (Map.Entry<Integer, Integer> e: map1.entrySet()) {
	int global_id_x = e.getKey();
	if (map2.containsKey(global_id_x)) {
	  int local_id2_x = map2.get(global_id_x);
	  int local_id1_x = map1.get(global_id_x);
	  for (Map.Entry<Integer, Integer> f: map1.entrySet()) {
	    int global_id_y = f.getKey();
	    if (map2.containsKey(global_id_y)) {
	      int local_id2_y = map2.get(global_id_y);
	      int local_id1_y = map1.get(global_id_y);
	      mat2[local_id2_x][local_id2_y] = mat1[local_id1_x][local_id1_y];
	    }
	  }
	}
      }
      return mat2;
    } else {
      double[][] mat2 = new double[n2][1];
      /* override some elements */
      for (Map.Entry<Integer, Integer> e: map1.entrySet()) {
	int global_id_x = e.getKey();
	if (map2.containsKey(global_id_x)) {
	  int local_id2_x = map2.get(global_id_x);
	  int local_id1_x = map1.get(global_id_x);
	  mat2[local_id2_x][0] = mat1[local_id1_x][0];
	}
      }
      return mat2;
    }
  }


  /**
   * sample_multivariate_normal:
   *  draw [n_samples] samples from multivariate normal distribution, and take average
   *  ref: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
   */
  public static double[] 
  sample_multivariate_normal(double[][] _mean, double[][] _variance, int n_samples) {
    int n = _variance.length;
    Matrix var = new Matrix(_variance);
    Matrix mean = new Matrix(_mean);
    Matrix c_fact = (new CholeskyDecomposition(var)).getL();
    Random rand = new Random();
    double[] _sample = new double[n];
    /* start sampling */
    for (int ns = 0; ns < n_samples; ns++) {
      double[][] _z = new double[n][1];
      for (int i = 0; i < n; i++) {
	_z[i][0] = rand.nextGaussian();
      }
      Matrix z = new Matrix(_z);
      Matrix sample = mean.plus(c_fact.times(z));
      for (int i = 0; i < n; i++) {
	_sample[i] += sample.get(i,0);
      }
    }
    /* end sampling */

    for (int i = 0; i < n; i++) {
      _sample[i] /= n_samples;
    }

    return _sample;
  }

  /**
   * sample_multivariate_normal (override):
   *  draw [n_samples] samples from multivariate normal distribution, and take average
   *  variance is diagnoal in this method
   */
  public static double[] 
  sample_multivariate_normal(double[] _mean, double[] _variance, int n_samples) {
    int n = _variance.length;
    Random rand = new Random();
    double[] _sample = new double[n];
    /* start sampling */
    for (int ns = 0; ns < n_samples; ns++) {
      double[] _z = new double[n];
      for (int i = 0; i < n; i++) {
	_z[i] = rand.nextGaussian();
      }
      for (int i = 0; i < n; i++) {
	_sample[i] += _z[i] * Math.sqrt(_variance[i]) + _mean[i];
      }
    }
    /* end sampling */

    for (int i = 0; i < n; i++) {
      _sample[i] /= n_samples;
    }

    return _sample;
  }

}

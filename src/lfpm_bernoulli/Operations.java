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
	/* random prior */ 
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
   */
  public static double[][] 
  sample_multivariate_normal(double[][] mean, double variance, int n_samples) {
    if (variance < 0) {
      System.out.println("Variance = " + variance);
      System.out.println("ERROR: variance should be non-negative!");
      Scanner sc = new Scanner(System.in); int gu = sc.nextInt();
    }
    if (n_samples == 0) 
      return mean;

    Random rand = new Random();
    int n = mean.length; int K = Main.K;
    double[][] sample = new double[n][K];
    /* start sampling */
    for (int ns = 0; ns < n_samples; ns++) {
      for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	double z = rand.nextGaussian();
	sample[i][k] += z * Math.sqrt(variance) + mean[i][k];
      }
    }
    /* end sampling */

    for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
      sample[i][k] /= n_samples;
    }

    return sample;
  }

  /**
   * sample_multivariate_normal:
   *  draw [n_samples] samples from multivariate normal distribution
   *  when the variance is diagnoal (and identical)
   */
  public static double[] 
  sample_multivariate_normal(double[] mean, double variance, int n_samples) {
    if (variance < 0) {
      System.out.println("Variance = " + variance);
      System.out.println("ERROR: variance should be non-negative!");
      Scanner sc = new Scanner(System.in); int gu = sc.nextInt();
    }
    if (n_samples == 0) 
      return mean;

    Random rand = new Random();
    int n = mean.length;
    double[] sample = new double[n];
    /* start sampling */
    for (int ns = 0; ns < n_samples; ns++) {
      for (int i = 0; i < n; i++) {
	double z = rand.nextGaussian();
	sample[i] += z * Math.sqrt(variance) + mean[i];
      }
    }
    /* end sampling */

    for (int i = 0; i < n; i++) {
      sample[i] /= n_samples;
    }

    return sample;
  }

  /**
   * log_sum_exp:
   *  for inputs a1, a2, ...
   *  output log (e^a1 + e^a2 + ... )
   */
  public static double log_sum_exp(List<Double> ls) {
    double ins_log = 0;
    Collections.sort(ls);
    double v_max = ls.get(ls.size()-1);
    if (Double.isNaN(v_max)) {
      System.out.println("ERROR6");
    }
    for (int i = 0; i < ls.size(); i++) {
      ins_log += Math.exp(ls.get(i) - v_max);
    }
    double res = v_max + Math.log(ins_log);
    if (Double.isNaN(res)) {
      System.out.println("ERROR4");
      Scanner sc = new Scanner(System.in);
      int gu = sc.nextInt();
    }
    return res;
  }

  /** 
   * log_one_plus_exp:
   *	input: x
   *	output: log( 1 + e^x )
   */
  public static double log_one_plus_exp(double x) {
    if (x <= 0) {
      double res = Math.log(1.0 + Math.exp(x));
      return res;
    } else {
      double res = Math.log(1.0 + Math.exp(-x));
      return x + res;
    }
  }


  /**
   * calculate the inner product of two vectors
   */
  public static double
  inner_product(double[] arr1, double[] arr2, int dim) {
    double res = 0;
    for (int i = 0; i < dim; i++) 
      res += arr1[i] * arr2[i];
    return res;
  }

  /** 
   * sigmoid: output 1/(1+e^{-x})
   */
  public static double
  sigmoid(double x) {
    if (x > 0) {
      double res = 1.0 / (1.0 + Math.exp(-x));
      return res;
    } else {
      double res = 1.0 / (1.0 + Math.exp(x));
      return 1-res;
    }
  }

}

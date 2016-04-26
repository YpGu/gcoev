import java.io.*;
import java.io.File;
import java.util.*;
import Jama.*;

public class Main {
  public static int t0 = 0;
  public static int T = 17;
  public static double lambda = 0.5;
//  public static double sigma = 0.3;
  public static double sigma = 1;
  public static double delta = 0.3;
  public static double scale = 0.2;
  public static double scale_0 = 0;
  public static int N_SAMPLES = 100;   // number of samples from multi-variate normal distribution
  public static Random rand = new Random();

  public static int n = 110;    // TODO
  public static int K = 15;
  public static double lr_1 = 0.03;
  public static double lr_2 = 0.005;
  public static int MAX_ITER = 100;
  public static int INNER_ITER = 100;

  /* global data */
  public static List<double[][]> GS = new ArrayList<double[][]>(T);   // graph
  public static List<double[][]> AS = new ArrayList<double[][]>(T);   // adjacency matrix (off-diagonal) 
  public static List<Double> delta_s = new ArrayList<Double>(T);
  public static List<Double> delta_prime_s = new ArrayList<Double>(T);

  /* intrinsic features */
  public static List<double[][]> h_s = new ArrayList<double[][]>(T);	  // h: latent attribute (N*K)
  public static List<double[][]> h_hat_s = new ArrayList<double[][]>(T);  // \hat{h}: variational parameter (N*K)
  public static List<double[][]> mu_s = new ArrayList<double[][]>(T);	  // \mu: forward mean (N*K)
  public static List<Double> v_s = new ArrayList<Double>(T);		  // V: forward variance 
  public static List<double[][]> mu_hat_s = new ArrayList<double[][]>(T); // \hat{\mu}: backward mean (N*K)
  public static List<Double> v_hat_s = new ArrayList<Double>(T);	  // \hat{V}: backward variance (K)

  /* impression features */
  public static List<double[][]> h_prime_s = new ArrayList<double[][]>(T);	// h': latent attribute (N*K)
  public static List<double[][]> h_hat_prime_s = new ArrayList<double[][]>(T);  // \hat{h}': variational parameter (N*K)
  public static List<double[][]> mu_prime_s = new ArrayList<double[][]>(T);	// \mu': forward mean (N*K)
  public static List<Double> v_prime_s = new ArrayList<Double>(T);		// V': forward variance (K)
  public static List<double[][]> mu_hat_prime_s = new ArrayList<double[][]>(T);	// \hat{\mu}': backward mean (N*K)
  public static List<Double> v_hat_prime_s = new ArrayList<Double>(T);		// \hat{V}': backward variance (K)

  /* gradients (forward/backward)
   * denote T' = T-t0
   * then we need to store T'*T' gradients: 
   *  e.g. \partial \mu^{t} / \partial h^{s} for all (t,s)
   * each (t,s) is saved as grad_mu_s.get(t * T' + s)
   */
  public static List<double[][]> grad_mu_s = new ArrayList<double[][]>(T*T);	    // grad w.r.t \mu: forward (\part t)/(\part s) (each i)
  public static List<double[][]> grad_mu_hat_s = new ArrayList<double[][]>(T*T);	    // grad w.r.t \hat{\mu}: backward 
  public static List<double[][]> grad_mu_prime_s = new ArrayList<double[][]>(T*T);	    // grad w.r.t \mu': forward
  public static List<double[][]> grad_mu_hat_prime_s = new ArrayList<double[][]>(T*T);  // grad of \hat{\mu}': backward

  /* gradients (global) */
  public static List<double[][]> grad_h_hat_s = new ArrayList<double[][]>(T);	    // grad of ELBO w.r.t. \mu (each i, k)
  public static List<double[][]> grad_h_hat_prime_s = new ArrayList<double[][]>(T);	    // grad of ELBO w.r.t \hat{\mu} (each i, k)

  public static void test1(String seed) {
    /* read, init data & parameters */
    for (int t = t0; t < T; t++) {
//      String fileDir = "../../data/graph/" + Integer.toString(t) + ".csv";  // original co-voting dataset
//      String fileDir = "./data/" + Integer.toString(t) + ".csv";  // artificial toy dataset
      String fileDir = "../../data_sm/nips_17/out/" + seed + "/" + Integer.toString(t) + ".train.csv";  // nips dataset (smaller)
      Map<Integer, Double> freq = FileParser.readCSVDict(fileDir);

      double[][] G = new double[n][n];
      double[][] A = new double[n][n];
      double[][] mu = new double[n][K];
      double[][] mu_hat = new double[n][K];
      double[][] mu_prime = new double[n][K];
      double[][] mu_hat_prime = new double[n][K];
      double[][] h = new double[n][K];
      double[][] h_hat = new double[n][K];
      FileParser.readCSVGraph(fileDir, freq, G, A);
      for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	mu[i][k] = scale_0 * (rand.nextDouble() - 0.5);
	mu_hat[i][k] = scale_0 * (rand.nextDouble() - 0.5);
	mu_prime[i][k] = mu[i][k];
	mu_hat_prime[i][k] = mu_hat[i][k];
	h[i][k] = scale * (rand.nextDouble() - 0.5);
	h_hat[i][k] = scale * (rand.nextDouble() - 0.5);
      }

      GS.add(G); AS.add(A);
      mu_s.add(mu); mu_hat_s.add(mu_hat); mu_prime_s.add(mu_prime); mu_hat_prime_s.add(mu_hat_prime);
      h_s.add(h); h_prime_s.add(h); h_hat_s.add(h_hat); h_hat_prime_s.add(h);

      /* for test */
      delta_s.add(0.1);
      delta_prime_s.add(0.1);

      v_s.add(0.1); v_hat_s.add(0.1); 
      v_prime_s.add(0.1); v_hat_prime_s.add(0.1);

      System.out.println("done! t = " + t);
    }

    for (int t = t0; t < T; t++) {
      for (int s = t0; s < T; s++) {
	grad_mu_s.add(new double[n][K]);
	grad_mu_hat_s.add(new double[n][K]);
	grad_mu_prime_s.add(new double[n][K]);
	grad_mu_hat_prime_s.add(new double[n][K]);
      }
      grad_h_hat_s.add(new double[n][K]);
      grad_h_hat_prime_s.add(new double[n][K]);
    }
    /* end initialization */

    /* outer for-loop */
    for (int iter = 0; iter < MAX_ITER; iter++) {
//      Scanner sc = new Scanner(System.in); int gu; gu = sc.nextInt();
      System.out.println("====== iter = " + iter + " ======");
      /** intrinsic feature **/
      forward1(true, iter); backward1(true);
      compute_gradient1(iter);
      double old_obj = 0;
      /* gradient descent: inner for-loop here */
      for (int inner_iter = 0; inner_iter < INNER_ITER; inner_iter++) {
	/* update variational parameters \hat{h} using gradient descent */
	for (int t = 0; t < T-t0; t++) {
	  double[][] h_hat_t = h_hat_s.get(t);
	  double[][] grad_h_hat_t = grad_h_hat_s.get(t);
	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    h_hat_t[i][k] += lr_1 * grad_h_hat_t[i][k];
	  }
	  h_hat_s.set(t, h_hat_t);
	}
	/* update \hat{\mu} and \hat{V}, since both are function of \hat{h} */
	forward1(false, iter); backward1(false);
	double obj1 = compute_objective1();
	if (inner_iter%10 == 0) 
	  System.out.println("(1) iter = " + inner_iter + ", obj 1 = " + obj1);
	if (inner_iter != 0 && obj1 < old_obj) {
	  lr_1 *= 0.8;
	  break;
	}
	old_obj = obj1;
      }
      /* sample */
      for (int t = 0; t < T-t0; t++) {
	double[][] samples = Operations.sample_multivariate_normal(mu_hat_s.get(t), v_hat_s.get(t), N_SAMPLES);
	double[][] h_t = new double[n][K];
	for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	  h_t[i][k] = samples[i][k];
	}
	h_s.set(t, h_t);
      }


      /** impression feature **/
      forward2(true); backward2(true);
      compute_gradient2(iter);
      /* gradient descent: inner for-loop here */
      for (int inner_iter = 0; inner_iter < INNER_ITER; inner_iter++) {
	/* update \hat{h}' using gradient descent */
	for (int t = 0; t < T-t0; t++) {
	  double[][] h_hat_prime_t = h_hat_prime_s.get(t);
	  double[][] grad_h_hat_prime_t = grad_h_hat_prime_s.get(t);
	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    h_hat_prime_t[i][k] += lr_2 * grad_h_hat_prime_t[i][k];
	  }
	  h_hat_prime_s.set(t, h_hat_prime_t);
	}
	/* update \hat{\mu}' and \hat{V}', since both are function of \hat{h}' */
	forward2(false); backward2(false);
	double obj2 = compute_objective2();
	if (inner_iter%10 == 0) 
	  System.out.println("(2) iter = " + inner_iter + ", obj 2 = " + obj2);
	if (inner_iter != 0 && obj2 < old_obj) {
	  lr_2 *= 0.8;
	  break;
	}
	old_obj = obj2;
      }
      /* sample */
      for (int t = 0; t < T-t0; t++) {
	double[][] samples = Operations.sample_multivariate_normal(mu_hat_prime_s.get(t), v_hat_prime_s.get(t), N_SAMPLES);
	double[][] h_prime_t = new double[n][K];
	for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	  h_prime_t[i][k] = samples[i][k];
	}
	h_prime_s.set(t, h_prime_t);
      }

      /** output **/
      for (int t = 0; t < T-t0; t++) {
	double[][] h_t = h_s.get(t); 
	double[][] h_prime_t = h_prime_s.get(t);
	/* output filename: 
	 *    ./res/<seed>/h_<time>_<iter>.txt 
	 */
	FileParser.output(h_t, "./res/" + seed + "/h_" + (t+t0) + "_" + iter + ".txt");
	FileParser.output(h_prime_t, "./res/" + seed + "/h_p_" + (t+t0) + "_" + iter + ".txt");
      }
    }
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
   * compute_objective1:
   *  return the lower bound when h' is fixed
   */
  public static double compute_objective1() {
    double res = 0;
    for (int t = 0; t < T-t0; t++) {
      if (t != 0) {
	double[][] G_t = GS.get(t);
	double[][] h_prime_t = h_prime_s.get(t);
	double[][] h_prime_pre_t = h_prime_s.get(t-1);
	double[][] mu_hat_t = mu_hat_s.get(t);
	double[][] mu_hat_pre_t = mu_hat_s.get(t-1);
	double delta_t = delta_s.get(t);

	Matrix a = new Matrix(AS.get(t-1));
	Matrix hprime_pre_t = new Matrix(h_prime_s.get(t-1));
	Matrix ave_neighbors = a.times(hprime_pre_t);

	double[] hp2delta2 = new double[n];
	for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	  hp2delta2[i] += 0.5 * h_prime_t[i][k] * h_prime_t[i][k] * delta_t * delta_t;
	}

	for (int i = 0; i < n; i++) {
	  /* first term */
	  List<Double> powers = new ArrayList<Double>();
	  for (int l = 0; l < n; l++) {
	    double hp_muh = Operations.inner_product(h_prime_t[l], mu_hat_t[i], K);
	    powers.add(hp_muh + hp2delta2[l]);
	  }
	  double lse = log_sum_exp(powers);

	  for (int j = 0; j < n; j++) if (G_t[i][j] != 0) {
	    double hp_muh = Operations.inner_product(h_prime_t[j], mu_hat_t[i], K);
	    res += G_t[i][j] * (hp_muh - lse);
	  }

	  /* second term */
	  for (int k = 0; k < K; k++) {
	    double diff = mu_hat_t[i][k] - (1-lambda) * mu_hat_pre_t[i][k] - lambda * ave_neighbors.get(i,k);
	    res -= 0.5 * diff * diff / (sigma*sigma);
	  }
	}
      } else {
	/*
	double[][] G_t = GS.get(t);
	double[][] h_prime_t = h_prime_s.get(t);
	double[] mu_hat_t = mu_hat_s.get(t);
	double delta_t = delta_s.get(t);
	int[][] neg_sam_t = neg_samples.get(t);

	for (int i = 0; i < n; i++) {
	  // first term 
	  for (int j = 0; j < n; j++) if (G_t[i][j] != 0) {
	    List<Double> powers = new ArrayList<Double>();
	    for (int _l = 0; _l < NEG; _l++) {
	      int l = neg_sam_t[i][_l];
	      powers.add(h_prime_t[l][0] * mu_hat_t[i] 
		  + 0.5 * h_prime_t[l][0] * h_prime_t[l][0] * delta_t * delta_t);
	    }
	    double lse = log_sum_exp(powers);
	    res += G_t[i][j] * (h_prime_t[j][0] * mu_hat_t[i] - lse);
	  }
	}
	*/
      }
    }
    return res;
  }

  public static void compute_gradient1(int iteration) {
    double[][][] tmp_grad_h_hat_s = new double[T-t0][n][K];

    for (int t = 0; t < T-t0; t++) {
//      System.out.println("compute gradient 1, t = " + t);
      double delta_t = delta_s.get(t);
      double[][] G_t = GS.get(t);
      double[][] h_prime_t = h_prime_s.get(t);
      double[][] mu_hat_t = mu_hat_s.get(t);

      if (t != 0) {
	double[][] mu_hat_pre_t = mu_hat_s.get(t-1);

	Matrix a = new Matrix(AS.get(t-1));
	Matrix hprime_pre_t = new Matrix(h_prime_s.get(t-1));
	Matrix ave_neighbors = a.times(hprime_pre_t);

	/* TODO: check whether we can save computation by comparing s and t */
	for (int s = 0; s < T-t0; s++) {
	  double[][] grad_hat_t = grad_mu_hat_s.get( t * (T-t0) + s );
	  double[][] grad_hat_pre_t = grad_mu_hat_s.get( (t-1) * (T-t0) + s );
	  double[] hp2delta2 = new double[n];
	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    hp2delta2[i] += 0.5 * h_prime_t[i][k] * h_prime_t[i][k] * delta_t * delta_t;
	  }

	  for (int i = 0; i < n; i++) {
	    /* first term */
	    double[] weighted_exp_num = new double[K]; 
	    double weighted_exp_den = 0;
	    for (int l = 0; l < n; l++) {
	      double hp_muh = Operations.inner_product(h_prime_t[l], mu_hat_t[i], K);
	      double e = Math.exp(hp_muh + hp2delta2[l]);
	      if (Double.isNaN(e)) {
		/* check if e explodes */
		System.out.println("ERROR2");
		Scanner sc = new Scanner(System.in);
		int gu; gu = sc.nextInt();
	      }
	      for (int k = 0; k < K; k++) {
		weighted_exp_num[k] += h_prime_t[l][k] * e;
		weighted_exp_den += e;
	      }
	    }
	    for (int j = 0; j < n; j++) for (int k = 0; k < K; k++) {
	      double weighted_exp = weighted_exp_num[k] / weighted_exp_den;
	      double gi1 = G_t[i][j] * grad_hat_t[i][k] * (h_prime_t[j][k] - weighted_exp);
	      tmp_grad_h_hat_s[s][i][k] += gi1;
	    }

	    /* second term */
	    for (int k = 0; k < K; k++) {
	      double gi2 = -(mu_hat_t[i][k] - (1-lambda) * mu_hat_pre_t[i][k] - lambda * ave_neighbors.get(i,k))
		* (grad_hat_t[i][k] - (1-lambda) * grad_hat_pre_t[i][k]) / (sigma*sigma);
	      tmp_grad_h_hat_s[s][i][k] += gi2;
	    }
	  }
	}
      } else {
	/* no such term (t=0) in ELBO */
	/*
	for (int s = 0; s < T-t0; s++) {
	  double[] grad_hat_t = grad_mu_hat_s.get(t * (T-t0) + s);

	  for (int i = 0; i < n; i++) {
	    double n_it = 0;
	    for (int j = 0; j < n; j++) n_it += G_t[i][j];

	    // first term 
	    double gi1 = -mu_hat_t[i] * grad_hat_t[i] / (sigma * sigma);
	    tmp_grad_h_hat_s[s][i] += gi1;

	    // second term 
	    double gi2 = 0;
	    double weighted_exp_num = 0, weighted_exp_den = 0;
	    for (int j = 0; j < NEG; j++) {
	      int l = neg_samples.get(t)[i][j];
	      double hpl = h_prime_t[l][0];
	      double muit = mu_hat_t[i];
	      double e = Math.exp(hpl * muit + 0.5 * hpl * hpl * delta_t * delta_t);
	      // TODO: check if e explodes 
	      if (Double.isNaN(e)) {
		System.out.println("ERROR3");
		Scanner sc = new Scanner(System.in);
		int gu; gu = sc.nextInt();
	      }
	      weighted_exp_num += hpl * e;
	      weighted_exp_den += e;
	    }
	    double weighted_exp = weighted_exp_num / weighted_exp_den;
	    for (int j = 0; j < n; j++) {
	      gi2 += G_t[i][j] * grad_hat_t[i] * (h_prime_t[j][0] - weighted_exp);
	    }
	    tmp_grad_h_hat_s[s][i] += gi2;
	  }
	}
	*/
      }
      /* end if-else */
    }

    /* update global gradient */
    for (int t = 0; t < T-t0; t++) {
      double[][] grad = new double[n][K];
      for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	grad[i][k] = tmp_grad_h_hat_s[t][i][k];
      }
      grad_h_hat_s.set(t, grad);
    }
    FileParser.output_2d(grad_h_hat_s, "./grad/grad_" + iteration + ".txt");

    return;
  }

  /**
   * compute_objective2:
   *  return the lower bound when h is fixed
   */
  public static double compute_objective2() {
    double res = 0;
    for (int t = 0; t < T-t0; t++) {
      if (t != 0) {
	double[][] G_t = GS.get(t); double[][] G_t_pre = GS.get(t-1);
	double[][] h_t = h_s.get(t);
	double[][] h_pre_t = h_s.get(t-1);
	double[][] mu_hat_prime_t = mu_hat_prime_s.get(t);
	double[][] mu_hat_prime_pre_t = mu_hat_prime_s.get(t-1);
	double delta_t = delta_s.get(t);

	double[][] a_pre = AS.get(t-1);
	double[][] ave_neighbors = new double[n][K];
	for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (G_t_pre[i][j] != 0) {
	  for (int k = 0; k < K; k++) {
	    ave_neighbors[i][k] += a_pre[i][j] * mu_hat_prime_pre_t[j][k];
	  }
	}

	for (int i = 0; i < n; i++) {
	  /* first term */
	  double h2delta2 = 0;
	  for (int k = 0; k < K; k++) {
	    h2delta2 += 0.5 * h_t[i][k] * h_t[i][k] * delta_t * delta_t;
	  }
	  List<Double> powers = new ArrayList<Double>();
	  for (int l = 0; l < n; l++) {
	    double h_muhp = Operations.inner_product(h_t[i], mu_hat_prime_t[l], K);
	    powers.add(h_muhp + h2delta2);
	  }
	  double lse = log_sum_exp(powers);

	  for (int j = 0; j < n; j++) if (G_t[i][j] != 0) {
	    double h_muhp = Operations.inner_product(h_t[i], mu_hat_prime_t[j], K);
	    res += G_t[i][j] * (h_muhp - lse);
	  }

	  /* second term */
	  for (int k = 0; k < K; k++) {
	    double diff = h_t[i][k] - (1-lambda) * h_pre_t[i][k] - lambda * ave_neighbors[i][k];
	    res -= 0.5 * diff * diff / (sigma*sigma);
	  }

	  /* third term */
	  for (int k = 0; k < K; k++) {
	    double diff_3 = mu_hat_prime_t[i][k] - mu_hat_prime_pre_t[i][k];
	    res -= 0.5 * diff_3 * diff_3 / (sigma*sigma);
	  }
	}
      } else {
	/*
	double[][] G_t = GS.get(t);
	double[][] h_t = h_s.get(t);
	double[] mu_hat_prime_t = mu_hat_prime_s.get(t);
	double delta_t = delta_s.get(t);
	int[][] neg_sam_t = neg_samples.get(t);

	for (int i = 0; i < n; i++) {
	  // first term 
	  for (int j = 0; j < n; j++) if (G_t[i][j] != 0) {
	    List<Double> powers = new ArrayList<Double>();
	    for (int _l = 0; _l < NEG; _l++) {
	      int l = neg_sam_t[i][_l];
	      powers.add(h_t[i][0] * mu_hat_prime_t[l]
		  + 0.5 * h_t[i][0] * h_t[i][0] * delta_t * delta_t);
	    }
	    double lse = log_sum_exp(powers);
	    res += G_t[i][j] * (h_t[i][0] * mu_hat_prime_t[j] - lse);
	  }
	}
	*/
      }
    }
    return res;
  }

  public static void compute_gradient2(int iteration) {
    double[][][] tmp_grad_h_hat_prime_s = new double[T-t0][n][K];

    /* 
     * compute 
     *	  nti[t][i] = \sum_{j} { n_{ij} }
     * and 
     *	  nti_h[t][j][k] = \sum_{i} { n_{ij}^{t} h_{ik}^{t} }
     */
    double[][] nti = new double[T-t0][n];
    double[][][] nti_h = new double[T-t0][n][K];
    for (int t = 0; t < T-t0; t++) {
      double[][] G_t = GS.get(t);
      double[][] h_t = h_s.get(t);  // h^{t}
      for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
	nti[t][i] += G_t[i][j];
	for (int k = 0; k < K; k++) {
	  nti_h[t][j][k] += G_t[i][j] * h_t[i][k];
	}
      }
    }

    for (int t = 0; t < T-t0; t++) {
      double delta_t = delta_prime_s.get(t);
      double[][] h_t = h_s.get(t);    // h^{t}
      double[][] h_hat_prime_t = h_hat_prime_s.get(t);	// \hat{h}^{t}
      double[][] mu_hat_t = mu_hat_s.get(t);	// \hat{\mu}^{t}
      double[][] mu_hat_prime_t = mu_hat_prime_s.get(t);	// \hat{\mu}'^{t}
      double[][] h_prime_t = h_prime_s.get(t);

      if (t != 0) {
	Matrix a = new Matrix(AS.get(t-1));
	Matrix hprime_pre_t = new Matrix(h_prime_s.get(t-1));
	Matrix ave_neighbors = a.times(hprime_pre_t);

	double[][] G_pre_t = GS.get(t-1);   // G^{t-1}
	double[][] A_pre_t = AS.get(t-1);   // A^{t-1}
	double[][] h_pre_t = h_s.get(t-1);  // h^{t-1}
	double[][] mu_hat_prime_pre_t = mu_hat_prime_s.get(t-1);  // \hat{\mu}'^{t-1}  [t]

	for (int s = 0; s < T-t0; s++) {
	  double[][] grad_mu_hat_prime_t = grad_mu_hat_prime_s.get( t * (T-t0) + s );
	  double[][] grad_mu_hat_prime_pre_t = grad_mu_hat_prime_s.get( (t-1) * (T-t0) + s );
	  double[] h2delta2 = new double[n];
	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    h2delta2[i] += 0.5 * h_t[i][k] * h_t[i][k] * delta_t * delta_t;
	  }

	  /* compute weighted_exp for later use */
	  double[][][] weighted_exp_num = new double[K][n][n];
	  double[][] weighted_exp_den = new double[K][n];
	  double[][][] weighted_exp = new double[K][n][n];
	  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
	    double h_muhp = Operations.inner_product(h_t[j], mu_hat_prime_t[i], K);
	    for (int k = 0; k < K; k++) {
	      weighted_exp_num[k][i][j] = h_t[j][k] * Math.exp(h_muhp + h2delta2[j]);
	      weighted_exp_den[k][j] += Math.exp(h_muhp + h2delta2[j]);
	    }
	  }
	  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < K; k++) {
	    weighted_exp[k][i][j] = weighted_exp_num[k][i][j] / weighted_exp_den[k][j];
	  }
	  /* compute sum_mu_hat_prime for later use */
	  double[] sum_mu_hat_prime = new double[K];
	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    sum_mu_hat_prime[k] += mu_hat_prime_pre_t[i][k];
	  }

	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    /* first term */
	    double g1 = nti_h[t][i][k] * grad_mu_hat_prime_t[i][k];
	    tmp_grad_h_hat_prime_s[s][i][k] += g1;
	    
	    /* second term */
	    double g2 = 0;
	    for (int j = 0; j < n; j++) {
	      g2 -= nti[t][j] * weighted_exp[k][i][j] * grad_mu_hat_prime_t[i][k];
	    }
	    tmp_grad_h_hat_prime_s[s][i][k] += g2;

	    /* third term */
	    for (int j = 0; j < n; j++) if (G_pre_t[j][i] != 0) {
	      double g3 = ( h_t[j][k] - (1-lambda) * h_pre_t[j][k] - lambda * A_pre_t[j][i] * sum_mu_hat_prime[k] )
		* lambda * A_pre_t[j][i] * grad_mu_hat_prime_pre_t[i][k] / ( sigma*sigma ) ;
	      tmp_grad_h_hat_prime_s[s][j][k] += g3;   // j instead of i!
	    }
	  }

	  /* fourth term */
	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    double g4 = -( mu_hat_prime_t[i][k] - mu_hat_prime_pre_t[i][k] ) 
	      * ( grad_mu_hat_prime_t[i][k] - grad_mu_hat_prime_pre_t[i][k] ) / ( sigma*sigma );
	    tmp_grad_h_hat_prime_s[s][i][k] += g4;
	  }
	}
      } else {
	/*
	for (int s = 0; s < T-t0; s++) {
	  double[] grad_mu_hat_prime_t = grad_mu_hat_prime_s.get(t * (T-t0) + s);
	  for (int i = 0; i < n; i++) {
	    // first term 
	    double g1 = nti_hp[t][i] * grad_mu_hat_prime_t[i];
	    tmp_grad_h_hat_prime_s[s][i] += g1;
	    
	    // second term 
	    double g2 = 0;
	    for (int _j = 0; _j < NEG; _j++) {
	      double weighted_exp_num = 0, weighted_exp_den = 0;
	      int j = neg_samples.get(t)[i][_j];
	      double htj = h_t[j][0]; double muhti = mu_hat_t[i];
	      weighted_exp_num += htj * Math.exp(htj * muhti + 0.5 * htj * htj * delta_t * delta_t);
	      for (int _k = 0; _k < NEG; _k++) {
		int k = neg_samples.get(t)[i][_k];
		double muhtk = mu_hat_t[k];
		weighted_exp_den += Math.exp(htj * muhtk + 0.5 * htj * htj * delta_t * delta_t);
	      }
	      g2 -= nti[t][j] * weighted_exp_num / weighted_exp_den * grad_mu_hat_prime_t[i];
	    }
	    tmp_grad_h_hat_prime_s[s][i] += g2;
	  }

	  // fourth term (if any)
	  if (s == t) for (int i = 0; i < n; i++) {
	    double g4 = -h_hat_prime_t[i][0] / (sigma*sigma);
	    tmp_grad_h_hat_prime_s[s][i] += g4;
	  }
	}
	*/
      }

    }

    /* update global gradient */
    for (int t = 0; t < T-t0; t++) {
      double[][] grad = new double[n][K];
      for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	grad[i][k] = tmp_grad_h_hat_prime_s[t][i][k];
      }
      grad_h_hat_prime_s.set(t, grad);
    }
    FileParser.output_2d(grad_h_hat_prime_s, "./grad/grad_prime_" + iteration + ".txt");

    return;
  }

  /** 
   * forward pass 1: update intrinsic features 
   *  (1) mu (mu_s) 
   *  (2) grad_mu (grad_mu_s) 
   *  (3) variance V (v_s)
   */
  public static void forward1(boolean update_grad, int iter) {
    /*
    if (iter == 4) {
      int t = 15;
      double[][] h_t = new double[n][K];
      double[][] h_hat_t = new double[n][K];
      double[][] h_prime_t = new double[n][K];
      double[][] h_hat_prime_t = new double[n][K];
      for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	h_t[i][k] = h_s.get(t-1)[i][k];
	h_hat_t[i][k] = h_hat_s.get(t-1)[i][k];
	h_prime_t[i][k] = h_prime_s.get(t-1)[i][k];
	h_hat_prime_t[i][k] = h_hat_prime_s.get(t-1)[i][k];
      }
      h_s.set(t, h_t); h_hat_s.set(t, h_hat_t);
      h_prime_s.set(t, h_prime_t); h_hat_prime_s.set(t, h_hat_prime_t);
    }
    */

    for (int t = 0; t < T-t0; t++) {
//      System.out.println("forward 1;\tt = " + t1);
      if (t != 0) {
	double delta_t = delta_s.get(t);      // delta_t
	double[][] h_hat_t = h_hat_s.get(t);  // \hat{h}^t  [t]
	double[][] mu_pre_t = mu_s.get(t-1);    // mu^{t-1} (N*1)
	double V_pre_t = v_s.get(t-1);	      // V^{t-1}

	Matrix a = new Matrix(AS.get(t-1));   // A^{t-1}
	Matrix hprime_pre_t = new Matrix(h_prime_s.get(t-1));   // h'^{t-1} 
	Matrix ave_neighbors = a.times(hprime_pre_t);

	/* calculate \mu */
	double[][] mu_t = new double[n][K];
	double factor_1 = ( delta_t*delta_t ) / ( delta_t*delta_t + sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t );
	double factor_2 = ( sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t ) 
	  / ( delta_t*delta_t + sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t );
	for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	  mu_t[i][k] = factor_1 * ( (1-lambda) * mu_pre_t[i][k] + lambda * ave_neighbors.get(i,k) ) 
	    + factor_2 * h_hat_t[i][k];
	}
	/* calculate V */
	double V_t = factor_2 * delta_t * delta_t;

	/* update \mu and V */
	mu_s.set(t, mu_t);
	v_s.set(t, V_t);

	/* calculate and update grad_mu */
	if (update_grad) for (int s = 0; s < T-t0; s++) {
	  double[][] grad_pre_t_s = grad_mu_s.get( (t-1) * (T-t0) + s );
	  double[][] grad_t_s = new double[n][K];
	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    grad_t_s[i][k] = factor_1 * (1-lambda) * grad_pre_t_s[i][k];
	    if (t == s) {
	      grad_t_s[i][k] += factor_2;
	    }
	  }
	  grad_mu_s.set( t * (T-t0) + s, grad_t_s );
	}
      } else {
	/* mu, V: random init (keep unchanged) */
	/* grad_mu: set to 0 (keep unchanged) */
      }
//      Scanner sc = new Scanner(System.in);
//      int gu; gu = sc.nextInt();
      /* end for each t */
    }
  }

  /**
   * forward pass 2: update impression features 
   *  (1) mu' (mu_prime_s) 
   *  (2) variance V' (v_prime_s)
   */
  public static void forward2(boolean update_grad) {
    for (int t1 = t0; t1 < T; t1++) {
      int t = t1-t0;
//      System.out.println("forward 2;\tt = " + t1);
      if (t != 0) {
	double delta_t_prime = delta_prime_s.get(t);	  // delta'_t
	double[][] mu_prime_pre_t = mu_prime_s.get(t-1);	  // mu'^{t-1}  [t]
	double[][] h_hat_prime_t = h_hat_prime_s.get(t);  // \hat{h}'^{t} (n*1)
	double V_prime_pre_t = v_prime_s.get(t-1);	  // V'^{t-1} [t]

	/* calculate \mu */
	double[][] mu_prime_t = new double[n][K];
	double factor_1 = ( delta_t_prime*delta_t_prime ) 
	  / ( delta_t_prime*delta_t_prime + sigma*sigma + V_prime_pre_t );
	double factor_2 = ( sigma*sigma + V_prime_pre_t ) 
	  / ( delta_t_prime*delta_t_prime + sigma*sigma + V_prime_pre_t );
	for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	  mu_prime_t[i][k] = factor_1 * mu_prime_pre_t[i][k] + factor_2 * h_hat_prime_t[i][k];
	}
	/* calculate V */
	double v_prime_t = factor_2 * delta_t_prime * delta_t_prime;

	/* update */
	mu_prime_s.set(t, mu_prime_t);
	v_prime_s.set(t, v_prime_t);

	/* calculate and update grad_mu_prime */
	if (update_grad) for (int s = 0; s < T-t0; s++) {
	  double[][] grad_mu_prime_pre_t_s = grad_mu_prime_s.get( (t-1) * (T-t0) + s );
	  double[][] grad_mu_prime_t_s = new double[n][K];
	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    grad_mu_prime_t_s[i][k] = factor_1 * grad_mu_prime_pre_t_s[i][k];
	    if (s == t) {
	      grad_mu_prime_t_s[i][k] += factor_2;
	    }
	  }
	  grad_mu_prime_s.set( t * (T-t0) + s, grad_mu_prime_t_s );
	}
      } else {
	/* for \mu'_i: ignore the first term in the summation */
	/* for V': 
	 *    option 1: use v_prime_init for all i
	 *    option 2: ignore v_prime_{t-1} 
	 */
	/*
	double delta_t_prime = delta_prime_s.get(t);	      // delta'_t
	double[][] h_hat_prime_t = h_prime_s.get(t);   // \hat{h}'^{t} (n*1)
	*/
	/* calculate new round of parameters */
	/*
	double[] mu_prime_t = new double[n];
	for (int i = 0; i < n; i++) {
	  double c = (sigma*sigma) / (sigma*sigma + delta_t_prime*delta_t_prime);
	  mu_prime_t[i] = c * h_hat_prime_t[i][0];
	}
	double v_prime_t = v_prime_init;
	*/
	/* update */
	/*
	mu_prime_s.set(t, mu_prime_t);
	v_prime_s.set(t, v_prime_t);
	*/
      }
      /* end for each t */
    }
  }

  /* 
   * backward pass 1: update 
   *  (1) \hat{mu} (mu_hat_s) 
   *  (2) \hat{grad_mu} (grad_mu_hat_s)
   *  (3) \hat{V} (v_hat_s)
   */
  public static void backward1(boolean update_grad) {
    for (int t1 = T-1; t1 > t0; t1--) {
      int t = t1-t0;
//      System.out.println("backward 1;\tt = " + t1);
      if (t != T-1-t0) {
	double V_pre_t = v_s.get(t-1);	    // V^{t-1}
	double V_hat_t = v_hat_s.get(t);    // \hat{V}^{t}
	double[][] mu_pre_t = mu_s.get(t-1);  // \mu^{t-1}
	double[][] mu_hat_t = mu_hat_s.get(t);// \hat{\mu}^{t}  [t-1]

	Matrix A_pre_t = new Matrix(AS.get(t-1));		// A^{t-1}
	Matrix hprime_pre_t = new Matrix(h_prime_s.get(t-1));   // h'^{t-1} 
	Matrix ave_neighbors = A_pre_t.times(hprime_pre_t);	// n * 1

	/* calculate \hat{\mu} at time t-1 */
	double factor_1 = (1-lambda) * V_pre_t / ( sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t );
	double factor_2 = ( sigma*sigma ) / ( sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t );
	double[][] mu_hat_pre_t = new double[n][K];
	for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	  mu_hat_pre_t[i][k] = factor_1 * (mu_hat_t[i][k] - lambda * ave_neighbors.get(i,k))
	    + factor_2 * mu_pre_t[i][k];
	}
	/* calculate \hat{V} at time t-1 */
	double V_hat_pre_t = V_pre_t + factor_1 * factor_1 * ( V_hat_t - (1-lambda)*(1-lambda) * V_pre_t - (sigma*sigma) );

	/* update \mu and V */
	mu_hat_s.set(t-1, mu_hat_pre_t);
	v_hat_s.set(t-1, V_hat_pre_t);

	/* calculate and update grad_mu_hat at time t-1 */
	if (update_grad) for (int s = 0; s < T-t0; s++) {
	  double[][] grad_hat_t_s = grad_mu_hat_s.get( t * (T-t0) + s );
	  double[][] grad_pre_t_s = grad_mu_s.get( (t-1) * (T-t0) + s );
	  double[][] grad_hat_pre_t_s = new double[n][K];
	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    grad_hat_pre_t_s[i][k] = factor_1 * grad_hat_t_s[i][k] + factor_2 * grad_pre_t_s[i][k];
	  }
	  grad_mu_hat_s.set( (t-1) * (T-t0) + s, grad_hat_pre_t_s );
	}
      } else {
	/* 
	 * initial condition for backward pass: 
	 *  (1) \hat{mu}^{T} = mu^{T}
	 *  (2) \hat{V}^{T} = V^{T}
	 *  (3) \hat{grad_mu}^{T/s} = grad_mu^{T/s}, \forall s
	 */
	mu_hat_s.set(t, mu_s.get(t));
	v_hat_s.set(t, v_s.get(t));
	if (update_grad) for (int s = 0; s < T-t0; s++) {
	  grad_mu_hat_s.set( t * (T-t0) + s, grad_mu_s.get(t * (T-t0) + s) );
	}
      }
//      Scanner sc = new Scanner(System.in);
//      int gu; gu = sc.nextInt();
      /* end for each t */
    }
  }

  /**
   * backward pass 2: update impression features
   *  (1) \hat{mu}' (mu_hat_prime_s) 
   *  (2) \hat{V}' (v_hat_prime_s)
   */
  public static void backward2(boolean update_grad) {
    double c = (1-lambda)*(1-lambda) / (sigma*sigma);
    for (int t1 = T-1; t1 > t0; t1--) {
      int t = t1-t0;
//      System.out.println("backward 2;\tt = " + t1);
      if (t != T-1-t0) {
	double[][] mu_prime_pre_t = mu_prime_s.get(t-1);	  // \mu'^{t-1}  [t-1]
	double[][] mu_hat_prime_t = mu_hat_prime_s.get(t);  // \hat{\mu}'^{t}  [t-1]
	double V_prime_pre_t = v_prime_s.get(t-1);	  // V'^{t-1} 
	double V_hat_prime_t = v_hat_prime_s.get(t);	  // \hat{V}'{t}

	/* calculate \hat{\mu}' at time t-1 */
	double[][] mu_hat_prime_pre_t = new double[n][K];
	double factor_1 = ( sigma*sigma ) / ( V_prime_pre_t + sigma*sigma );
	double factor_2 = ( V_prime_pre_t ) / ( V_prime_pre_t + sigma*sigma );
	for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	  mu_hat_prime_pre_t[i][k] = factor_1 * mu_prime_pre_t[i][k] + factor_2 * mu_hat_prime_t[i][k];
	}
	/* calculate \hat{V}' at time t-1 */
	double v_hat_prime_pre_t = V_prime_pre_t 
	  + factor_2 * factor_2 * (V_hat_prime_t - V_prime_pre_t - sigma*sigma);

	/* update \mu and V */
	mu_hat_prime_s.set(t-1, mu_hat_prime_pre_t);
	v_hat_prime_s.set(t-1, v_hat_prime_pre_t);

	/* calculate and update grad_mu_hat_prime at time t-1 */
	if (update_grad) for (int s = 0; s < T-t0; s++) {
	  double[][] grad_mu_prime_pre_t_s = grad_mu_prime_s.get( (t-1) * (T-t0) + s );
	  double[][] grad_mu_hat_prime_t_s = grad_mu_hat_prime_s.get( t * (T-t0) + s );
	  double[][] grad_mu_hat_prime_pre_t_s = new double[n][K];
	  for (int i = 0; i < n; i++) for (int k = 0; k < K; k++) {
	    grad_mu_hat_prime_pre_t_s[i][k] = factor_1 * grad_mu_prime_pre_t_s[i][k] + factor_2 * grad_mu_hat_prime_t_s[i][k];
	  }
	  grad_mu_hat_prime_s.set( (t-1) * (T-t0) + s, grad_mu_hat_prime_pre_t_s );
	}
      } else {
	/*
	 * initial condition for backward pass:
	 *  (1) \hat{\mu}'^{T} = \mu'^{T}
	 *  (2) \hat{V}'^{T} = V'^{T}
	 *  (3) \hat{grad_mu}'^{T/s} = grad_mu'^{T/s}, \forall s
	 */
	mu_hat_prime_s.set(t, mu_prime_s.get(t));
	v_hat_prime_s.set(t, v_prime_s.get(t));
	if (update_grad) for (int s = 0; s < T-t0; s++) {
	  grad_mu_hat_prime_s.set( t * (T-t0) + s, grad_mu_prime_s.get(t * (T-t0) + s) );
	}
      }
      /* end for each t */
    }
  }


  /**
   * toy example
   */
  public static void test2() {
    int N = 500;
    double[][] m1 = new double[N][N];
    double[][] m2 = new double[N][N];
    double[][] m3 = new double[N][N];

    // init
    Random rand = new Random();
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
      m1[i][j] = 10 * (rand.nextDouble() - 0.2);
      m2[i][j] = 20 * (rand.nextDouble() - 0.8);
    }

    // inverse 
    System.out.println("Start");
    Matrix mat1 = new Matrix(m1);
    Matrix mat2 = mat1.inverse();
    Matrix mat3 = mat1.times(mat2);
    double[][] m4 = mat3.getArray();
    /*
    for (int i = 0; i < m4.length; i++) {
      int ss = 10;
      for (int j = 0; j < ss; j++) {
	System.out.printf("%f ", m4[i][j]);
      }
      System.out.print("\n");
    }
    */
    System.out.println("Done");

    /*
    // matrix *
    System.out.println("Start");
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
      double cell = 0;
      for (int k = 0; k < N; k++) 
	cell += m1[i][k] * m2[k][j];
//      System.out.printf("%f ", cell);
      m3[i][j] = cell;
    }
    System.out.println("Done");
    */
  }

  public static void main(String[] args) {
    if (args.length != 1) {
      System.out.println("Usage: java Main <seed>");
      System.exit(0);
    }
    String seed = args[0];
    File f = new File("./res/" + seed);
    f.mkdir();

    test1(seed);
  }

}



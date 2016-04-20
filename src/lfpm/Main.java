import java.io.*;
import java.util.*;
import Jama.*;

public class Main {
  public static int t0 = 105;
  public static int T = 110;
  public static int NEG = 5;
  public static double lambda = 0.5;
  public static double sigma = 0.3;
  public static double delta = 0.3;
  public static double scale = 0.2;
  public static double eps = 1e-6;    // avoid sigular 
  public static int N_SAMPLES = 20;   // number of samples from multi-variate normal distribution
  public static int MAX_USER_TIME = 1000;   // maximum of number of users that appear simultaneously
  public static Random rand = new Random(0);

  /* global data */
  public static List<Integer> NS = new ArrayList<Integer>(T);	      // number of users
  public static List<double[][]> GS = new ArrayList<double[][]>(T);   // graph
  public static List<double[][]> AS = new ArrayList<double[][]>(T);   // adjacency matrix (off-diagonal) 
  public static List<int[][]> neg_samples = new ArrayList<int[][]>(T);   // negative samples, used in computing gradients 
  public static List<Map<Integer, Integer>> id_map_s = new ArrayList<Map<Integer, Integer>>(T);	  // id map: global ID -> local ID
  public static List<Double> delta_s = new ArrayList<Double>(T);
  public static List<Double> delta_prime_s = new ArrayList<Double>(T);

  /* intrinsic features */
  public static List<double[][]> h_s = new ArrayList<double[][]>(T);	  // h: latent attribute
  public static List<double[][]> h_hat_s = new ArrayList<double[][]>(T);  // \hat{h}: variational parameter
  public static List<double[]> mu_s = new ArrayList<double[]>(T);	  // \mu: forward mean (N * T)
  public static List<double> v_s = new ArrayList<double>(T);		  // V: forward variance (1 * T)
  public static List<double[]> mu_hat_s = new ArrayList<double[][]>(T);	  // \hat{\mu}: backward mean (N * T)
  public static List<double> v_hat_s = new ArrayList<double>(T);	  // \hat{V}: backward variance  (1 * T)

  /* impression features */
  public static List<double[][]> h_prime_s = new ArrayList<double[][]>(T);	// h': latent attribute
  public static List<double[][]> h_hat_prime_s = new ArrayList<double[][]>(T);  // \hat{h}': variational parameter
  public static List<double[]> mu_prime_s = new ArrayList<double[]>(T);	// \mu': forward mean
  public static List<double[]> v_prime_s = new ArrayList<double[]>(T);	// V': forward variance (diagnoal)
  public static List<double[]> mu_hat_prime_s = new ArrayList<double[]>(T);	// \hat{\mu}': backward mean
  public static List<double[]> v_hat_prime_s = new ArrayList<double[]>(T);	// \hat{V}': backward variance
  public static double v_prime_init = 0.01;	// init variance for every h'

  /* gradients (forward/backward)
   * suppose T' = T-t0
   * then we need to store T'*T' gradients: 
   *  e.g. \partial \mu^{t} / \partial h^{s} for all (t,s)
   * each (t,s) is saved as grad_mu_s.get(t * T' + s)
   */
  public static List<double[]> grad_mu_s = new ArrayList<double[]>(T*T);	    // grad w.r.t \mu: forward (\part t)/(\part s) (each i)
  public static List<double[]> grad_mu_hat_s = new ArrayList<double[]>(T*T);	    // grad w.r.t \hat{\mu}: backward 
  public static List<double[]> grad_mu_prime_s = new ArrayList<double[]>(T*T);	    // grad w.r.t \mu': forward
  public static List<double[]> grad_mu_hat_prime_s = new ArrayList<double[]>(T*T);  // grad of \hat{\mu}': backward

  /* gradients (global) */
  public static List<double> grad_h_hat_s = new ArrayList<double>(T);	    // grad of ELBO w.r.t. \mu (each i)
  public static List<double> grad_h_hat_prime_s = new ArrayList<double>(T); // grad of ELBO w.r.t \hat{\mu} (each i)

  public static void test1() {
    /* read, init data & parameters */
    for (int t = t0; t < T; t++) {
      String fileDir = "../../data/graph/" + Integer.toString(t) + ".csv";
      Map<Integer, Integer> idMap = new HashMap<Integer, Integer>();
      Map<Integer, Integer> idMapInv = new HashMap<Integer, Integer>();
      Map<Integer, Double> freq = FileParser.readCSVDict(fileDir, idMap, idMapInv);
      id_map_s.add(idMap);
      FileParser.output(idMap, "./dict/" + t + ".map");

      int n = idMap.size();
      NS.add(n);
      double[][] G = new double[n][n];
      double[][] A = new double[n][n];
      double[] mu = new double[n]; double[] mu_hat = new double[n];
      double[] mu_prime = new double[n]; double[] mu_hat_prime = new double[n];
      double[][] h = new double[n][1];
      double[][] h_hat = new double[n][1];
      FileParser.readCSVGraph(fileDir, freq, idMap, G, A);
      for (int i = 0; i < n; i++) {
	mu[i] = scale * (rand.nextDouble() - 0.5);
	mu_hat[i] = scale * (rand.nextDouble() - 0.5);
	mu_prime[i] = mu[i][0];
	mu_hat_prime[i] = mu_hat[i][0];
	h[i][0] = scale * (rand.nextDouble() - 0.5);
	h_hat[i][0] = scale * (rand.nextDouble() - 0.5);
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
      int n = NS.get(t-t0);
      for (int s = t0; s < T; s++) {
	grad_mu_s.add(new double[n]);
	grad_mu_hat_s.add(new double[n]);
	grad_mu_prime_s.add(new double[n]);
	grad_mu_hat_prime_s.add(new double[n]);
      }
      grad_h_hat_s.add(new double[n]);
      grad_h_hat_prime_s.add(new double[n]);
    }

    /* negative samples */
    for (int t = 0; t < T-t0; t++) {
      int n = NS.get(t);
      int[] users = new int[n];
      for (int i = 0; i < n; i++) users[i] = i;
 
      int[][] local_neg_samples = new int[n][NEG];
      for (int i = 0; i < n; i++) {
	Collections.shuffle(Arrays.asList(users));
	for (int j = 0; j < NEG; j++) {
	  local_neg_samples[i][j] = users[j];
	}
      }
      neg_samples.add(local_neg_samples);
    }

    /* end initialization */

    /* outer for-loop */
    for (int iter = 0; iter < 10; iter++) {
      /* intrinsic feature */
      forward1(true);
      backward1(true);
      /* gradient descent */
      compute_gradient1(iter);
      /* inner for-loop here */
      for (int inner_iter = 0; inner_iter < 20; inner_iter++) {
	double lr = 0.001;
	/* update \hat{h} using gradient descent */
	for (int t = 0; t < T-t0; t++) {
	  int n = NS.get(t);
	  double[][] h_hat_t = h_hat_s.get(t);
	  double[] grad_h_hat_t = grad_h_hat_s.get(t);
	  for (int i = 0; i < n; i++) {
	    h_hat_t[i][0] += lr * grad_h_hat_t[i];
	  }
	  h_hat_s.set(t, h_hat_t);
	}
	/* update mu and V, since both are function of h */
	forward1(false); backward1(false);

	double obj1 = compute_objective1();
	System.out.println("(1) iter = " + inner_iter + ", obj 1 = " + obj1);
      }

      /* impression feature */
      forward2(true); backward2(true);
      /* gradient descent */
      compute_gradient2();
      /* inner for-loop here */
      for (int inner_iter = 0; inner_iter < 20; inner_iter++) {
	double lr = 0.001;
	/* update \hat{h} using gradient descent */
	for (int t = 0; t < T-t0; t++) {
	  int n = NS.get(t);
	  double[][] h_hat_prime_t = h_hat_prime_s.get(t);
	  double[] grad_h_hat_prime_t = grad_h_hat_prime_s.get(t);
	  for (int i = 0; i < n; i++) {
	    h_hat_prime_t[i][0] += lr * grad_h_hat_prime_t[i];
	  }
	  h_hat_s.set(t, h_hat_prime_t);
	}
	/* update mu' and V', since both are function of h' */
	forward2(false); backward2(false);

	double obj2 = compute_objective2();
	System.out.println("(2) iter = " + inner_iter + ", obj 2 = " + obj2);
      }


      /* sample and update */
      forward1(false); backward1(false);
      forward2(false); backward2(false);
      for (int t = 0; t < T-t0; t++) {
	double[] samples = Operations.sample_multivariate_normal(mu_hat_s.get(t), v_hat_s.get(t), N_SAMPLES);
	double[][] h_s_arr = new double[samples.length][1];
	for (int i = 0; i < samples.length; i++) {
	  h_s_arr[i][0] = samples[i];
	}
	h_s.set(t, h_s_arr);
      }
      for (int t = 0; t < T-t0; t++) {
	double[] samples = Operations.sample_multivariate_normal(mu_hat_prime_s.get(t), v_hat_prime_s.get(t), N_SAMPLES);
	double[][] h_s_arr = new double[samples.length][1];
	for (int i = 0; i < samples.length; i++) {
	  h_s_arr[i][0] = samples[i];
	}
	h_prime_s.set(t, h_s_arr);
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
	int n = NS.get(t);
	double[][] G_t = GS.get(t);
	double[][] h_prime_t = h_prime_s.get(t);
	double[][] h_prime_pre_t = Operations.translate(h_prime_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));
	double[][] mu_hat_t = mu_hat_s.get(t);
	double[][] mu_hat_pre_t = Operations.translate(mu_hat_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));
	double delta_t = delta_s.get(t);
	int[][] neg_sam_t = neg_samples.get(t);

	for (int i = 0; i < n; i++) {
	  /* first term */
	  for (int j = 0; j < n; j++) if (G_t[i][j] != 0) {
	    List<Double> powers = new ArrayList<Double>();
	    for (int _l = 0; _l < NEG; _l++) {
	      int l = neg_sam_t[i][_l];
	      powers.add(h_prime_t[l][0] * mu_hat_t[i][0] 
		  + 0.5 * h_prime_t[l][0] * h_prime_t[l][0] * delta_t * delta_t);
	    }
	    double lse = log_sum_exp(powers);
	    res += G_t[i][j] * (h_prime_t[j][0] * mu_hat_t[i][0] - lse);
	  }
	  /* second term */
	  double weighted_neighbor = 0, degree = 0;
	  for (int j = 0; j < n; j++) if (G_t[i][j] != 0) {
	    weighted_neighbor += h_prime_pre_t[j][0];
	    degree += 1;
	  }
	  if (degree != 0) weighted_neighbor /= degree;
	  double diff = mu_hat_t[i][0] - (1-lambda) * mu_hat_pre_t[i][0] - lambda * weighted_neighbor;
	  res -= 0.5 * diff * diff / (sigma*sigma);
	}
      } else {
	int n = NS.get(t);
	double[][] G_t = GS.get(t);
	double[][] h_prime_t = h_prime_s.get(t);
	double[][] mu_hat_t = mu_hat_s.get(t);
	double delta_t = delta_s.get(t);
	int[][] neg_sam_t = neg_samples.get(t);

	for (int i = 0; i < n; i++) {
	  /* first term */
	  for (int j = 0; j < n; j++) if (G_t[i][j] != 0) {
	    List<Double> powers = new ArrayList<Double>();
	    for (int _l = 0; _l < NEG; _l++) {
	      int l = neg_sam_t[i][_l];
	      powers.add(h_prime_t[l][0] * mu_hat_t[i][0] 
		  + 0.5 * h_prime_t[l][0] * h_prime_t[l][0] * delta_t * delta_t);
	    }
	    double lse = log_sum_exp(powers);
	    res += G_t[i][j] * (h_prime_t[j][0] * mu_hat_t[i][0] - lse);
	  }
	}
      }
    }
    return res;
  }

  /* TODO: there may be bugs in this method as objective is not increasing */
  public static void compute_gradient1(int iteration) {
    double[][] tmp_grad_h_hat_s = new double[T-t0][MAX_USER_TIME];

    for (int t = 0; t < T-t0; t++) {
//      System.out.println("compute gradient 1, t = " + t);
      int n = NS.get(t);
      double delta_t = delta_s.get(t);
      double[][] G_t = GS.get(t);
      double[][] h_prime_t = h_prime_s.get(t);
      double[][] mu_hat_t = mu_hat_s.get(t);

      if (t != 0) {
	int n_pre = NS.get(t-1);
	double[][] A_pre_t = AS.get(t-1);
	double[][] h_prime_pre_t = h_prime_s.get(t-1);
	double[] mu_hat_pre_t = Operations.translate(mu_hat_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));
	double[] weighted_neighbors_tmp = new double[n_pre];
	for (int i = 0; i < n_pre; i++) for (int j = 0; j < n_pre; j++) {
	  weighted_neighbors_tmp[i] += A_pre_t[i][j] * h_prime_pre_t[j][0];
	}
	double[] weighted_neighbors = Operations.translate(weighted_neighbors_tmp, id_map_s.get(t-1), id_map_s.get(t));

	/* TODO: check whether we can save computation by comparing s and t */
	for (int s = 0; s < T-t0; s++) {
	  double grad_hat_t = grad_mu_hat_s.get(t * (T-t0) + s);
	  double grad_hat_pre_t = Operations.translate(grad_mu_hat_s.get((t-1) * (T-t0) + s), id_map_s.get(t-1), id_map_s.get(t));

	  for (int i = 0; i < n; i++) {
	    double n_it = 0;
	    for (int j = 0; j < n; j++) n_it += G_t[i][j];

	    /* first term */
	    double gi1 = -(mu_hat_t[i][0] - (1-lambda) * mu_hat_pre_t[i][0] - lambda * weighted_neighbors[i])
	      * (grad_hat_t[i][0] - (1-lambda) * grad_hat_pre_t[i][0]) / (sigma * sigma);
	    tmp_grad_h_hat_s[s][i] += gi1;

	    /* second term */
	    double gi2 = 0;
	    double weighted_exp_num = 0, weighted_exp_den = 0;
	    for (int j = 0; j < NEG; j++) {
	      int l = neg_samples.get(t)[i][j];
	      double hpl = h_prime_t[l][0];
	      double muit = mu_hat_t[i][0];
	      double e = Math.exp(hpl * muit + 0.5 * hpl * hpl * delta_t * delta_t);
	      /* TODO: check if e explodes */
	      if (Double.isNaN(e)) {
		System.out.println("ERROR2");
		Scanner sc = new Scanner(System.in);
		int gu; gu = sc.nextInt();
	      }

	      weighted_exp_num += hpl * e;
	      weighted_exp_den += e;
	    }
	    double weighted_exp = weighted_exp_num / weighted_exp_den;
	    for (int j = 0; j < n; j++) {
	      gi2 += G_t[i][j] * grad_hat_t[i][0] * (h_prime_t[j][0] - weighted_exp);
	    }
	    tmp_grad_h_hat_s[s][i] += gi2;
	  }
	}
      } else {
	for (int s = 0; s < T-t0; s++) {
	  double[][] grad_hat_t = grad_mu_hat_s.get(t * (T-t0) + s);

	  for (int i = 0; i < n; i++) {
	    double n_it = 0;
	    for (int j = 0; j < n; j++) n_it += G_t[i][j];

	    /* first term */
	    double gi1 = -mu_hat_t[i][0] * grad_hat_t[i][0] / (sigma * sigma);
	    tmp_grad_h_hat_s[s][i] += gi1;

	    /* second term */
	    double gi2 = 0;
	    double weighted_exp_num = 0, weighted_exp_den = 0;
	    for (int j = 0; j < NEG; j++) {
	      int l = neg_samples.get(t)[i][j];
	      double hpl = h_prime_t[l][0];
	      double muit = mu_hat_t[i][0];
	      double e = Math.exp(hpl * muit + 0.5 * hpl * hpl * delta_t * delta_t);
	      /* TODO: check if e explodes */
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
	      gi2 += G_t[i][j] * grad_hat_t[i][0] * (h_prime_t[j][0] - weighted_exp);
	    }
	    tmp_grad_h_hat_s[s][i] += gi2;
	  }
	}
      }
      /* end if-else */
    }

    /* update global gradient */
    for (int t = 0; t < T-t0; t++) {
      int n = NS.get(t);
      double[][] grad = new double[n][1];
      for (int i = 0; i < n; i++) grad[i][0] = tmp_grad_h_hat_s[t][i];
      grad_h_hat_s.set(t, grad);
    }
    FileParser.output(grad_h_hat_s, "./grad/grad_" + iteration + ".txt");

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
	int n = NS.get(t);
	double[][] G_t = GS.get(t);
	double[][] h_t = h_s.get(t);
	double[][] h_pre_t = Operations.translate(h_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));
	double[] mu_hat_prime_t = mu_hat_prime_s.get(t);
	double[] mu_hat_prime_pre_t = Operations.translate(mu_hat_prime_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));
	double delta_t = delta_s.get(t);
	int[][] neg_sam_t = neg_samples.get(t);

	for (int i = 0; i < n; i++) {
	  /* first term */
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
	  /* second term */
	  double weighted_neighbor = 0, degree = 0;
	  for (int j = 0; j < n; j++) if (G_t[i][j] != 0) {
	    weighted_neighbor += mu_hat_prime_pre_t[j];
	    degree += 1;
	  }
	  if (degree != 0) weighted_neighbor /= degree;
	  double diff = h_t[i][0] - (1-lambda) * h_pre_t[i][0] - lambda * weighted_neighbor;
	  res -= 0.5 * diff * diff / (sigma*sigma);
	  /* third term */
	  double diff_3 = mu_hat_prime_t[i] - mu_hat_prime_pre_t[i];
	  res -= 0.5 * diff_3 * diff_3 / (sigma*sigma);
	}
      } else {
	int n = NS.get(t);
	double[][] G_t = GS.get(t);
	double[][] h_t = h_s.get(t);
	double[] mu_hat_prime_t = mu_hat_prime_s.get(t);
	double delta_t = delta_s.get(t);
	int[][] neg_sam_t = neg_samples.get(t);

	for (int i = 0; i < n; i++) {
	  /* first term */
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
      }
    }
    return res;
  }

  public static void compute_gradient2() {
    double[][] tmp_grad_h_hat_prime_s = new double[T-t0][1000];

    /* 
     * compute \sum_{j} { n_{ij}^{t} h'_{j}^{t} }
     * and \sum_{j} { n_{ij} }
     */
    double[][] nti = new double[T-t0][1000];
    double[][] nti_hp = new double[T-t0][1000];
    for (int t = 0; t < T-t0; t++) {
      int n = NS.get(t);
      double[][] G_t = GS.get(t);
      double[][] h_t = h_s.get(t);  // h^{t}
      for (int i = 0; i < n; i++) {
	double sum1 = 0, sum2 = 0;
	for (int j = 0; j < n; j++) {
	  sum1 += G_t[i][j];
	  sum2 += G_t[i][j] * h_t[i][0];
	}
	nti[t][i] = sum1;
	nti_hp[t][i] = sum2;
      }
    }

    for (int t = 0; t < T-t0; t++) {
      int n = NS.get(t);
      double delta_t = delta_prime_s.get(t);
      double[][] h_t = h_s.get(t);    // h^{t}
      double[][] h_hat_prime_t = h_hat_prime_s.get(t);	// \hat{h}^{t}
      double[][] mu_hat_t = mu_hat_s.get(t);	// \hat{\mu}^{t}
      double[][] h_prime_t = h_prime_s.get(t);

      if (t != 0) {
	int n_pre = NS.get(t-1);
	double[][] G_pre_t = Operations.translate(GS.get(t-1), id_map_s.get(t-1), id_map_s.get(t));   // G^{t-1}
	double[][] A_pre_t = Operations.translate(AS.get(t-1), id_map_s.get(t-1), id_map_s.get(t));   // A^{t-1}
	double[][] h_pre_t = Operations.translate(h_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));  // h^{t-1}
	double[][] h_hat_prime_pre_t = Operations.translate(h_hat_prime_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));  // \hat{h}^{t-1}
	double[][] mu_hat_pre_t = Operations.translate(mu_hat_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));  // \hat{\mu}^{t-1}

	for (int s = 0; s < T-t0; s++) {
	  double[] grad_mu_hat_prime_t = grad_mu_hat_prime_s.get(t * (T-t0) + s);
	  for (int i = 0; i < n; i++) {
	    /* first term */
	    double g1 = nti_hp[t][i] * grad_mu_hat_prime_t[i];
	    tmp_grad_h_hat_prime_s[s][i] += g1;
	    
	    /* second term */
	    double g2 = 0;
	    for (int _j = 0; _j < NEG; _j++) {
	      double weighted_exp_num = 0, weighted_exp_den = 0;
	      int j = neg_samples.get(t)[i][_j];
	      double htj = h_t[j][0]; double muhti = mu_hat_t[i][0];
	      weighted_exp_num += htj * Math.exp(htj * muhti + 0.5 * htj * htj * delta_t * delta_t);
	      for (int _k = 0; _k < NEG; _k++) {
		int k = neg_samples.get(t)[i][_k];
		double muhtk = mu_hat_t[k][0];
		weighted_exp_den += Math.exp(htj * muhtk + 0.5 * htj * htj * delta_t * delta_t);
	      }
	      g2 -= nti[t][j] * weighted_exp_num / weighted_exp_den * grad_mu_hat_prime_t[i];
	    }
	    tmp_grad_h_hat_prime_s[s][i] += g2;

	    /* third term */
	    double g3 = 0;
	    for (int j = 0; j < n; j++) if (G_pre_t[j][i] != 0) {
	      g3 += (h_t[j][0] - (1-lambda) * h_pre_t[j][0] - lambda * A_pre_t[j][i] * mu_hat_pre_t[i][0]);
	    }
	    g3 /= (sigma * sigma);
	    tmp_grad_h_hat_prime_s[s][i] += g3;
	  }

	  /* fourth term (if any) */
	  if (s == t) for (int i = 0; i < n; i++) {
	    double g4 = -(h_hat_prime_t[i][0] - h_hat_prime_pre_t[i][0]) / (sigma*sigma);
	    tmp_grad_h_hat_prime_s[s][i] += g4;
	  }
	  else if (s == t-1) for (int i = 0; i < n; i++) {
	    double g4 = (h_hat_prime_t[i][0] - h_hat_prime_pre_t[i][0]) / (sigma*sigma);
	    tmp_grad_h_hat_prime_s[s][i] += g4;
	  }
	}
      } else {
	for (int s = 0; s < T-t0; s++) {
	  double[] grad_mu_hat_prime_t = grad_mu_hat_prime_s.get(t * (T-t0) + s);
	  for (int i = 0; i < n; i++) {
	    /* first term */
	    double g1 = nti_hp[t][i] * grad_mu_hat_prime_t[i];
	    tmp_grad_h_hat_prime_s[s][i] += g1;
	    
	    /* second term */
	    double g2 = 0;
	    for (int _j = 0; _j < NEG; _j++) {
	      double weighted_exp_num = 0, weighted_exp_den = 0;
	      int j = neg_samples.get(t)[i][_j];
	      double htj = h_t[j][0]; double muhti = mu_hat_t[i][0];
	      weighted_exp_num += htj * Math.exp(htj * muhti + 0.5 * htj * htj * delta_t * delta_t);
	      for (int _k = 0; _k < NEG; _k++) {
		int k = neg_samples.get(t)[i][_k];
		double muhtk = mu_hat_t[k][0];
		weighted_exp_den += Math.exp(htj * muhtk + 0.5 * htj * htj * delta_t * delta_t);
	      }
	      g2 -= nti[t][j] * weighted_exp_num / weighted_exp_den * grad_mu_hat_prime_t[i];
	    }
	    tmp_grad_h_hat_prime_s[s][i] += g2;
	  }

	  /* fourth term (if any) */
	  if (s == t) for (int i = 0; i < n; i++) {
	    double g4 = -h_hat_prime_t[i][0] / (sigma*sigma);
	    tmp_grad_h_hat_prime_s[s][i] += g4;
	  }
	}
      }

    }

    /* update global gradient */
    for (int t = 0; t < T-t0; t++) {
      int n = NS.get(t);
      double[][] grad = new double[n][1];
      for (int i = 0; i < n; i++) grad[i][0] = tmp_grad_h_hat_prime_s[t][i];
      grad_h_hat_prime_s.set(t, grad);
    }

    return;
  }

  /** 
   * forward pass 1: update intrinsic features 
   *  (1) mu (mu_s) 
   *  (2) grad_mu (grad_mu_s) 
   *  (3) variance V (v_s)
   */
  public static void forward1(boolean update_grad) {
    for (int t1 = t0; t1 < T; t1++) {
      int t = t1-t0;
//      System.out.println("forward 1;\tt = " + t1);
      if (t != 0) {
	int n_pre = NS.get(t-1);	      // N_{t-1}
	int n = NS.get(t);		      // N_t
	double delta_t = delta_s.get(t);      // delta_t
	double[] mu_pre_t = Operations.translate(mu_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));  // mu^{t-1} (N*1)
	double V_pre_t = v_s.get(t-1);
	double[][] h_hat_t = h_hat_s.get(t);    // \hat{h}^t  [t]

	Matrix a = new Matrix(AS.get(t-1));   // A^{t-1}
	Matrix hprime_pre_t = new Matrix(h_prime_s.get(t-1));   // h'^{t-1} 

	/* calculate \mu */
	double[] mu_t = new double[n];
	Matrix ave_neighbors = Operations.translate(a.times(hprime_pre_t), id_map_s.get(t-1), id_map_s.get(t), false);  // n*1
	double factor_1 = delta_t*delta_t / ( delta_t*delta_t + sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t );
	double factor_2 = ( sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t ) 
	  / ( delta_t*delta_t + sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t );
	for (int i = 0; i < n; i++) {
	  mu_t[i] = factor_1 * ( (1-lambda) * mu_pre_t + lambda * ave_neighbors.get(i,0) ) 
	    + factor_2 * h_hat_t[i][0];
	}
	/* calculate V */
	double V_t = ( delta_t*delta_t ) * ( sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t ) 
	  / ( delta_t*delta*t + sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t );

	/* update */
	mu_s.set(t, mu_t);
	v_s.set(t, V_t);

	/* calculate and update grad_mu */
	if (update_grad) for (int s = 0; s < T-t0; s++) {
	  double grad_pre_t_s = grad_mu_s.get( (t-1) * (T-t0) + s );
	  double grad_t_s = factor_1 * (1-lambda) * grad_pre_t_s;
	  if (t == s) {
	    grad_t_s += factor_2;
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
	int n_pre = NS.get(t-1);	      // N_{t-1}
	int n = NS.get(t);		      // N_t
	double delta_t_prime = delta_prime_s.get(t);	      // delta'_t
	double[] mu_prime_pre_t = Operations.translate(mu_prime_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));  // mu'^{t-1}  [t]
	double[][] h_hat_prime_t = h_hat_prime_s.get(t);   // \hat{h}'^{t} (n*1)
	double v_prime_pre_t = v_prime_s.get(t-1);	  // V'^{t-1} [t]

	/* calculate new round of parameters (mu, V) */
	double[] mu_prime_t = new double[n];
	double factor_1 = ( delta_t_prime*delta_t_prime ) / ( delta_t_prime*delta_t_prime + sigma*sigma + (1-lambda)*(1-lambda)*V_prime_pre_t );
	double factor_2 = ( sigma*sigma + (1-lambda)*(1-lambda)*V_prime_pre_t ) 
	  / ( delta_t_prime*delta_t_prime + sigma*sigma + (1-lambda)*(1-lambda)*V_prime_pre_t );
	for (int i = 0; i < n; i++) {
	  mu_prime_t[i] = factor_1 * mu_prime_pre_t[i] + factor_2 * h_hat_prime_t[i][0];
	}
	double v_prime_t = ( delta_t_prime*delta_t_prime ) * ( V_prime_pre_t + sigma*sigma ) 
	  / ( delta_t_prime*delta_t_prime + V_prime_pre_t + sigma*sigma );

	/* update */
	mu_prime_s.set(t, mu_prime_t);
	v_prime_s.set(t, v_prime_t);

	/* calculate and update gradient */
	if (update_grad) for (int s = 0; s < T-t0; s++) {
	  double grad_mu_prime_pre_t_s = grad_mu_prime_s.get( (t-1) * (T-t0) + s );
	  double grad_mu_prime_t_s = factor_1 * grad_mu_prime_pre_t_s;
	  if (s == t) {
	    grad_mu_prime_t_s += factor_2;
	  }
	  grad_mu_prime_s.set( t * (T-t0) + s, grad_mu_prime_t_s );
	}
      } else {
	/* for \mu'_i: ignore the first term in the summation */
	/* for V': 
	 *    option 1: use v_prime_init for all i
	 *    option 2: ignore v_prime_{t-1} 
	 */
	int n = NS.get(t);
	double delta_t_prime = delta_prime_s.get(t);	      // delta'_t
	double[][] h_hat_prime_t = h_prime_s.get(t);   // \hat{h}'^{t} (n*1)
	/* calculate new round of parameters */
	double[] mu_prime_t = new double[n];
	double[] v_prime_t = new double[n];
	for (int i = 0; i < n; i++) {
	  double c = (sigma*sigma) / (sigma*sigma + delta_t_prime*delta_t_prime);
	  mu_prime_t[i] = c * h_hat_prime_t[i][0];
	  v_prime_t[i] = v_prime_init;
	}
	/* update */
	mu_prime_s.set(t, mu_prime_t);
	v_prime_s.set(t, v_prime_t);
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
	int n_pre = NS.get(t-1);    // N_{t-1}
	double rat = (1.0-lambda)/sigma/sigma;
	double V_pre_t = v_s.get(t-1);	// V^{t-1}
	double V_hat_t = v_hat_s.get(t);  // \hat{V}^{t}
	Matrix K_pre_t = (V_pre_t.inverse().plus(Matrix.identity(n_pre, n_pre).times(rat*rat))).inverse();	// K^{t-1}
	Matrix A_pre_t  = new Matrix(AS.get(t-1));	// A^{t-1}
	Matrix hprime_pre_t = new Matrix(h_prime_s.get(t-1));   // h'^{t-1} 
	Matrix mu_pre_t = new Matrix(mu_s.get(t-1));	// \mu^{t-1}
	Matrix mu_hat_t = Operations.translate(new Matrix(mu_hat_s.get(t)), id_map_s.get(t), id_map_s.get(t-1), false);   // \hat{\mu}^{t}  [t-1]

	Matrix ave_neighbors = A_pre_t.times(hprime_pre_t);  // n_pre * 1

	/* calculate \hat{\mu}^{t-1} */
	double factor_1 = (1-lambda)*(1-lambda) * V_pre_t / ( sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t );
	double factor_2 = ( sigma*sigma ) / ( sigma*sigma + (1-lambda)*(1-lambda)*V_pre_t );
	double[] mu_hat_pre_t = new double[n_pre];
	for (int i = 0; i < n_pre; i++) {
	  mu_hat_pre_t[i] = factor_1 * (mu_hat_t[i] - lambda * ave_neighbors.get(i,0)) 
	    + factor_2 * mu_pre_t[i];
	}
	/* calculate \hat{V}^{t-1} */
	double V_hat_pre_t = V_pre_t + factor_1 * factor_1 * ( V_hat_t - V_pre_t - (sigma*sigma) / ((1-lambda)*(1-lambda)) );

	/* update */
	mu_hat_s.set(t-1, mu_hat_pre_t);
	v_hat_s.set(t-1, V_hat_pre_t);

	/* calculate and update \hat{grad_mu}^{t-1} */
	if (update_grad) for (int s = 0; s < T-t0; s++) {
	  double grad_hat_t_s = grad_mu_hat_s.get( t * (T-t0) + s );
	  double grad_pre_t_s = grad_mu_s.get( (t-1) * (T-t0) + s );
	  double grad_hat_pre_t_s = factor_1 * grad_hat_t_s + factor_2 * grad_pre_t_s;
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
	  grad_mu_hat_s.set(t * (T-t0) + s, grad_mu_s.get(t * (T-t0) + s));
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
	int n_pre = NS.get(t-1);    // N_{t-1}
	double[] mu_prime_pre_t = mu_prime_s.get(t-1);	  // \mu'^{t-1}  [t-1]
	double[] mu_hat_prime_t = Operations.translate(mu_hat_prime_s.get(t), id_map_s.get(t), id_map_s.get(t-1));  // \hat{\mu}'^{t}  [t-1]
	double V_prime_pre_t = v_prime_s.get(t-1);	  // V'^{t-1} 
	double V_hat_prime_t = v_hat_prime_s.get(t);	  // \hat{V}'{t}

	/* calculate the next round (t-1) of parameters (mu, V) */
	double[] mu_hat_prime_pre_t = new double[n_pre];
	double factor_1 = ( sigma*sigma ) / ( V_prime_pre_t + sigma*sigma );
	double factor_2 = ( V_prime_pre_t ) / ( V_prime_pre_t + sigma*sigma );
	for (int i = 0; i < n_pre; i++) {
	  mu_hat_prime_pre_t[i] = factor_1 * mu_prime_pre_t[i] + factor_2 * mu_hat_prime_t[i];
	}
	double v_hat_prime_pre_t = V_prime_pre_t 
	  + factor_2 * factor_2 * (V_hat_prime_t - V_prime_pre_t - sigma*sigma);

	/* update */
	mu_hat_prime_s.set(t-1, mu_hat_prime_pre_t);
	v_hat_prime_s.set(t-1, v_hat_prime_pre_t);

	/* calculate and update gradient */
	if (update_grad) for (int s = 0; s < T-t0; s++) {
	  double grad_mu_prime_pre_t_s = grad_mu_prime_s.get( (t-1) * (T-t0) + s );
	  double grad_mu_hat_prime_t_s = grad_mu_hat_prime_s.get( t * (T-t0) + s );
	  double grad_mu_hat_prime_pre_t_s = factor_1 * grad_mu_prime_pre_t_s + factor_2 * grad_mu_hat_prime_t_s;
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
	  grad_mu_hat_prime_s.set(t * (T-t0) + s, grad_mu_prime_s.get(t * (T-t0) + s));
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
    test1();
  }

}



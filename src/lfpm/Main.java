import java.io.*;
import java.util.*;
import Jama.*;

public class Main {
  public static int t0 = 100;
  public static int T = 110;
  public static double lambda = 0.5;
  public static double sigma = 0.3;
  public static double delta = 0.3;
  public static double scale = 0.2;
  public static double eps = 1e-6;   // avoid sigular 
  public static Random rand = new Random(0);

  /* global data */
  public static List<Integer> NS = new ArrayList<Integer>(T);	      // number of users
  public static List<double[][]> GS = new ArrayList<double[][]>(T);   // graph
  public static List<double[][]> AS = new ArrayList<double[][]>(T);   // adjacency matrix (off-diagonal) 
  public static List<Map<Integer, Integer>> id_map_s = new ArrayList<Map<Integer, Integer>>(T);	  // id map: global ID -> local ID
  public static List<Double> delta_s = new ArrayList<Double>(T);
  public static List<Double> delta_prime_s = new ArrayList<Double>(T);

  /* intrinsic features */
  public static List<double[][]> h_s = new ArrayList<double[][]>(T);	  // h: latent attribute
  public static List<double[][]> h_hat_s = new ArrayList<double[][]>(T);  // \hat{h}: variational parameter
  public static List<double[][]> mu_s = new ArrayList<double[][]>(T);	  // \mu: forward mean
  public static List<double[][]> v_s = new ArrayList<double[][]>(T);	  // V: forward variance
  public static List<double[][]> mu_hat_s = new ArrayList<double[][]>(T); // \hat{\mu}: backward mean
  public static List<double[][]> v_hat_s = new ArrayList<double[][]>(T);  // \hat{V}: backward variance 

  /* impression features */
  public static List<double[][]> h_prime_s = new ArrayList<double[][]>(T);	// h': latent attribute
  public static List<double[][]> h_hat_prime_s = new ArrayList<double[][]>(T);  // \hat{h}': variational parameter
  public static List<double[]> mu_prime_s = new ArrayList<double[]>(T);	// \mu': forward mean
  public static List<double[]> v_prime_s = new ArrayList<double[]>(T);	// V': forward variance (diagnoal)
  public static List<double[]> mu_hat_prime_s = new ArrayList<double[]>(T);	// \hat{\mu}': backward mean
  public static List<double[]> v_hat_prime_s = new ArrayList<double[]>(T);	// \hat{V}': backward variance
  public static double v_prime_init = 0.01;	// init variance for every h'

  /* gradients */
  public static List<double[][]> grad_mu_s = new ArrayList<double[][]>(T);	// grad of \mu: forward
  public static List<double[][]> grad_mu_hatS = new ArrayList<double[][]>(T);	// grad of \hat{\mu}: backward

  public static void test1() {
    /* read, init data & parameters */
    for (int t = t0; t < T; t++) {
      String fileDir = "../../data/graph/" + Integer.toString(t) + ".csv";
      Map<Integer, Integer> idMap = new HashMap<Integer, Integer>();
      Map<Integer, Integer> idMapInv = new HashMap<Integer, Integer>();
      Map<Integer, Double> freq = FileParser.readCSVDict(fileDir, idMap, idMapInv);
      id_map_s.add(idMap);

      int n = idMap.size();
      NS.add(n);
      double[][] G = new double[n][n];
      double[][] A = new double[n][n];
      double[][] mu = new double[n][1];
      double[][] mu_hat = new double[n][1];
      double[][] H = new double[n][1];
      double[][] h_prime = new double[n][1];
      double[][] h_hat = new double[n][1];
      FileParser.readCSVGraph(fileDir, freq, idMap, G, A);
      for (int i = 0; i < n; i++) {
	mu[i][0] = scale * (rand.nextDouble() - 0.5);
	mu_hat[i][0] = scale * (rand.nextDouble() - 0.5);
	H[i][0] = scale * (rand.nextDouble() - 0.5);
	h_prime[i][0] = scale * (rand.nextDouble() - 0.5);
	h_hat[i][0] = scale * (rand.nextDouble() - 0.5);
      }

      GS.add(G); AS.add(A);
      mu_s.add(mu); mu_hat_s.add(mu); h_s.add(H); h_prime_s.add(h_prime);

      // for test
      delta_s.add(0.1);
      delta_prime_s.add(0.1);
      double[][] v = new double[n][n];
      double[][] v_hat = new double[n][n];
      double[] v_prime = new double[n];
      double[][] grad_h = new double[n][1];
      double[][] grad_h_hat = new double[n][1];
      for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
	  v[i][j] = 0.01 * scale * (rand.nextDouble() - 0.5);
	  v_hat[i][j] = 0.01 * scale * (rand.nextDouble() - 0.5);
	}
	v_prime[i] = scale * rand.nextDouble();
      }
      double[] mu_prime = new double[n];
      for (int i = 0; i < n; i++) mu_prime[i] = scale * (rand.nextDouble() - 0.5);
      Matrix mat_V = new Matrix(v);
      Matrix vtv = mat_V.transpose().times(mat_V).plus(Matrix.identity(n, n).times(eps));
      Matrix vtvi = vtv.inverse();
      v_s.add(vtv.getArray()); h_hat_s.add(h_hat);
      v_hat_s.add(v_hat); 
      grad_mu_s.add(grad_h); grad_mu_hatS.add(grad_h_hat);
      mu_prime_s.add(mu_prime); mu_hat_prime_s.add(mu_prime);
      v_prime_s.add(v_prime); v_hat_prime_s.add(v_prime);

      System.out.println("done! t = " + t);
    }

    /* outer for-loop */
    for (int iter = 0; iter < 10; iter++) {
      forward1();
      backward1();
      /* gradient descent */
      /* inner for-loop here */

      forward2();
      backward2();
      /* gradient descent */
      /* inner for-loop here */
    }
  }

  /* 
   * forward pass 1: update intrinsic features 
   *  (1) mu (mu_s) 
   *  (2) grad_mu (grad_mu_s) 
   *  (3) variance V (v_s)
   */
  public static void forward1() {
    for (int t1 = t0; t1 < T; t1++) {
      int t = t1-t0;
      System.out.println("forward 1;\tt = " + t1);
      if (t != 0) {
	int old_n = NS.get(t-1);	      // N_{t-1}
	int n = NS.get(t);		      // N_t
	double delta_t = delta_s.get(t);	      // delta_t
	Matrix a = new Matrix(AS.get(t-1));   // A^{t-1}
	Matrix mu_pre_t = new Matrix(mu_s.get(t-1));   // mu^{t-1}
	Matrix grad_mu_pre_t = new Matrix(grad_mu_s.get(t-1));   // grad_mu^{t-1}
	Matrix V_pre_t = new Matrix(v_s.get(t-1));     // V^{t-1}
	Matrix h_hat_t = Operations.translate(new Matrix(h_hat_s.get(t)), id_map_s.get(t), id_map_s.get(t-1), false);    // \hat{h}^t  [t-1]
	Matrix hprime_pre_t = new Matrix(h_prime_s.get(t-1));   // h'^{t-1} 
	Matrix iplusv = Matrix.identity(old_n, old_n).times(delta*delta).plus(V_pre_t.times((1-lambda)*(1-lambda)));
	Matrix Sigma = (iplusv.inverse().plus(Matrix.identity(old_n, old_n).times(1.0/delta_t/delta_t))).inverse();   // \Sigma  [t-1]

	/* calculate \mu */
	Matrix wsum = mu_pre_t.times(1-lambda).plus(a.times(hprime_pre_t).times(lambda));	// for \mu
	Matrix addi = Sigma.times(h_hat_t.minus(wsum)).times(1.0/delta_t/delta_t);	// for \mu
	Matrix mu_t = Operations.translate(wsum.plus(addi), id_map_s.get(t-1), id_map_s.get(t), true);	    // mu^t  [t]
	/* calculate grad_mu */
	Matrix g_wsum = grad_mu_pre_t.times(1-lambda).plus(a.times(hprime_pre_t).times(lambda));  // for grad_mu
	Matrix g_addi = Sigma.times(h_hat_t.minus(g_wsum)).times(1.0/delta_t/delta_t);	// for grad_mu
	Matrix grad_mu_t = Operations.translate(g_wsum.plus(g_addi), id_map_s.get(t-1), id_map_s.get(t), true); // grad_mu^t  [t]
	/* calculate V */
	Matrix V_t = Operations.translate(Sigma, id_map_s.get(t-1), id_map_s.get(t), false);   // V^t  [t]
	V_t = V_t.plus(Matrix.identity(n, n).times(eps));   // make it invertible

	/* update */
	mu_s.set(t, mu_t.getArray());
	grad_mu_s.set(t, grad_mu_t.getArray());
	v_s.set(t, V_t.getArray());

	System.out.println(mu_t.getRowDimension());
      } else {
	// random init: keep unchanged
      }
      Scanner sc = new Scanner(System.in);
//      int gu; gu = sc.nextInt();
      /* end for each t */
    }
  }

  /* 
   * forward pass 2: update impression features 
   *  (1) mu' (mu_prime_s) 
   *  (2) variance V' (v_prime_s)
   */
  public static void forward2() {
    for (int t1 = t0; t1 < T; t1++) {
      int t = t1-t0;
      System.out.println("forward 2;\tt = " + t1);
      if (t != 0) {
	int old_n = NS.get(t-1);	      // N_{t-1}
	int n = NS.get(t);		      // N_t
	double delta_t_prime = delta_prime_s.get(t);	      // delta'_t
	double[] mu_prime_pre_t = Operations.translate(mu_prime_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));  // mu'^{t-1}  [t]
	double[][] h_hat_prime_t = h_hat_prime_s.get(t);   // \hat{h}'^{t} (n*1)
	double[] v_prime_pre_t = Operations.translate(v_prime_s.get(t-1), id_map_s.get(t-1), id_map_s.get(t));	  // V'^{t-1} (diagnoal)  [t]
	/* calculate new round of parameters */
	double[] mu_prime_t = new double[n];
	double[] v_prime_t = new double[n];
	for (int i = 0; i < n; i++) {
	  double denominator = v_prime_pre_t[i] + sigma*sigma + delta_t_prime*delta_t_prime;
	  mu_prime_t[i] = delta_t_prime * delta_t_prime * mu_prime_pre_t[i] + (v_prime_pre_t[i] + sigma*sigma) * h_hat_prime_t[i][0];
	  mu_prime_t[i] /= denominator;
	  v_prime_t[i] = delta_t_prime * delta_t_prime * (v_prime_pre_t[i] + sigma*sigma) / denominator;
	}
	/* update */
	mu_prime_s.set(t, mu_prime_t);
	v_prime_s.set(t, v_prime_t);
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
   *  (2) \hat{grad_mu} (grad_mu_hatS)
   *  (3) \hat{V} (v_hat_s)
   */
  public static void backward1() {
    for (int t1 = T-1; t1 > t0; t1--) {
      int t = t1-t0;
      System.out.println("backward 1;\tt = " + t1);
      if (t != T-1-t0) {
	int n_pre_t = NS.get(t-1);    // N_{t-1}
	double rat = (1.0-lambda)/sigma/sigma;
	Matrix V_pre_t = new Matrix(v_s.get(t-1));	// V^{t-1}
	Matrix K_pre_t = (V_pre_t.inverse().plus(Matrix.identity(n_pre_t, n_pre_t).times(rat*rat))).inverse();	// K^{t-1}
	Matrix A_pre_t  = new Matrix(AS.get(t-1));	// A^{t-1}
	Matrix hprime_pre_t = new Matrix(h_prime_s.get(t-1));   // h'^{t-1} 
	Matrix mu_pre_t = new Matrix(mu_s.get(t-1));	// \mu^{t-1}
	Matrix grad_mu_pre_t = new Matrix(grad_mu_s.get(t-1));	// grad_mu^{t-1}
	Matrix mu_hat_t = Operations.translate(new Matrix(mu_hat_s.get(t)), id_map_s.get(t), id_map_s.get(t-1), false);   // \hat{\mu}^{t}  [t-1]
	Matrix grad_mu_hat_t = Operations.translate(new Matrix(grad_mu_hatS.get(t)), id_map_s.get(t), id_map_s.get(t-1), false);   // \hat{grad_mu}^{t}  [t-1]

	/* calculate \hat{\mu}^{t-1} */
	Matrix add1 = mu_hat_t.minus(A_pre_t.times(hprime_pre_t));
	Matrix add2 = V_pre_t.inverse().times(mu_pre_t).times(sigma*sigma/(1-lambda));
	Matrix mu_hat_pre_t = K_pre_t.times(add1.plus(add2)).times((1-lambda)/sigma/sigma);
	/* calculate \hat{grad_mu}^{t-1} */
	Matrix g_add1 = grad_mu_hat_t.minus(A_pre_t.times(hprime_pre_t));
	Matrix g_add2 = V_pre_t.inverse().times(grad_mu_pre_t).times(sigma*sigma/(1-lambda));
	Matrix grad_mu_hat_pre_t = K_pre_t.times(g_add1.plus(g_add2)).times((1-lambda)/sigma/sigma);
	/* calculate \hat{V}^{t-1} */
	Matrix V_hat_t = Operations.translate(new Matrix(v_hat_s.get(t)), id_map_s.get(t), id_map_s.get(t-1), false);	    // \hat{V}^{t}  [t-1]
	Matrix kvkt = K_pre_t.times(V_hat_t).times(K_pre_t.transpose());
	Matrix V_hat_pre_t = K_pre_t.plus(kvkt.times(rat*rat));

	/* update */
	mu_hat_s.set(t-1, mu_hat_pre_t.getArray());
	grad_mu_hatS.set(t-1, grad_mu_hat_pre_t.getArray());
	v_hat_s.set(t-1, V_hat_pre_t.getArray());
	System.out.println(mu_hat_t.getRowDimension());
      } else {
	/* 
	 * initial condition for backward pass: 
	 *  (1) \hat{mu}^{T} = mu^{T}
	 *  (2) \hat{grad_mu}^{T} = grad_mu^{T}
	 *  (3) \hat{V}^{T} = V^{T}
	 */
	mu_hat_s.set(t, mu_s.get(t));
	grad_mu_hatS.set(t, grad_mu_s.get(t));
	v_hat_s.set(t, v_s.get(t));
      }
      Scanner sc = new Scanner(System.in);
//      int gu; gu = sc.nextInt();
      /* end for each t */
    }
  }

  /* 
   * backward pass 2: update impression features
   *  (1) \hat{mu} (mu_hat_s) 
   *  (2) \hat{V} (v_hat_s)
   */
  public static void backward2() {
    double c = (1-lambda)*(1-lambda) / (sigma*sigma);
    for (int t1 = T-1; t1 > t0; t1--) {
      int t = t1-t0;
      System.out.println("backward 2;\tt = " + t1);
      if (t != T-1-t0) {
	int n_pre_t = NS.get(t-1);    // N_{t-1}
	double[] mu_prime_pre_t = mu_prime_s.get(t-1);	  // \mu'^{t-1}  [t-1]
	double[] mu_hat_prime_t = Operations.translate(mu_hat_prime_s.get(t), id_map_s.get(t), id_map_s.get(t-1));  // \hat{\mu}'^{t}  [t-1]
	double[] v_prime_pre_t = v_prime_s.get(t-1);	  // V'^{t-1}  [t-1]
	double[] v_hat_prime_t = Operations.translate(v_hat_prime_s.get(t), id_map_s.get(t), id_map_s.get(t-1));    // \hat{V}'{t}  [t-1]
	/* calculate the next round (t-1) of parameters */
	double[] mu_hat_prime_pre_t = new double[n_pre_t];
	double[] v_hat_prime_pre_t = new double[n_pre_t];
	for (int i = 0; i < n_pre_t; i++) {
	  double denominator = v_prime_pre_t[i] + sigma*sigma;
	  mu_hat_prime_pre_t[i] = sigma * sigma * mu_prime_pre_t[i] + v_prime_pre_t[i] * mu_hat_prime_t[i];
	  mu_hat_prime_pre_t[i] /= denominator;
	  /* TODO: what if v_hat_prime_t[i] - v_prime_pre_t[t] - sigma*sigma < 0? */
	  v_hat_prime_pre_t[i] = v_prime_pre_t[i] 
	    + (v_prime_pre_t[i] * v_prime_pre_t[i] / denominator / denominator) * (v_hat_prime_t[i] - v_prime_pre_t[t] - sigma*sigma);
	}
	/* update */
	mu_hat_prime_s.set(t-1, mu_hat_prime_pre_t);
	v_hat_prime_s.set(t-1, v_hat_prime_pre_t);
      } else {
	/* 
	 * initial condition for backward pass:
	 *  (1) \hat{\mu}'^{T} = \mu'^{T}
	 *  (2) \hat{V}'^{T} = V'^{T}
	 */
	mu_hat_prime_s.set(t, mu_prime_s.get(t));
	v_hat_prime_s.set(t, v_prime_s.get(t));
      }
      /* end for each t */
    }
  }
	
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



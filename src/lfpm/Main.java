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
  public static double eps = 0.0001;   // avoid sigular 
  public static Random rand = new Random(0);

  public static List<Integer> NS = new ArrayList<Integer>(T);
  public static List<double[][]> GS = new ArrayList<double[][]>(T);
  public static List<double[][]> AS = new ArrayList<double[][]>(T);
  public static List<double[][]> HS = new ArrayList<double[][]>(T);	      // h
  public static List<double[][]> HPrimeS = new ArrayList<double[][]>(T);      // h'
  public static List<double[][]> HHatS = new ArrayList<double[][]>(T);	      // \hat{h}
  public static List<double[][]> HHatPrimeS = new ArrayList<double[][]>(T);   // \hat{h}'

  /* intrinsic features */
  public static List<double[][]> muS = new ArrayList<double[][]>(T);    // forward
  public static List<double[][]> VS = new ArrayList<double[][]>(T);	// V^{(t)} 

  /* impression features */
  public static List<double[][]> muHatS = new ArrayList<double[][]>(T); // backward
  public static List<double[][]> VHatS = new ArrayList<double[][]>(T);	// \hat{V}^{(t)} 

  /* gradients */
  public static List<double[][]> grad_muS = new ArrayList<double[][]>(T);
  public static List<double[][]> grad_mu_hatS = new ArrayList<double[][]>(T);

  public static List<Map<Integer, Integer>> idMapS = new ArrayList<Map<Integer, Integer>>(T);
  public static List<Double> deltaS = new ArrayList<Double>(T);

  public static void test1() {
    // read, init data & parameters
    for (int t = t0; t < T; t++) {
      String fileDir = "../../data/graph/" + Integer.toString(t) + ".csv";
      Map<Integer, Integer> idMap = new HashMap<Integer, Integer>();
      Map<Integer, Integer> idMapInv = new HashMap<Integer, Integer>();
      Map<Integer, Double> freq = FileParser.readCSVDict(fileDir, idMap, idMapInv);
      idMapS.add(idMap);

      int n = idMap.size();
      NS.add(n);
      double[][] G = new double[n][n];
      double[][] A = new double[n][n];
      double[][] mu = new double[n][1];
      double[][] muHat = new double[n][1];
      double[][] H = new double[n][1];
      double[][] HPrime = new double[n][1];
      double[][] h_hat = new double[n][1];
      FileParser.readCSVGraph(fileDir, freq, idMap, G, A);
      for (int i = 0; i < n; i++) {
	mu[i][0] = scale * (rand.nextDouble() - 0.5);
	muHat[i][0] = scale * (rand.nextDouble() - 0.5);
	H[i][0] = scale * (rand.nextDouble() - 0.5);
	HPrime[i][0] = scale * (rand.nextDouble() - 0.5);
	h_hat[i][0] = scale * (rand.nextDouble() - 0.5);
      }

      GS.add(G); AS.add(A);
      muS.add(mu); muHatS.add(mu); HS.add(H); HPrimeS.add(HPrime);
      // for test
      deltaS.add(0.1);
      double[][] V = new double[n][n];
      double[][] VHat = new double[n][n];
      double[][] grad_h = new double[n][1];
      double[][] grad_h_hat = new double[n][1];
      for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
	V[i][j] = 0.01 * scale * (rand.nextDouble() - 0.5);
	VHat[i][j] = 0.01 * scale * (rand.nextDouble() - 0.5);
      }
      Matrix mat_V = new Matrix(V);
      Matrix vtv = mat_V.transpose().times(mat_V).plus(Matrix.identity(n, n).times(eps));
      Matrix vtvi = vtv.inverse();
      VS.add(vtv.getArray()); HHatS.add(h_hat);
      VHatS.add(VHat); 
      grad_muS.add(grad_h); grad_mu_hatS.add(grad_h_hat);

      System.out.println("done! t = " + t);
    }

    for (int iter = 0; iter < 10; iter++) {
      forward1();
      backward1();
    }
  }

  /* 
   * forward pass 1: update 
   *  (1) mu (muS) 
   *  (2) grad_mu (grad_muS) 
   *  (3) variance V (VS)
   */
  public static void forward1() {
    for (int t1 = t0; t1 < T; t1++) {
      int t = t1-t0;
      System.out.println("forward\tt = " + t1);
      if (t != 0) {
	int old_n = NS.get(t-1);	      // N_{t-1}
	int n = NS.get(t);		      // N_t
	double delta_t = deltaS.get(t);	      // delta_t
	Matrix a = new Matrix(AS.get(t-1));   // A^{t-1}
	Matrix mu_pre_t = new Matrix(muS.get(t-1));   // mu^{t-1}
	Matrix grad_mu_pre_t = new Matrix(grad_muS.get(t-1));   // grad_mu^{t-1}
	Matrix V_pre_t = new Matrix(VS.get(t-1));     // V^{t-1}
	Matrix h_hat_t = Operations.translate(new Matrix(HHatS.get(t)), idMapS.get(t), idMapS.get(t-1), false);    // \hat{h}^t  [t-1]
	Matrix hprime_pre_t = new Matrix(HPrimeS.get(t-1));   // h'^{t-1} 
	Matrix iplusv = Matrix.identity(old_n, old_n).times(delta*delta).plus(V_pre_t.times((1-lambda)*(1-lambda)));
	Matrix Sigma = (iplusv.inverse().plus(Matrix.identity(old_n, old_n).times(1.0/delta_t/delta_t))).inverse();   // \Sigma  [t-1]

	/* calculate \mu */
	Matrix wsum = mu_pre_t.times(1-lambda).plus(a.times(hprime_pre_t).times(lambda));	// for \mu
	Matrix addi = Sigma.times(h_hat_t.minus(wsum)).times(1.0/delta_t/delta_t);	// for \mu
	Matrix mu_t = Operations.translate(wsum.plus(addi), idMapS.get(t-1), idMapS.get(t), true);	    // mu^t  [t]
	/* calculate grad_mu */
	Matrix g_wsum = grad_mu_pre_t.times(1-lambda).plus(a.times(hprime_pre_t).times(lambda));  // for grad_mu
	Matrix g_addi = Sigma.times(h_hat_t.minus(g_wsum)).times(1.0/delta_t/delta_t);	// for grad_mu
	Matrix grad_mu_t = Operations.translate(g_wsum.plus(g_addi), idMapS.get(t-1), idMapS.get(t), true); // grad_mu^t  [t]
	/* calculate V */
	Matrix V_t = Operations.translate(Sigma, idMapS.get(t-1), idMapS.get(t), false);   // V^t  [t]
	V_t = V_t.plus(Matrix.identity(n, n).times(eps));   // make it invertible

	/* update */
	muS.set(t, mu_t.getArray());
	grad_muS.set(t, grad_mu_t.getArray());
	VS.set(t, V_t.getArray());

	System.out.println(mu_t.getRowDimension());
      } else {
	// random init: keep unchanged
      }
      Scanner sc = new Scanner(System.in);
//      int gu; gu = sc.nextInt();
    }
  }

  /* 
   * backward pass 1: update 
   *  (1) \hat{mu} (muHatS) 
   *  (2) \hat{grad_mu} (grad_mu_hatS)
   *  (3) \hat{V} (VHatS)
   */
  public static void backward1() {
    for (int t1 = T-1; t1 > t0; t1--) {
      int t = t1-t0;
      System.out.println("backward\tt = " + t1);
      if (t != T-1-t0) {
	int n_pre_t = NS.get(t-1);    // N_{t-1}
	double rat = (1.0-lambda)/sigma/sigma;
	Matrix V_pre_t = new Matrix(VS.get(t-1));	// V^{t-1}
	Matrix K_pre_t = (V_pre_t.inverse().plus(Matrix.identity(n_pre_t, n_pre_t).times(rat*rat))).inverse();	// K^{t-1}
	Matrix A_pre_t  = new Matrix(AS.get(t-1));	// A^{t-1}
	Matrix hprime_pre_t = new Matrix(HPrimeS.get(t-1));   // h'^{t-1} 
	Matrix mu_pre_t = new Matrix(muS.get(t-1));	// \mu^{t-1}
	Matrix grad_mu_pre_t = new Matrix(grad_muS.get(t-1));	// grad_mu^{t-1}
	Matrix mu_hat_t = Operations.translate(new Matrix(muHatS.get(t)), idMapS.get(t), idMapS.get(t-1), false);   // \hat{\mu}^{t}  [t-1]
	Matrix grad_mu_hat_t = Operations.translate(new Matrix(grad_mu_hatS.get(t)), idMapS.get(t), idMapS.get(t-1), false);   // \hat{grad_mu}^{t}  [t-1]

	/* calculate \hat{\mu}^{t-1} */
	Matrix add1 = mu_hat_t.minus(A_pre_t.times(hprime_pre_t));
	Matrix add2 = V_pre_t.inverse().times(mu_pre_t).times(sigma*sigma/(1-lambda));
	Matrix mu_hat_pre_t = K_pre_t.times(add1.plus(add2)).times((1-lambda)/sigma/sigma);
	/* calculate \hat{grad_mu}^{t-1} */
	Matrix g_add1 = grad_mu_hat_t.minus(A_pre_t.times(hprime_pre_t));
	Matrix g_add2 = V_pre_t.inverse().times(grad_mu_pre_t).times(sigma*sigma/(1-lambda));
	Matrix grad_mu_hat_pre_t = K_pre_t.times(g_add1.plus(g_add2)).times((1-lambda)/sigma/sigma);
	/* calculate \hat{V}^{t-1} */
	Matrix V_hat_t = Operations.translate(new Matrix(VHatS.get(t)), idMapS.get(t), idMapS.get(t-1), false);	    // \hat{V}^{t}  [t-1]
	Matrix kvkt = K_pre_t.times(V_hat_t).times(K_pre_t.transpose());
	Matrix V_hat_pre_t = K_pre_t.plus(kvkt.times(rat*rat));

	/* update */
	muHatS.set(t-1, mu_hat_pre_t.getArray());
	grad_mu_hatS.set(t-1, grad_mu_hat_pre_t.getArray());
	VHatS.set(t-1, V_hat_pre_t.getArray());
	System.out.println(mu_hat_t.getRowDimension());
      } else {
	/* 
	 * initial condition for backward pass: 
	 *  (1) \hat{mu}^{T} = mu^{T}
	 *  (2) \hat{grad_mu}^{T} = grad_mu^{T}
	 *  (3) \hat{V}^{T} = V^{T}
	 */
	muHatS.set(t, muS.get(t));
	grad_mu_hatS.set(t, grad_muS.get(t));
	VHatS.set(t, VS.get(t));
      }
      Scanner sc = new Scanner(System.in);
//      int gu; gu = sc.nextInt();
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



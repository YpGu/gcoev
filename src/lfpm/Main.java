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
  public static Random rand = new Random(0);

  public static List<Integer> NS = new ArrayList<Integer>(T);
  public static List<double[][]> GS = new ArrayList<double[][]>(T);
  public static List<double[][]> AS = new ArrayList<double[][]>(T);
  public static List<double[][]> muS = new ArrayList<double[][]>(T);    // forward
  public static List<double[][]> muHatS = new ArrayList<double[][]>(T); // backward
  public static List<double[][]> HS = new ArrayList<double[][]>(T);
  public static List<double[][]> HPrimeS = new ArrayList<double[][]>(T);
  public static List<Map<Integer, Integer>> idMapS = new ArrayList<Map<Integer, Integer>>(T);

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
      FileParser.readCSVGraph(fileDir, freq, idMap, G, A);
      for (int i = 0; i < n; i++) {
	mu[i][0] = scale * (rand.nextDouble() - 0.5);
	muHat[i][0] = scale * (rand.nextDouble() - 0.5);
	H[i][0] = scale * (rand.nextDouble() - 0.5);
	HPrime[i][0] = scale * (rand.nextDouble() - 0.5);
      }

      GS.add(G); AS.add(A);
      muS.add(mu); muHatS.add(mu); HS.add(H); HPrimeS.add(HPrime);

      System.out.println("done! t = " + t);
    }

    forward1();
    backward1();
  }

  /* 
   * forward pass: update mu (muS)
   */
  public static void forward1() {
    for (int t1 = t0; t1 < T; t1++) {
      int t = t1-t0;
      System.out.println("forward\tt = " + t1);
      if (t != 0) {
	int old_n = NS.get(t-1);	      // N_{t-1}
	Matrix a = new Matrix(AS.get(t-1));   // A^{t-1}
	Matrix mu_pre_t = new Matrix(muS.get(t-1));   // mu^{t-1}
	Matrix pt = (a.times(a.transpose()).plus(Matrix.identity(old_n, old_n))).times(sigma*sigma);  // P^{t-1}
	Matrix kt = pt.times(pt.plus(Matrix.identity(old_n, old_n).times(delta*delta)).inverse());    // K^{t}
	Matrix h_pre_t = Operations.translate(new Matrix(HS.get(t)), idMapS.get(t), idMapS.get(t-1)); // h^{t} (in style t-1)
	Matrix mu_t = a.times(mu_pre_t).plus(kt.times(h_pre_t.minus(a.times(mu_pre_t))));   // mu^{t} (in style t-1)
	mu_t = Operations.translate(mu_t, idMapS.get(t-1), idMapS.get(t));    // mu^{t} (in style t)

	muS.set(t, mu_t.getArray());	      // update
	for (int i = 0; i < mu_t.getRowDimension(); i++) {
	  System.out.printf("%f ", mu_t.get(i,0));
	}
	System.out.print("\n");
//	System.out.println(mu_t.getRowDimension());
      } else {
	// random init: keep unchanged
      }
      Scanner sc = new Scanner(System.in);
//      int gu; gu = sc.nextInt();
    }
  }

  /*
   * backward pass: update \hat{mu} (muHatS)
   */
  public static void backward1() {
    for (int t1 = T-1; t1 >= t0; t1--) {
      int t = t1-t0;
      System.out.println("backward\tt = " + t1);
      if (t != T-1-t0) {
	int n = NS.get(t);    // N_{t}
	Matrix a = new Matrix(AS.get(t));	// A^{t}
	Matrix jt = a.transpose().times((Matrix.identity(n, n).plus(a.times(a.transpose()))).inverse());    // J^{t}
	Matrix mu_t = new Matrix(muS.get(t));	// mu^{t}
	Matrix mu_next_t = Operations.translate(new Matrix(muHatS.get(t+1)), idMapS.get(t+1), idMapS.get(t));	// \hat{mu}^{t+1}
	Matrix mu_hat_t = mu_t.plus(jt.times(mu_next_t.minus(a.times(mu_t))));	  // \hat{mu}^{t}

	muHatS.set(t, mu_hat_t.getArray());	// update
//	System.out.println(mu_hat_t.getRowDimension());
      } else {
	// initial condition for backward pass: \hat{mu}^{T} = mu^{T}
	Matrix mu_hat_t = new Matrix(muS.get(t));
	muHatS.set(t, mu_hat_t.getArray());	// update
      }
      Scanner sc = new Scanner(System.in);
      int gu; gu = sc.nextInt();
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



import java.io.*;
import java.util.*;
import Jama.*;

public class Main {
  public static int t0 = 1;
  public static int T = 10;
  public static double lambda = 0.5;
  public static double sigma = 0.3;
  public static double delta = 0.3;
  public static double scale = 0.2;

  public static void test1() {
    List<Integer> NS = new ArrayList<Integer>(T);
    List<double[][]> GS = new ArrayList<double[][]>(T);
    List<double[][]> AS = new ArrayList<double[][]>(T);
    List<double[][]> muS = new ArrayList<double[][]>(T);
    List<double[][]> muHatS = new ArrayList<double[][]>(T);
    List<double[][]> HS = new ArrayList<double[][]>(T);
    List<double[][]> HHatS = new ArrayList<double[][]>(T);
    List<Map<Integer, Integer>> idMapS = new ArrayList<Map<Integer, Integer>>(T);
    Random rand = new Random(0);

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
      double[][] HHat = new double[n][1];
      FileParser.readCSVGraph(fileDir, freq, idMap, G, A);
      for (int i = 0; i < n; i++) {
	mu[i][0] = scale * (rand.nextDouble() - 0.5);
	muHat[i][0] = scale * (rand.nextDouble() - 0.5);
	H[i][0] = scale * (rand.nextDouble() - 0.5);
	HHat[i][0] = scale * (rand.nextDouble() - 0.5);
      }

      GS.add(G); AS.add(A);
      muS.add(mu); muHatS.add(mu); HS.add(H); HHatS.add(HHat);

      System.out.println("done! t = " + t);
    }
    /*
    Scanner sc = new Scanner(System.in);
    int gu; gu = sc.nextInt();
    */

    // forward pass 
    for (int t = t0; t < T; t++) {
      System.out.println("forward\tt = " + t);
      if (t != t0) {
	int old_n = NS.get(t-1);
	Matrix a = new Matrix(AS.get(t-1));
	Matrix h = new Matrix(HS.get(t));
	Matrix pt = (a.times(a.transpose()).plus(Matrix.identity(old_n, old_n))).times(sigma*sigma);
	Matrix kt = pt.times(pt.plus(Matrix.identity(old_n, old_n).times(delta*delta)).inverse());
	Matrix old_ht = Operations.translate(h, idMapS.get(t), idMapS.get(t-1));
	/*
	for (int i = 0; i < h.getRowDimension(); i++) System.out.println(h.get(i,0));
	System.out.println("-----------------");
	for (int i = 0; i < old_ht.getRowDimension(); i++) System.out.println(old_ht.get(i,0));
	*/
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



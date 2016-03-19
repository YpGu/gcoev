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

}

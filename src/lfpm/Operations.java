import java.util.*;
import Jama.*;

public class Operations {
  public static Matrix 
  translate(Matrix arr1, Map<Integer, Integer> map1, Map<Integer, Integer> map2) {
    int n2 = map2.size();
    double[][] arr2 = new double[n2][1];
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

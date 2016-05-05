import java.io.*;
import java.util.*;
import java.lang.*;

public class FileParser {
  public static Map<Integer, Integer> 
  readCSVDict(String fileDir) {
    Map<Integer, Set<Integer> > x2ys = new HashMap<Integer, Set<Integer> >();
    try (BufferedReader br = new BufferedReader(new FileReader(fileDir))) {
      String currentLine;
      while ((currentLine = br.readLine()) != null) {
	// each line: id1, id2, (weight)
	String[] tokens = currentLine.split(",");
	int x = Integer.parseInt(tokens[0]);	  // global ID
	int y = Integer.parseInt(tokens[1]);	  // global ID
	// update frequencies
	if (!x2ys.containsKey(x)) {
	  Set<Integer> ys = new HashSet<Integer>();
	  ys.add(y);
	  x2ys.put(x, ys);
	} else {
	  Set<Integer> ys = x2ys.get(x);
	  ys.add(y);
	  x2ys.put(x, ys);
	}

	if (!x2ys.containsKey(y)) {
	  Set<Integer> ys = new HashSet<Integer>();
	  ys.add(x);
	  x2ys.put(y, ys);
	} else {
	  Set<Integer> ys = x2ys.get(y);
	  ys.add(x);
	  x2ys.put(y, ys);
	}
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    Map<Integer, Integer> freq = new HashMap<Integer, Integer>();
    for (Map.Entry<Integer, Set<Integer> > e: x2ys.entrySet()) {
      int x = e.getKey();
      Set<Integer> ys = e.getValue();
      int size = ys.size();
      freq.put(x, size);
    }

    return freq;
  }

  /** readCSVGraph:
   *	read network in .csv format
   *	G: binary edge
   *	A: normalized adjacency matrix
   *	label: whether the file contains existing links or non-existing links
   */
  public static void 
  readCSVGraph(String fileDir, Map<Integer, Integer> freq, double[][] G, double[][] A, List<Integer> lst, boolean label) {
    try (BufferedReader br = new BufferedReader(new FileReader(fileDir))) {
      String currentLine;
      while ((currentLine = br.readLine()) != null) {
	// each line: id1, id2, weight
	String[] tokens = currentLine.split(",");
	int x = Integer.parseInt(tokens[0]);
	int y = Integer.parseInt(tokens[1]);
	int e1 = x * Main.n + y, e2 = y * Main.n + x;
	if (label) {
	  /* need to make the network undirected */
	  lst.add(e1); lst.add(e2);
	  G[x][y] = 1.0; G[y][x] = 1.0;
	  A[x][y] = 1.0 / freq.get(x);
	  A[y][x] = 1.0 / freq.get(y);
	} else {
	  /* because of the way in which negative links are sampled,
	   * only directed links are added
	   */
	  lst.add(e1); 
	}
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    return;
  }

  public static void
  output(double[][] arr, String fileDir) {
    try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(fileDir)))) {
      for (int i = 0; i < arr.length; i++) {
        writer.printf("%d ", i);
	for (int j = 0; j < arr[i].length; j++) {
	  writer.printf("%f ", arr[i][j]);
	}
        writer.printf("\n");
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /* arr_s.get(t) is a n*1 array */
  public static void
  output_2d(List<double[][]> arr_s, String fileDir) {
    try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(fileDir)))) {
      for (int i = 0; i < arr_s.size(); i++) {
	double[][] arr = arr_s.get(i);
        writer.printf("%d ", i);
	for (int j = 0; j < arr.length; j++) {
	  writer.printf("%f ", arr[j][0]);
	}
        writer.printf("\n");
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /* arr_s.get(t) is a n*1 array */
  public static void
  output_1d(List<double[]> arr_s, String fileDir) {
    try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(fileDir)))) {
      for (int i = 0; i < arr_s.size(); i++) {
	double[] arr = arr_s.get(i);
        writer.printf("%d ", i);
	for (int j = 0; j < arr.length; j++) {
	  writer.printf("%f ", arr[j]);
	}
        writer.printf("\n");
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /* arr_s.get(t) is a n*1 array */
  public static void
  output(Map<Integer, Integer> id_map, String fileDir) {
    try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(fileDir)))) {
      for (Map.Entry<Integer, Integer> e: id_map.entrySet()) {
	int globalID = e.getKey();
	int localID = e.getValue();
	writer.printf("%d %d\n", globalID, localID);
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}

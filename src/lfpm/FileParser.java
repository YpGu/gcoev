import java.io.*;
import java.util.*;
import java.lang.*;

public class FileParser {
  public static Map<Integer, Double> 
  readCSVDict(String fileDir, Map<Integer, Integer> idMap, Map<Integer, Integer> idMapInv) {
    Map<Integer, Double> freq = new HashMap<Integer, Double>();
    try (BufferedReader br = new BufferedReader(new FileReader(fileDir))) {
      String currentLine;
      int newID = 0;
      while ((currentLine = br.readLine()) != null) {
	// each line: id1, id2, weight
	String[] tokens = currentLine.split(",");
	int id1 = Integer.parseInt(tokens[0]);	  // global ID
	int id2 = Integer.parseInt(tokens[1]);	  // global ID
	if (!idMap.containsKey(id1)) {
	  idMap.put(id1, newID);
	  idMapInv.put(newID, id1);
	  newID += 1;
	}
	if (!idMap.containsKey(id2)) {
	  idMap.put(id2, newID);
	  idMapInv.put(newID, id2);
	  newID += 1;
	}
	// update frequencies
	if (!freq.containsKey(id1)) 
	  freq.put(id1, 1.0);
	else 
	  freq.put(id1, freq.get(id1) + 1);
	if (!freq.containsKey(id2)) 
	  freq.put(id2, 1.0);
	else 
	  freq.put(id2, freq.get(id2) + 1);
      }
    }
    catch (IOException e) {
      e.printStackTrace();
    }

    return freq;
  }

  public static void 
  readCSVGraph(String fileDir, Map<Integer, Double> freq, Map<Integer, Integer> idMap, double[][] G, double[][] A) {
    try (BufferedReader br = new BufferedReader(new FileReader(fileDir))) {
      String currentLine;
      while ((currentLine = br.readLine()) != null) {
	// each line: id1, id2, weight
	String[] tokens = currentLine.split(",");
	int id1 = Integer.parseInt(tokens[0]);
	int id2 = Integer.parseInt(tokens[1]);
	int x = idMap.get(id1);		// local_id1 
	int y = idMap.get(id2);		// local_id2
	G[x][y] = 1.0; G[y][x] = 1.0;
	A[x][y] = (1-Main.lambda) / freq.get(id1);
	A[y][x] = (1-Main.lambda) / freq.get(id2);
      }
      int n = idMap.size();
      for (int x = 0; x < n; x++) {
	A[x][x] = Main.lambda;
      }
    }
    catch (IOException e) {
      e.printStackTrace();
    }

    return;
  }

	public static void
	output(double[][] arr, String fileDir) {
		try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(fileDir)))) {
			for (int i = 0; i < arr.length; i++) {
				writer.printf("%d\t", i);
				for (int j = 0; j < arr[0].length; j++) {
					writer.printf("%f\t", arr[i][j]);
				}
				writer.printf("\n");
			}
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void
	output(double[] arr, String fileDir) {
		try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(fileDir)))) {
			for (int i = 0; i < arr.length; i++) {
				writer.printf("%d\t%f\n", i, arr[i]);
			}
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}

}

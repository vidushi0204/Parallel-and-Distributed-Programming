# Maze Generator and Solver using MPI (Distributed Programming)

## Objective

The objective of this assignment is to generate and solve a 64x64 perfect maze using **Message Passing Interface (MPI)** in C++. A perfect maze is one where there is exactly one path between any two cells. The entry point of the maze is located at (0,63), and the exit is at (63,0). We used **BFS** and **Kruskal’s Algorithm** to generate the maze, and **DFS** and **Dijkstra’s Algorithm** to solve it.

## Maze Generation

### 1. **Breadth-First Search (BFS)**

#### Sequential Algorithm

- The grid is initialized such that all cells are walls, except for alternating cells in alternating rows, along with the start and end cells, which are marked as empty.
- A frontier is maintained, which consists of cells that are yet to be processed.
- Randomly shuffled frontiers are processed, and the walls between them and unvisited neighbors are removed, resulting in a randomly generated perfect maze.

#### MPI Approach

- The maze is divided into 4 parts (16x16 grids), and each processor generates a minimum spanning tree (MST) for its part using BFS.
- After generating individual components, the processors send their results to the master processor, which joins them and introduces some random edges to connect all components into a single spanning tree.

### 2. **Kruskal’s Algorithm**

#### Sequential Algorithm

- The maze is initialized as a grid of unconnected cells, and each pair of adjacent cells is considered to have a wall between them.
- The algorithm randomly removes walls between unconnected cells, ensuring that no cycles are created, thus generating a perfect maze.

#### MPI Approach

- The maze is divided into four parts (16x16 grids), each handled by a separate processor, which uses Kruskal’s algorithm to generate an MST for its part.
- After the trees are generated, the master processor collects the trees and combines them by adding random edges to connect all the parts into one perfect maze.

### Complexity and Speedup

- Sequential Time Complexity: **O(n²)** for a maze of size n x n.
- Parallel Time Complexity:
  - Each processor works on its part in **O(n²/p)** time.
  - MPI Gather introduces an overhead of **O(n² log p)**.
  - Total Time Complexity: **O(n²/p) + O(n² log p)**.
  - Speedup: **1 / (O(1/p) + O(log p))**.
  - Efficiency: **1 / (1 + O(p log p))**.

## Maze Solver

### 1. **Depth-First Search (DFS)**

#### Sequential Algorithm

- Starts at the entry point (0,63) and recursively explores adjacent non-wall cells until the exit point (63,0) is reached.
- The cells on the path to the exit are recorded.

#### MPI Approach

- Each processor processes a part of the maze and performs random DFS traversals on non-wall cells. If a path to the exit is found, the traversal halts.
- Frontiers are redistributed among processors using **MPI Broadcast** if no path is found.

### 2. **Dijkstra’s Algorithm**

#### Sequential Algorithm

- The maze is treated as a graph where each cell is a node. Dijkstra’s algorithm computes the shortest path from the entry to the exit by exploring cells in order of increasing distance from the start.
  
#### MPI Approach

- The maze is distributed among processors, each handling the shortest path calculation for its part of the maze.
- Processors communicate using **MPI** to share the shortest paths, and idle processors request additional work when they run out of cells to process.

### Solver Complexity

- DFS:
  - Sequential: **O(n²)**.
  - Parallel: **O(n²/p) + O(n² log p)**.
- Dijkstra:
  - Sequential: **O(n² log n²)**.
  - Parallel: **O(n² log n²/p) + O(n² log p)**.
  
## Optimizations

- **Disjoint MSTs**: For the Kruskal-based maze generation, the MST was divided into 4 disjoint components, minimizing communication overhead.
- **Selective Memory Transfer**: Only the relevant portions of the maze are transferred to processors to reduce the data size during communication.
- **Randomness**: A random number generator was used to introduce randomness during maze generation, making each run unique.
- **Efficient Data Handling**: Matrices and vectors were passed by reference or pointers to avoid unnecessary copying, improving performance.

## MPI Usage

- **MPI Gather**: Used to collect individual MSTs or path results from the processors and combine them in the master processor.
- **MPI Broadcast**: Used to redistribute frontiers during the DFS-based maze solving.
- **MPI Barrier**: Used to synchronize all processors at critical points in the algorithm.

## Conclusion

This project demonstrates the use of MPI for parallel maze generation and solving. The BFS-based maze generation is faster and simpler, while the Kruskal-based maze results in a more complex maze structure. DFS proves to be more efficient for solving the maze, requiring fewer cell visits than Dijkstra’s algorithm. Through parallelization, significant performance improvements were achieved, especially on larger mazes.

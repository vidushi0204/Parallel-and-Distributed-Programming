#ifndef MAZE_SOLVER_HPP
#define MAZE_SOLVER_HPP

#include <vector>
#include <utility>

void print_soln(std::vector<std::pair<int, int>>& path, int** maze_map, int sz);
void maze_solver(int dfs_or_dijkstra, int my_rank, int num_of_processors, int sz);

#endif // MAZE_SOLVER_HPP

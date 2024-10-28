#ifndef DFS_HPP
#define DFS_HPP

#include <vector>
#include <utility>

void set_all_dfs(int** visited, int** maze_map, int sz);
std::vector<std::pair<int, int>> child_dfs(int x, int y, int sz);
bool dfs(int x, int y, int** maze_map, int** visited, int sz, std::vector<std::pair<int, int>>& path);
std::vector<std::pair<int, int>> path_dfs(int** maze_map, int** visited, int sz);

#endif // DFS_HPP

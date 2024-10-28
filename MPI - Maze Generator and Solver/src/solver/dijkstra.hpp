#ifndef DIJKSTRA_HPP
#define DIJKSTRA_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <queue>

void set_all_dijkstra(std::pair<int, int>** parent, int** visited, int** maze_map, int sz);
std::vector<std::pair<int, int>> child_dijkstra(int x, int y, int sz);
std::vector<std::pair<int, int>> path_dijkstra(int** maze_map, int** visited, int sz, std::pair<int, int>** parent);

#endif 

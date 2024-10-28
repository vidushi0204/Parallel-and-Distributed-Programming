#ifndef BFS_HPP
#define BFS_HPP

#include <vector>
#include <utility>
using namespace std;

vector<pair<int,int>> child(int x, int y, int sz);
void set_all(int** visited, int** maze_map, int sz, int my_rank);
void generate_maze(int sz, int** maze_map, int** visited);
void bfs(int** maze_map, int** visited, int sz, int my_rank);

#endif 

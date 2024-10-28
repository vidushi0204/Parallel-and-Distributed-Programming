#ifndef PARALLEL_MAZE_HPP
#define PARALLEL_MAZE_HPP

#include <mpi.h>

void print_maze(int** maze_map, int sz);
void jodo(int** final_maze, int sz, int* offset, int my_size, int* badi_maze, int x, int y);
int** gather_maze(int* badi_maze, int sz, int my_size);
void maze_generator(int bfs_or_kruskal,int my_rank, int num_of_processors,int sz);

#endif // PARALLEL_MAZE_HPP

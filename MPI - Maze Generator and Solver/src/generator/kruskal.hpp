#ifndef KRUSKAL_HPP
#define KRUSKAL_HPP

#include <vector>
#include <random>
using namespace std;


struct Wall {
    int cell1;
    int cell2;
};

vector<Wall> shuffleWalls(std::vector<Wall>& walls);
void generateMaze(int numCellsX, int numCellsY, int** maze);
int** kruskal(int my_size,int my_rank);

#endif 
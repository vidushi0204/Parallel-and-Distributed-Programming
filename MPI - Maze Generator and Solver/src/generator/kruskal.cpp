// C++ program for the above approach 

#include <bits/stdc++.h> 
#include "kruskal.hpp"

using namespace std; 


// DSU data structure 
// path compression + rank by union 
class DSU { 
	int* parent; 
	int* rank; 

public: 
	DSU(int n) 
	{ 
		parent = new int[n]; 
		rank = new int[n]; 

		for (int i = 0; i < n; i++) { 
			parent[i] = i; 
			rank[i] = 1; 
		} 
	} 

	// Find function 
	int find(int i) 
	{ 
		if (parent[i] == i) 
			return i; 

		return parent[i] = find(parent[i]); 
	} 

	// Union function 
	void unite(int x, int y) 
	{ 
		int s1 = find(x); 
		int s2 = find(y); 

		if (s1 != s2) { 
			if (rank[s1] < rank[s2]) { 
				parent[s1] = s2; 
			} 
			else if (rank[s1] > rank[s2]) { 
				parent[s2] = s1; 
			} 
			else { 
				parent[s2] = s1; 
				rank[s1] += 1; 
			} 
		} 
	} 
}; 

// struct Wall {
//     int cell1;
//     int cell2;
// };

std::vector<Wall> gen_Wall(int numCellsX, int numCellsY) {
    std::vector<Wall> walls;
    for (int x = 0; x < numCellsX; ++x) {
        for (int y = 0; y < numCellsY; ++y) {
            if (x < numCellsX - 1) {
                walls.push_back({x * numCellsY + y, (x + 1) * numCellsY + y});
            }
            if (y < numCellsY - 1) {
                walls.push_back({x * numCellsY + y, x * numCellsY + y + 1});
            }
        }
    }
    return walls;
}

std::vector<Wall> shuffleWalls(std::vector<Wall>& walls) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(walls.begin(), walls.end(), g);
    return walls;
}

void generateMaze(int numCellsX, int numCellsY,int** maze) {
    std::vector<Wall> walls = gen_Wall(numCellsX, numCellsY);
    walls = shuffleWalls(walls);
    DSU dsu(numCellsX * numCellsY);
    int a = dsu.find(0);
    int b = dsu.find(numCellsX * numCellsY - 1);
    
    for(auto wall : walls){
        
        // Wall wall = walls[rand() % walls.size()]; // walls[rand() % walls.size()
        int cell1 = wall.cell1;
        int cell2 = wall.cell2;
        
        if (dsu.find(cell1) != dsu.find(cell2)) {
            // Remove the wall (not implemented in this code)
            // Join the sets of the formerly divided cells
            dsu.unite(cell1, cell2);
            // Output the information about joining the sets
            if(abs(cell1-cell2)==1){
                
                maze[2*(cell1/numCellsY)][2*(cell1%numCellsY)+1]=0;
            }

            else{
                maze[2*(cell1/numCellsY)+1][2*(cell1%numCellsY)]=0;
            }
        }
        
        if(cell1==0){
            a=dsu.find(cell1);
        }
        if(cell2==numCellsX*numCellsY-1){
            b=dsu.find(cell2);
        }
    }

    for(int i=0;i<2*numCellsX;i+=2){
        for(int j=0;j<2*numCellsY;j+=2){
         if(i%2==0) maze[i][j]=0;
        }
    }

}
int** kruskal(int sz,int my_rank) {
    int my_size=sz/2;
    
    int numCellsX = my_size;
    int numCellsY = my_size;
    int** maze = new int*[2*numCellsX];
    for(int i=0; i<2*numCellsX; i++){
        maze[i] = new int[2*numCellsY];
        for(int j=0; j<2*numCellsY; j++){
            maze[i][j] = 1;
        }
    }
    generateMaze(numCellsX, numCellsY,maze);
    if(my_rank==0){
        maze[0][sz-1]=0;
        maze[sz-1][0]=0;
    } else if(my_rank==2){
        maze[0][sz-1]=0;
    }else if(my_rank==3){
        int r=rand();
    
        if(r%2==0){maze[sz-1][sz-2]=0;}
        else{maze[sz-2][sz-1]=0;}
        maze[sz-1][sz-1]=0;
    }
    return maze;
}
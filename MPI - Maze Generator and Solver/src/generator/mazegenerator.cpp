#include <iostream>
#include<vector>
#include<algorithm>
#include<queue>
#include<fstream>
#include<random>
#include <mpi.h>
#include "bfs.hpp"
#include "kruskal.hpp"
#include "mazegenerator.hpp"

using namespace std;
void print_maze(int** maze_map,int sz){
    ofstream myfile;
    myfile.open("maze.txt");    
    
    for(int i=0; i<sz; i++){
        for(int j=0; j<sz; j++){
            if(maze_map[i][sz-1-j]==1){
                myfile<<"* ";
            }
            else{
                myfile<<maze_map[i][sz-j-1]<<" ";
            }
            
        }
        myfile<<endl;
    }
    myfile.close();
}


void jodo(int ** final_maze, int sz, int* offset, int my_size, int* badi_maze,int x,int y){
    for(int i=0; i<my_size; i++){
        for(int j=0; j<my_size; j++){
            final_maze[x+i][y+j]=badi_maze[*offset];
            *offset+=1;
        }
    }
}
int** gather_maze(int* badi_maze,int sz,int my_size){
    int** final_maze = new int*[sz];
    for(int i=0; i<sz; i++){
        final_maze[i]=new int[sz];
    }
    int offset=0;
    jodo(final_maze,sz,&offset,my_size,badi_maze,0,0);
    jodo(final_maze,sz,&offset,my_size,badi_maze,0,my_size);
    jodo(final_maze,sz,&offset,my_size,badi_maze,my_size,0);
    jodo(final_maze,sz,&offset,my_size,badi_maze,my_size,my_size);
    
    return final_maze;
}



void maze_generator(int bfs_or_kruskal,int my_rank, int num_of_processors,int sz){
    
    int my_size=sz/2;
    int **maze_map = new int*[my_size];
    if(bfs_or_kruskal==0){
        int** visited = new int*[my_size];
        bfs(maze_map,visited,my_size,my_rank);
        // if(my_rank==0)print_maze(maze_map,my_size);
    }else{
        maze_map = kruskal(my_size,my_rank);
        
    }

    int final_maze[my_size*my_size];
    
    for(int i=0; i<my_size; i++){
        for(int j=0; j<my_size; j++){
            final_maze[i*my_size+j]=maze_map[i][j];
        }
    }
    
    int badi_maze[sz*sz];
    
    MPI_Gather(final_maze,my_size*my_size,MPI_INT,badi_maze,my_size*my_size,MPI_INT,0,MPI_COMM_WORLD);

    if(my_rank==0){
        int** semifinal_maze = new int*[sz*sz];
        semifinal_maze = gather_maze(badi_maze,sz,my_size);
        print_maze(semifinal_maze,sz);
    }
}
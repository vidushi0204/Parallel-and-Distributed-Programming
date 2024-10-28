#include <bits/stdc++.h>
#include "./generator/bfs.hpp"
#include "./generator/kruskal.hpp"
#include "./generator/mazegenerator.hpp"
#include "./solver/mazesolver.hpp"
#include "./solver/dijkstra.hpp"
#include "./solver/dfs.hpp"
#include <mpi.h>

int main(int argc, char** argv){
    int sz=64;
    MPI_Init(&argc,&argv);
    int num_of_processors,my_rank;
    
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    string maze_algo, solve_algo;
    maze_algo=argv[2];
    solve_algo=argv[4];
    
    if(maze_algo=="bfs"){
        maze_generator(0,my_rank,num_of_processors,sz);
    }
    else if(maze_algo=="kruskal"){
        maze_generator(1,my_rank,num_of_processors,sz);
    }
    else{
        cout<<"Invalid maze generation algorithm"<<endl;
        return 0;
    }
    if(solve_algo=="dfs"){
        maze_solver(0,my_rank,num_of_processors,sz);
    }
    else if(solve_algo=="dijkstra"){
        maze_solver(1,my_rank,num_of_processors,sz);
    }
    else{
        cout<<"Invalid maze solving algorithm"<<endl;
        return 0;
    }
    
    MPI_Finalize();
    return 0;
}
#include <iostream>
#include<vector>
#include<algorithm>
#include<fstream>
#include<random>
#include <mpi.h>
#include "dfs.hpp"
#include "dijkstra.hpp"
#include "mazesolver.hpp"

using namespace std;

void print_soln(vector<pair<int,int>> &path, int** maze_map, int sz){
    for(auto u:path){
        maze_map[u.first][u.second]=2;
        // cout<<u.first<<" "<<u.second<<endl;
    }
    for(int i=0;i<sz;i++){
        for(int j=0;j<sz;j++){
            if(i==0 && j==sz-1){
                cout<<"S";
            }else if(i==sz-1 && j==0){
                cout<<"E";
            }else if(maze_map[i][j]==1){
                cout<<"*";
            }else if(maze_map[i][j]==2){
                cout<<"P";
            }else if(maze_map[i][j]==0){
                cout<<" ";
            }
        }
        cout<<endl;
    }
}

void maze_solver(int dfs_or_dijkstra,int my_rank, int num_of_processors,int sz){

    int my_sz=sz/2;

    int** maze_map = new int*[sz];
    int ** visited = new int*[sz];
    vector<pair<int,int>> path;
    if(dfs_or_dijkstra==0){
        set_all_dfs(visited,maze_map,my_sz);
        path=path_dfs(maze_map,visited,my_sz);
    }else{
        pair<int,int> ** parent = new pair<int,int>*[sz];
        set_all_dijkstra(parent,visited,maze_map,my_sz);
        path=path_dijkstra(maze_map,visited,my_sz,parent);
    }

    int* path_sz = new int[num_of_processors];
    int* path_sz_proc= new int[1];
    path_sz_proc[0]=path.size();
    MPI_Gather(path_sz_proc,1,MPI_INT,path_sz,1,MPI_INT,0,MPI_COMM_WORLD);
    if(my_rank==0){
        int final_path_size=0;
        for(int i=0;i<num_of_processors;i++){
            final_path_size+=path_sz[i];
        }
        int* final_path = new int[final_path_size];
    }

    if(my_rank==0) print_soln(path,maze_map,sz);

}
#include <iostream>
#include<vector>
#include<algorithm>
#include<fstream>
#include<random>
#include "dfs.hpp"
#include <mpi.h>
using namespace std;

void set_all_dfs(int** visited, int **maze_map, int my_sz){
    int sz=my_sz*2;
    ifstream myfile;
    myfile.open("maze.txt");    
    char a;
    for(int i=0;i<sz;i++){
        visited[i]=new int[sz];
        maze_map[i]=new int[sz];
        for(int j=0;j<sz;j++){
            visited[i][j]=0;
            myfile>>a;
            if(a=='*'){
                maze_map[i][j]=1;
            }else {
                maze_map[i][j]=0;
            }
        }
    }
    myfile.close();
}

vector<pair<int,int>> child_dfs(int x, int y,int sz){
    vector<pair<int,int>> children;
    if(x>0){
        children.push_back(make_pair(x-1,y));
    }
    if(x<sz-1){
        children.push_back(make_pair(x+1,y));
    }
    if(y>0){
        children.push_back(make_pair(x,y-1));
    }
    if(y<sz-1){
        children.push_back(make_pair(x,y+1));
    }
    return children;
}

bool dfs(int x,int y, int** maze_map, int** visited, int sz, vector<pair<int,int>> &path, vector<int> frontier_local){
    if(x==sz-1 && y==0){
        return true;
    }
    if(visited[x][y]==1){
        return false;
    }
    visited[x][y]=1;

    int size_of_frontier=frontier_local.size();
    int global_frontier[4*size_of_frontier];

    //local frontier 
    int local_frontier[size_of_frontier];
    size_of_frontier--;
    for(int i=0;i<size_of_frontier;i++){
        local_frontier[i]=frontier_local[i];
    }


    if(size_of_frontier) MPI_Gather(local_frontier,size_of_frontier,MPI_INT,global_frontier,size_of_frontier,MPI_INT,0,MPI_COMM_WORLD);

    vector<pair<int,int>> children = child_dfs(x,y,sz);

    if(size_of_frontier) MPI_Barrier(MPI_COMM_WORLD);
    for(auto u:children){
        if(visited[u.first][u.second]==0 && maze_map[u.first][u.second]==0){
            if(dfs(u.first,u.second,maze_map,visited,sz,path,frontier_local)){
                path.push_back(make_pair(u.first,u.second));
                return true;
            }
        }
    }


    return false;
}
vector<pair<int,int>> path_dfs(int ** maze_map, int** visited, int my_sz){
    int sz=my_sz*2;
    vector<int> frontier_local;
    frontier_local.push_back(0);
    vector<pair<int,int>> path;
    // send maze to all threads
    int maze_local[my_sz];
    for(int i=0;i<my_sz;i++){
        maze_local[i]=0;
    }

    for(int i=0;i<sz;i++){
        MPI_Bcast(maze_local,my_sz,MPI_INT,0,MPI_COMM_WORLD);
    }

    dfs(0,sz-1,maze_map,visited,sz,path,frontier_local); 
    path.push_back(make_pair(0,sz-1)); 
    reverse(path.begin(),path.end());  
    return path;
}



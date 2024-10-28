#include <iostream>
#include<vector>
#include<algorithm>
#include<fstream>
#include<queue>
#include<random>
#include <mpi.h>
#include "dijkstra.hpp"


using namespace std;
const int inf=1e9;
void set_all_dijkstra(pair<int,int>** parent,int** visited, int **maze_map, int my_sz){
    int sz=my_sz*2;
    ifstream myfile;
    myfile.open("maze.txt");    
    char a;
    for(int i=0;i<sz;i++){
        visited[i]=new int[sz];
        maze_map[i]=new int[sz];
        parent[i]=new pair<int,int>[sz];
        for(int j=0;j<sz;j++){
            visited[i][j]=inf;
            myfile>>a;
            if(a=='*'){
                maze_map[i][j]=1;
            }else {
                maze_map[i][j]=0;
            }
        }
    }
    visited[0][sz-1]=0;
    parent[0][sz-1]=make_pair(-1,-1);
    myfile.close();
}

vector<pair<int,int>> child_dijkstra(int x, int y,int sz){
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

void update_distances_in_neighbour(int x, int y, int* local_maze, int my_sz){
    //get the neighbours of the current cell
    int local_neighbour[4];

    if(x>0)
    {
        MPI_Gather(&local_maze[(x-1)*my_sz],my_sz,MPI_INT,local_neighbour,my_sz,MPI_INT,0,MPI_COMM_WORLD);
    }
}


vector<pair<int,int>> path_dijkstra(int ** maze_map, int** visited, int my_sz,pair<int,int>** parent){
    int sz=my_sz*2;
    vector<pair<int,int>> path;
    priority_queue<pair<int,pair<int,int>>> pq;

    int local_maze[my_sz];
    for(int i=0;i<my_sz;i++){
        local_maze[i]=0;
    }
    

    for(int i=0;i<sz;i++){
        MPI_Bcast(local_maze,my_sz,MPI_INT,0,MPI_COMM_WORLD);
    }

    pq.push(make_pair(0,make_pair(0,sz-1)));
    

    pair<int,int> curr;

    while(!pq.empty()){
        curr=pq.top().second;
        pq.pop();
        if(curr.first==sz-1 && curr.second==sz-1){
            break;
        }
        if(visited[curr.first][curr.second]==1){
            continue;
        }


        // update the distasneces of the neighbours
        update_distances_in_neighbour(0,0,local_maze,my_sz);

        
        visited[curr.first][curr.second]=1;
        vector<pair<int,int>> children = child_dijkstra(curr.first,curr.second,sz);
        for(auto u:children){
            if(visited[u.first][u.second]!=inf || maze_map[u.first][u.second]) continue;
            if(visited[u.first][u.second]>visited[curr.first][curr.second]+1){
                visited[u.first][u.second]=visited[curr.first][curr.second]+1;
                parent[u.first][u.second]=curr;
                pq.push(make_pair(-1*visited[u.first][u.second],u));
            }
        }
    }
    
    curr=make_pair(sz-1,0);
    while(curr!=make_pair(-1,-1)){
        path.push_back(curr);
        curr=parent[curr.first][curr.second];
    }
    reverse(path.begin(),path.end());  
    return path;
}

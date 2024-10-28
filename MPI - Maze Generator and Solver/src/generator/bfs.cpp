#include<iostream>
#include<vector>
#include<algorithm>
#include<queue>
#include<fstream>
#include<random>
#include "bfs.hpp"

using namespace std;

vector<pair<int,int>> child(int x, int y,int sz){
    vector<pair<int,int>> children;
    if(x>1){
        children.push_back(make_pair(x-2,y));
    }
    if(x<sz-2){
        children.push_back(make_pair(x+2,y));
    }
    if(y>1){
        children.push_back(make_pair(x,y-2));
    }
    if(y<sz-2){
        children.push_back(make_pair(x,y+2));
    }
    return children;
}
void set_all(int** visited,int** maze_map, int sz,int my_rank){
    for(int i=0; i<sz; i++){
        visited[i]=new int[sz];
        maze_map[i]=new int[sz];
        for(int j=0; j<sz; j++){
            visited[i][j]=0;
            maze_map[i][j]=1;
        }
    }

    for(int i=0; i<sz; i++){
        if(i%2==1){
            for(int j=0; j<sz; j++){
                maze_map[i][j]=1;
            }
            continue;
        }
        for(int j=0; j<sz; j++){
            if((i+j)%2==0) maze_map[i][j]=0;
        }
    }
    if(my_rank==0){
        maze_map[0][sz-1]=0;
        maze_map[sz-1][0]=0;
    } else if(my_rank==2){
        maze_map[0][sz-1]=0;
    }else if(my_rank==3){
        int r=rand();
    
        if(r%2==0){maze_map[sz-1][sz-2]=0;}
        else{maze_map[sz-2][sz-1]=0;}
        maze_map[sz-1][sz-1]=0;
    }
}


void generate_maze(int sz,int** maze_map,int** visited){
    vector<pair<int,int>> bfs_curr;
    pair<int,int> curr;
    bfs_curr.push_back(make_pair(0,0));
    visited[0][0]=1;
    while(!bfs_curr.empty()){
        random_device rand1;
        mt19937 rng1(rand1());
        shuffle(bfs_curr.begin(), bfs_curr.end(), rng1);
        curr=bfs_curr[bfs_curr.size()-1];
        bfs_curr.pop_back();

        vector<pair<int,int>> children = child(curr.first, curr.second,sz);
        if(children.size()==0){
            continue;
        }
        random_device rand2;
        mt19937 rng2(rand2());
        shuffle(children.begin(),children.end(), rng2);
        for(auto u: children){
            if(visited[u.first][u.second]==1){
                continue;
            }  
            maze_map[(u.first+curr.first)/2][(u.second+curr.second)/2]=0;
            visited[u.first][u.second]=1;
            bfs_curr.push_back(u);           
        }

    }  
}

void bfs(int** maze_map,int** visited,int my_size, int my_rank){

    set_all(visited,maze_map,my_size,my_rank);
    generate_maze(my_size,maze_map,visited);
}

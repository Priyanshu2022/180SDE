// is it possible to colour the graph with m colours
// try to color current node with every color
// bool funtion because only check if we can
bool issafe(int node,int color[],bool graph[101][101],int n,int i){
    for(int k=0;k<n;k++){
        if(k!=node && graph[k][node]==1 && color[k]==i) return false;
    }
    return true;
}   
bool solve(int node,bool graph[101][101],int m,int n,int color[]){
    if(node==n) return true;
    for(int i=1;i<=m;i++){
        if(issafe(node,color,graph,n,i)){
            color[node]=i;
            if(solve(node+1,graph,m,n,color)) return true;
            color[node]=0;
        }
    }
    return false;
}
bool graphColoring(bool graph[101][101], int m, int V)
{
    int color[V]={0};
    return solve(0,graph,m,V,color);
}
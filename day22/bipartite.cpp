// bipartite graph
// graph which can be coloured using 2 colors
// bfs , if any componenet gives false then the answer is false
// maintain a color array
bool bipartiteBfs(int node,vector<int> adj[],vector<int> &color){
        queue<int> q;
        q.push(node);
        while(!q.empty()){
            int temp=q.front();
            q.pop();
            for(auto it:adj[temp]){
                if(color[it]==-1) {
                    color[it]=1-color[temp];
                    q.push(it);
                }
                else if(color[it]==color[temp]) return false;
            }
        }
        return true;
    }
	bool isBipartite(int V, vector<int>adj[]){
	    vector<int> color(V,-1);
	    for(int i=0;i<V;i++){
	        if(color[i]==-1){
	            if(bipartiteBfs(i,adj,color)==false) return false;
	        }
	    }
	    return true;
	}


// dfs
// maintain color
bool bipartiteDfs(int node, vector<int> adj[], int color[]) {
    for(auto it : adj[node]) {
        if(color[it] == -1) {
            color[it] = 1 - color[node];
            if(!bipartiteDfs(it, adj, color)) {
                return false; 
            }
        } else if(color[it] == color[node]) return false; 
    }
    return true; 
}
bool checkBipartite(vector<int> adj[], int n) {
    int color[n];
    memset(color, -1, sizeof color); 
    for(int i = 0;i<n;i++) {
        if(color[i] == -1) {
            color[i] = 1;
            if(!bipartiteDfs(i, adj, color)) {
                return false;
            }
        } 
    }
    return true; 
}
// strongly connected comoponent (directed graph)
// Kosaraju's Algorithm
// from every node of that component we can reach to every other node
	// find toposort of the graph
	// transpose the graph
	// then run dfs according to topo sort
	void dfs(int node,vector<int> adj[],vector<int> &vis,stack<int> &st){
	    vis[node]=1;
	    for(auto it:adj[node]){
	        if(!vis[it]) dfs(it,adj,vis,st);
	    }
	    st.push(node);
	}
	void revdfs(int node,vector<int> transpose[],vector<int> &vis){
	    vis[node]=1;
	    for(auto it:transpose[node]){
	        if(!vis[it]) revdfs(it,transpose,vis);
	    }
	}
    int kosaraju(int V, vector<int> adj[])
    {
        stack<int> st;
        vector<int> vis(V,0);
        for(int i=0;i<V;i++) if(!vis[i]) dfs(i,adj,vis,st);
        vector<int> transpose[V];
        for(int i=0;i<V;i++){
            vis[i]=0;
            for(auto it:adj[i]){
                transpose[it].push_back(i);
            }
        }
        int ans=0;
        while(!st.empty()){
            int node=st.top();
            st.pop();
            if(!vis[node]){
                ans++;
                revdfs(node,transpose,vis);
            }
        }
        return ans;
    }
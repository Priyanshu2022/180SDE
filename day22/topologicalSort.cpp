// topological sort
// can only be possible of directed acyclic graph
// if u->v is a edge then u always appears before v

// using dfs -> n+e (tc)
// run dfs and in every node if not visited
// put in answer after recursively calling all it's adjacent
// then reverse the answer array
void solve(int i,vector<int> adj[],vector<int> &vis,vector<int> &ans,int V){
	    vis[i]=1;
	    
	    for(auto it:adj[i]){
	        if(!vis[it]){
	            solve(it,adj,vis,ans,V);
	        }
	    }
	    ans.push_back(i);
	 
	}
	vector<int> topoSort(int V, vector<int> adj[]) 
	{
	    vector<int> vis(V+1,0);
	    vector<int> ans;
	    for(int i=0;i<V;i++){
	        if(!vis[i]){
	            solve(i,adj,vis,ans,V);
	        }
	    }
	    reverse(ans.begin(),ans.end());
	    return ans;
	}


// using bfs
// maintain indegree and push all indegree node's of zeros to the queue
// not run bfs while queue is not empty , and whenever go to adjacent reduce indegree and check if zero
vector<int> topo(int N, vector<int> adj[]) {
        queue<int> q; 
	    vector<int> indegree(N, 0); 
	    for(int i = 0;i<N;i++) {
	        for(auto it: adj[i]) {
	            indegree[it]++; 
	        }
	    }
	    
	    for(int i = 0;i<N;i++) {
	        if(indegree[i] == 0) {
	            q.push(i); 
	        }
	    }
	    vector<int> topo;
	    while(!q.empty()) {
	        int node = q.front(); 
	        q.pop(); 
	        topo.push_back(node);
	        for(auto it : adj[node]) {
	            indegree[it]--;
	            if(indegree[it] == 0) {
	                q.push(it); 
	            }
	        }
	    }
	    return topo;
	}
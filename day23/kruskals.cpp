// mst using kruskals
// first transfer adj to all edges list
// then sort according to weigth
// then iterate on this edges 
// if have different parent , add weith to answer
    int findPar(int node,vector<int> &parent){
    	if(node==parent[node]){
    		return node;
    	}
    	return parent[node]=findPar(parent[node],parent);
    }
    void Union(int u,int v,vector<int> &rank,vector<int> &parent){
    	u=findPar(u,parent);
    	v=findPar(v,parent);
    	if(rank[u]<rank[v]){
    		parent[u]=v;
    	}
    	else if(rank[v]<rank[u]){
    		parent[v]=u;
    	}
    	else{
    		parent[v]=u;
    		rank[u]++;
    	} 
    }
    int spanningTree(int V, vector<vector<int>> adj[])
    {
        vector<pair<int,pair<int,int>>> v;
        for(int i=0;i<V;i++){
            for(auto it:adj[i]){
                v.push_back({it[1],{i,it[0]}});
                
            }
        }
        vector<int> rank(V,0);
        vector<int> parent(V);
        for(int i=0;i<V;i++){
            parent[i]=i;
        }
        sort(v.begin(),v.end());
        int ans=0;
        for(int i=0;i<v.size();i++){
            if(findPar(v[i].second.first,parent)!=findPar(v[i].second.second,parent)){
                ans+=v[i].first;
                Union(v[i].second.first,v[i].second.second,rank,parent);
            }
        }
        return ans;
    }
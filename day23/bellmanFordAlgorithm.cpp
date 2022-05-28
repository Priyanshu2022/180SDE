// dijkstra is not possible in a graph with negative cycle
// make a distTo vector
// iterate over all edges , n-1 times , and update if possible
// run one more time for check , if yet it any edge is getting relaxed , means graph has a negative cycle
int isNegativeWeightCycle(int n, vector<vector<int>>edges){
	    vector<int> distTo(n,1e7);
	    for(int i=1;i<=n-1;i++){
	        for(auto it:edges){
	            if(distTo[it[1]]>distTo[it[0]]+it[2]){
	                distTo[it[1]]=distTo[it[0]]+it[2];
	            }
	        }
	    }
	    bool f=0;
	    for(auto it:edges){
	        if(distTo[it[1]]>distTo[it[0]]+it[2]){
	            f=1;
	            break;
	        }
	    }
	    if(f) return 1;
	    else return 0;
	}
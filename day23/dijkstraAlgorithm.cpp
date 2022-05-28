// dijkstra -> single source shortest path
// vector<pair<int,int>> v[V] -> for each vertex , number of vertex connected with edges
// min heap
// distTo vector
// update if possible
vector <int> dijkstra(int V, vector<vector<int>> adj[], int S)
    {
        vector<pair<int,int>> v[V];
        for(int i=0;i<V;i++){
            // v[i].push_back({adj[i][0],adj[i][1]});
            for(auto cur:adj[i]){
                v[i].push_back({cur[0],cur[1]});
            }
        }
        priority_queue<pair<int,int>,vector<pair<int,int> >,greater<pair<int,int> > > pq;
        vector<int> distTo(V,INT_MAX);
        distTo[S]=0;
        pq.push({0,S});
        while(!pq.empty()){
            int dist=pq.top().first;
            int node=pq.top().second;
            pq.pop();
            for(auto it:v[node]){
                int next=it.first;
                int nexdis=it.second;
                if(nexdis+dist<distTo[next]){
                    distTo[next]=dist+nexdis;
                    pq.push({distTo[next],next});
                }
            }
        }
        return distTo;
    }
// cycle detection in undirected
// dfs -> maintain parent to differentiate between next and previous node
bool checkForCycle(int node, int parent, vector < int > & vis, vector < int > adj[]) {
      vis[node] = 1;
      for (auto it: adj[node]) {
        if (!vis[it]) {
          if (checkForCycle(it, node, vis, adj))
            return true;
        } else if (it != parent)
          return true;
      }

      return false;
    }
    bool isCycle(int V, vector < int > adj[]) {
      vector < int > vis(V + 1, 0);
      for (int i = 0; i < V; i++) {
        if (!vis[i]) {
          if (checkForCycle(i, -1, vis, adj)) return true;
        }
      }

      return false;
    } 


// bfs -> queue will be of node and parent
bool checkForCycle(int s, int V, vector<int> adj[], vector<int> &visited)
    {
        // Create a queue for BFS
        queue<pair<int, int>> q;
        visited[s] = true;
        q.push({s, -1});
        while (!q.empty())
        {
            int node = q.front().first;
            int par = q.front().second;
            q.pop();
 
            for (auto it : adj[node])
            {
                if (!visited[it])
                {
                    visited[it] = true;
                    q.push({it, node});
                }
                else if (par != it)
                    return true;
            }
        }
        return false;
    }
    bool isCycle(int V, vector<int> adj[])
    {
        vector<int> vis(V - 1, 0);
        for (int i = 1; i <= V; i++)
        {
            if (!vis[i])
            {
                if (checkForCycle(i, V, adj, vis))
                    return true;
            }
        }
    }



// for directed graph
// dfs discussed above will not work
// as we can visit same node if direction is same
// maintain vis and dfsvis(if the node is visited in the current movement)
bool checkCycle(int node, vector < int > adj[], int vis[], int dfsVis[]) {
      vis[node] = 1;
      dfsVis[node] = 1;
      for (auto it: adj[node]) {
        if (!vis[it]) {
          if (checkCycle(it, adj, vis, dfsVis)) return true;
        } else if (dfsVis[it]) {
          return true;
        }
      }
      dfsVis[node] = 0;
      return false;
    }
    bool isCyclic(int N, vector < int > adj[]) {
      int vis[N], dfsVis[N];
      memset(vis, 0, sizeof vis);
      memset(dfsVis, 0, sizeof dfsVis);

      for (int i = 0; i < N; i++) {
        if (!vis[i]) {
          if (checkCycle(i, adj, vis, dfsVis)) {
            return true;
          }
        }
      }
      return false;
    }


// bfs 
// we will check if we can form topo sort then ,then no cycle
// count the total elements of topo sort if they are equal to n , then it does not have a cycle
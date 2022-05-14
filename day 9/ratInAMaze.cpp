void solve(vector<string> &ans,string s,int i,int j,vector<vector<int>> &m,int n,vector<vector<int>> &vis){
        if(i==n-1 && j==n-1 ){
            ans.push_back(s);
            return;
        }
        if(i>=n || i<0 || j>=n || j<0 || m[i][j]==0 || vis[i][j]==1){
            return;
        }
        vis[i][j]=1;
        solve(ans,s+'D',i+1,j,m,n,vis);
        solve(ans,s+'U',i-1,j,m,n,vis);
        solve(ans,s+'R',i,j+1,m,n,vis);
        solve(ans,s+'L',i,j-1,m,n,vis);
        vis[i][j]=0;
    }
    public:
    vector<string> findPath(vector<vector<int>> &m, int n) {
        
        vector<string> ans;
        if(m[n-1][n-1]==0) return ans;
        string s="";
        vector<vector<int>> vis(n,vector<int>(n,0));
        solve(ans,s,0,0,m,n,vis);
        return ans;
    }
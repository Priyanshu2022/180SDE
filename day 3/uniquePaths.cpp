// recursive dp or iterative dp
// or simply total paths =(n-1)+(m-1)=n+m-2
// choose n-1 from total choices
// n+m-2Cn-1
int func(int i,int j,int m,int n,vector<vector<int>>& dp){
        if(i>=n || j>=m) return 0;
        if(i==n-1 && j==m-1) return 1;
        if(dp[i][j]!=-1) return dp[i][j];
        int down=func(i+1,j,m,n,dp);
        int right=func(i,j+1,m,n,dp);
        return dp[i][j]=right+down;
    }
    int uniquePaths(int m, int n) {
        int i=0;
        int j=0;
        vector<vector<int>> dp(m,vector<int> (n,-1));
        // return func(i,j,m,n,dp);
        dp[0][0]=1;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(i==0 && j==0) continue;
                if(i==0) dp[i][j]=dp[i][j-1];
                else if(j==0) dp[i][j]=dp[i-1][j];
                else dp[i][j]=dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
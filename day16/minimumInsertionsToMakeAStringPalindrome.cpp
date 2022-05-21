// minimum insertions to make palindrom
// length-longest palindromic subsequence
int solve(int i,int j,string &text1,string &text2,vector<vector<int>> &dp){
        if(i==-1 || j==-1 ) return 0;
        if(dp[i][j]!=-1) return dp[i][j];
        if(text1[i]==text2[j]) return dp[i][j]=1+solve(i-1,j-1,text1,text2,dp);
        else 
        return dp[i][j]=max(solve(i-1,j,text1,text2,dp),solve(i,j-1,text1,text2,dp));
    }
    int longestPalindromeSubseq(string s) {
        string text1=s;
        reverse(s.begin(),s.end());
        string text2=s;
        vector<vector<int>> dp(text1.length(),vector<int>(text2.length(),-1));
        return solve(text1.length()-1,text2.length()-1,text1,text2,dp);
    }
    int minInsertions(string s) {
        return s.length()-longestPalindromeSubseq(s);
    }
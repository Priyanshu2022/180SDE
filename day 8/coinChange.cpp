// take the current and do not change i or leave the current and change the i
int solve(int i,int amount,vector<int>& coins,vector<vector<int>> &dp){
        if(amount==0) return 0;
        if(amount<0 || i>=coins.size()) return INT_MAX-1;
        if(dp[i][amount]!=-1) return dp[i][amount];
        return dp[i][amount]=min(1+solve(i,amount-coins[i],coins,dp),solve(i+1,amount,coins,dp));
    }
    int coinChange(vector<int>& coins, int amount) {
        vector<vector<int>> dp(coins.size(),vector<int>(amount+1,-1));
        int ans=solve(0,amount,coins,dp);
        if(ans==INT_MAX-1) return -1;
        else return ans;
    }
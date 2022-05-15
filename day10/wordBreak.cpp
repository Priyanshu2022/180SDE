// word break 
// return true if we can break s , into space separated strings of words of wordDict
bool solve(int i,string s,unordered_map<string,int> &dict,vector<int> &dp){
        if(i==s.length()) return true;
        if(dp[i]!=-1) return dp[i];
        string temp;
        for(int j=i;j<s.length();j++){
            temp+=s[j];
            if(dict.find(temp)!=dict.end()){
                if(solve(j+1,s,dict,dp)) return dp[i]=true;
            }
        }
        return dp[i]=false;
    }
    bool wordBreak(string s, vector<string>& wordDict) {
        vector<int> dp(s.length(),-1);
        unordered_map<string,int> dict;
        for(int i=0;i<wordDict.size();i++) dict[wordDict[i]]++;
        return solve(0,s,dict,dp);
    }
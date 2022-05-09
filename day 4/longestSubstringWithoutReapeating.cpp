// left to right is our substring
// if s[right] is in map update left
// map stores char and its index
// ans=max(ans,right-left+1)
int lengthOfLongestSubstring(string s) {
         unordered_map<char,int> mp;
         int left=0;
         int right=0;
         int n =s.length();
         int ans=0;
        while(right<n){
            if(mp.find(s[right])!=mp.end()) left=max(left,mp[s[right]]+1);
            mp[s[right]]=right; 
            ans=max(ans,right-left+1);
            right++;
        }
        return ans;
    }
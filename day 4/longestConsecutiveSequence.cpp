// check if cur-1 present in map
// if present , continue;
// else count from there
int longestConsecutive(vector<int>& nums) {
        unordered_map<int,int> mp;
        for(auto cur:nums){
            mp[cur]++;
        }
        int ans=0;
        for(auto cur:nums){
            if(mp[cur-1]!=0) continue;
            else{
                int count=0;
                int temp=cur;
                while(mp[temp]!=0){
                    count++;
                    temp++;
                }
                ans=max(count,ans);
            }
        }
        return ans;
    }
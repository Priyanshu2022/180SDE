// check subarray sum 
// if it is update
// else is already present in map update length
// or put it in map
int maxLen(vector<int>&A, int n)
    {   
        unordered_map<int,int> mp;
        int sum=0;
        int ans=0;
        for(int i=0;i<A.size();i++){
            sum+=A[i];
            if(sum==0) ans=max(ans,i+1);
            else{
                if(mp[sum]==0){
                    mp[sum]=i+1;
                }
                else{
                    ans=max(ans,i+1-mp[sum]);
                }
            }
        }
        return ans;
    }
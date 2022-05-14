// tc=2^n*k (k  is average length)
// sc=k*x (x is total combinations) (ignorign auxiliary space)
// cannot pick a element more than once
    void combinationSum(int ind,int target,vector<int>& candidates,vector<vector<int>> &ans,vector<int> &ds){

        if(target==0){
            ans.push_back(ds);
            return;
        }
        
        // for the same index we will not element of same value
        // we are one by one pickinig our first element 
        // then second element
        for(int i=ind;i<candidates.size();i++){
            if(i>ind && candidates[i]==candidates[i-1]) continue;
            if(candidates[i]>target) break; // if not able to pick this , can't pick ahead (as sorted)
            ds.push_back(candidates[i]);
            combinationSum(i+1,target-candidates[i],candidates,ans,ds);
            ds.pop_back();
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        // in combination sum 2, combinations should not repeat
        vector<vector<int>> ans;
        vector<int> ds;
        sort(candidates.begin(),candidates.end()); // we want solution in lexiographic order
        // we will put subsequences
        combinationSum(0,target,candidates,ans,ds);
        return ans;
    }
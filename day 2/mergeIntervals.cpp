// merge intervals
// iterate through intervals and make temp interval , check if cur's start is less than temp's end if yes make 
// temp end to max of both ends  , else push temp in ans and make temp = cur
vector<vector<int>> ans;
        sort(intervals.begin(),intervals.end());
        vector<int> temp=intervals[0];
        for(auto cur:intervals){
            if(cur[0]<=temp[1]){
                temp[1]=max(temp[1],cur[1]);
            }
            else{
                ans.push_back(temp);
                temp=cur;
            }
        }
        ans.push_back(temp);
        return ans;
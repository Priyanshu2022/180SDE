    // sliding window maximum 
    // this approach -> tc= n , sc= k
    // storing in dq in decreasing order
    // storing index in dq
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ans;
        deque<int> dq;
        for(int i=0;i<nums.size();i++){
            if(!dq.empty() && dq.front()==i-k) dq.pop_front();// remove out of bound element
            while(!dq.empty() && nums[dq.back()]<=nums[i]) dq.pop_back();// removing all element less than i
            dq.push_back(i);
            if(i>=k-1) ans.push_back(nums[dq.front()]);// after we have checked for 1st subarray of size k
        }
        return ans;
    }
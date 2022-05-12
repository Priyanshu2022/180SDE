// maximum consecutive one's
// if(nums[i]==1) count++ and update ans
// else count=0
int findMaxConsecutiveOnes(vector<int>& nums) {
        int ans=0;
        int count=0;
        for(int i=0;i<nums.size();i++)
        {
            if(nums[i]==1)
            {
                count++;
                ans=max(count,ans);
            }
            else
            {
                count=0;
            }
        }
        return ans;
    }
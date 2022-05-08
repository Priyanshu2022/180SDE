vector<int> majorityElement(vector<int>& nums) {
        int num1,num2;
        int c1=0;
        int c2=0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]==num1) c1++;
            else if(nums[i]==num2) c2++;
            else if(c1==0) num1=nums[i],c1++;
            else if(c2==0) num2=nums[i], c2++;
            else{
                c1--;
                c2--;
            }
        }
        vector<int> ans;
        int k=0,l=0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]==num1) k++;
            if(nums[i]==num2) l++;
        }
        if(k>nums.size()/3) ans.push_back(num1);
        if(l>nums.size()/3) ans.push_back(num2);
        
        return ans;   
    }
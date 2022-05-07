// find a i such that a[i+1]>a[i] then find index2 such that index2>i then swap both then reverse index1+1 to end
void nextPermutation(vector<int>& nums) {
        int index1=-1;
        for(int i=nums.size()-2;i>=0;i--){
            if(nums[i+1]>nums[i]){
                index1=i;
                break;
            }
        }
        int index2=-1;
        if (index1 < 0) {
    	    reverse(nums.begin(), nums.end());
    	}else{
            for(int i=nums.size()-1;i>=0;i--){
            if(nums[i]>nums[index1]){
                index2=i;
                swap(nums[index1],nums[index2]);
                break;
            }
        }
        reverse(nums.begin()+index1+1,nums.end());
        }
        
    }
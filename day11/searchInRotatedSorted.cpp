int search(vector<int>& nums, int target) {
        int n=nums.size();
        int l=0;
        int h=n-1;
        int pivot=-1;
        while(l<h){
            int m=(l+h)/2;
            if(nums[m]>=nums[0]) l=m+1;
            else h=m;
        }
        pivot=l;
        if(nums[pivot]<=target && target<=nums[n-1]){
            l=pivot;
            h=n-1;
        }
        else{
            l=0;
            h=pivot-1;
        }
        while(l<=h){
            int m=(l+h)/2;
            if(nums[m]==target) return m;
            else if(nums[m]>target) h=m-1;
            else l=m+1;
        }
        return -1;
    }
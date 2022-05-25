TreeNode* solve(int start ,int end,vector<int> &nums){
        if(start>end) return NULL;
        int mid=(start+end)/2;
        TreeNode* root=new TreeNode(nums[mid]);
        root->left=solve(start,mid-1,nums);
        root->right=solve(mid+1,end,nums);
        return root;
    }
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        int mid=(nums.size()-1)/2;
        TreeNode* root=new TreeNode(nums[mid]);
        root->left=solve(0,mid-1,nums);
        root->right=solve(mid+1,nums.size()-1,nums);
        return root;
    }
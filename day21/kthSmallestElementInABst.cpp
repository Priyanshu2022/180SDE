// kth smallest element in bst
// maintain a count cur
// in inorder we get the sorted order ( access and check between left and right calls)
// call for left and if does not give -1 return left;
int solve(TreeNode* root,int k,int &cur){
        if(root==NULL) return -1;
        int left=solve(root->left,k,cur);
        if(left!=-1) return left;
        if(cur==k-1) return root->val;
        cur++;
        int right=solve(root->right,k,cur);
        return right;
    }
    int kthSmallest(TreeNode* root, int k) {
        int cur=0;
        return solve(root,k,cur);
    }
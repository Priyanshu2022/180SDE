// maximum sum path 
// maintain a answer variable
// ans = max(ans, left +right+root->val),if either of the side is negative , then we would have already made it zero
int solve(TreeNode* root,int &ans){
        if(root==NULL){
            return 0;
        }
        int left=max(0,solve(root->left,ans));
        int right=max(0,solve(root->right,ans));
        ans=max(ans,left+right+root->val);
        return root->val+max(left,right);
    }
public:
    int maxPathSum(TreeNode* root) {
        int ans=INT_MIN;
        solve(root,ans);
        return ans;
    }
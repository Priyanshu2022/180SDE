
bool solve(TreeNode* root,long mini,long maxi){
        if(root==NULL) return true;
        if(root->val <=mini || root->val >=maxi){
            return false;
        }
        return solve(root->left,mini,root->val)&&solve(root->right,root->val,maxi);
    }
public:
    bool isValidBST(TreeNode* root) {
        return solve(root,LONG_MIN,LONG_MAX);
    }
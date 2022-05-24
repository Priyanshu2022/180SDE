bool solve(TreeNode* first,TreeNode* second){
        if(first==NULL && second==NULL ) return true;
        if(first && second && first->val==second->val)
        return solve(first->left,second->right)&&solve(first->right,second->left);
        return false;
    }
public:
    bool isSymmetric(TreeNode* root) {
        return solve(root,root);
    }
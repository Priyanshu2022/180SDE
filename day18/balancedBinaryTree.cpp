pair<int,int> solve(TreeNode* root){
        if(root==NULL){
            return {true,0};
        }
        pair<int,int> left=solve(root->left);
        pair<int,int> right=solve(root->right);
        int dif=abs(left.second-right.second);
        if(left.first && right.first && dif<=1 ){
            return {true,max(left.second,right.second)+1};
        }
        else{
            return {false,max(left.second,right.second)+1};
        }
    }
    bool isBalanced(TreeNode* root) {
        return solve(root).first;
    }
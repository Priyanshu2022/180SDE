pair<int,int> solve(Node* root){
        if(root==NULL){
            return {true,0};
        }
        if(root->left==NULL && root->right==NULL){
            return {true,root->data};
        }
        pair<int,int> left=solve(root->left);
        pair<int,int> right=solve(root->right);
        int sum=left.second+right.second;
        if(left.first && right.first && sum==root->data) return {true,sum+root->data};
        else return {false,sum+root->data};
    }
    bool isSumTree(Node* root)
    {
         return solve(root).first;
    }
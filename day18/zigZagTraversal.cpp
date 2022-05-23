vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if(root==NULL) return ans;
        queue<TreeNode*> q;
        q.push(root);
        bool f=1;
        while(!q.empty()){
            int size=q.size();
            vector<int> temp(size);
            for(int i=0;i<size;i++){
                int index=(f)?i:(size-i-1);
                TreeNode* cur=q.front();
                q.pop();
                temp[index]=cur->val;
                if(cur->left){
                    q.push(cur->left);
                }
                if(cur->right){
                    q.push(cur->right);
                }
            }
            ans.push_back(temp);
            f=!f;
        }
        return ans;
    }

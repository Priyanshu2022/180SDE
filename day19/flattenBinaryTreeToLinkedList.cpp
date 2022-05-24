// if cur's left exists
    // cur ka predecessor nikalo (left jaake right jaate jao)
    // then uske right ko cur ke right se connect karo 
    // and mark cur's right as it's left
    // and mark left as null
    // after that move cur to right
    void flatten(TreeNode* root) {
        TreeNode* cur=root;
        while(cur){
            if(cur->left){
                TreeNode* pred=cur->left;
                while(pred->right){
                    pred=pred->right;
                }
                pred->right=cur->right;
                cur->right=cur->left;
                cur->left=NULL;
            }
            cur=cur->right;
        }
    }
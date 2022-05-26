class BSTIterator {
    stack <TreeNode*> st;
public:
    void pushAll(TreeNode* root){
        while(root!=NULL){
            st.push(root);
            root=root->left;
        }
    }
    BSTIterator(TreeNode* root) {
        pushAll(root);
    }
    
    int next() {
        TreeNode* cur=st.top();
        st.pop();
        pushAll(cur->right);
        return cur->val;
    }
    
    bool hasNext() {
        return !st.empty();
    }
};
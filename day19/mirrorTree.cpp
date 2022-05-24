void mirror(Node* node) {
        // code here
        if(!node) return ;
        swap(node->left,node->right);
        mirror(node->left);
        mirror(node->right);
    }
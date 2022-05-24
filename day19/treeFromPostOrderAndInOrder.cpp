Node* solve(int &index,int in[],int post[],int inorderStart,int inorderEnd,int n,map<int,int> &nodeToIndex){
        if(index<0 || inorderStart>inorderEnd){
            return NULL;
        }
        int element=post[index];
        Node* root=new Node(element);
        int position=nodeToIndex[element];
        index--;
        root->right=solve(index,in,post,position+1,inorderEnd,n,nodeToIndex);
        root->left=solve(index,in,post,inorderStart,position-1,n,nodeToIndex);
        return root;
    }
Node *buildTree(int in[], int post[], int n) {
        map<int,int> nodeToIndex;
        for(int i=0;i<n;i++){
            nodeToIndex[in[i]]=i;
        }
        int postOrderStart=n-1;
        return solve(postOrderStart,in,post,0,n-1,n,nodeToIndex);
}
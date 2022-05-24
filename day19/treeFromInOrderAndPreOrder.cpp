// we want to increment index and want to reflect that in every call 
    // tc after using map => nlong + n =nlogn
    // sc = n of map , stack space = 
    Node* solve(int &index,int in[],int pre[],int inorderStart,int inorderEnd,int n,map<int,int> &nodeToIndex){
        if(index>=n || inorderStart>inorderEnd){
            return NULL;
        }
        int element=pre[index];
        Node* root=new Node(element);
        int position=nodeToIndex[element];
        index++;
        root->left=solve(index,in,pre,inorderStart,position-1,n,nodeToIndex);
        root->right=solve(index,in,pre,position+1,inorderEnd,n,nodeToIndex);
        return root;
    }
    Node* buildTree(int in[],int pre[], int n)
    {
        // to get inorder index at O(1)
        map<int,int> nodeToIndex;
        for(int i=0;i<n;i++){
            nodeToIndex[in[i]]=i;
        }
        int preOrderStart=0;
        return solve(preOrderStart,in,pre,0,n-1,n,nodeToIndex);
    }
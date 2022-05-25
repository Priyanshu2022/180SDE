Node* connect(Node* root) {
        if(root==NULL) return NULL;
        queue<Node*> q;
        q.push(root);
        while(!q.empty()){
            int size=q.size();
            for(int i=0;i<size;i++){
              Node* cur=q.front();
              q.pop();
              if(i<size-1){
                  Node* nex=q.front();
                  cur->next=nex;
              }
              else cur->next=NULL;
                
                if(cur->left){
                    q.push(cur->left);
                }
                if(cur->right){
                    q.push(cur->right);
                }
            }
            
        }
        return root;
    }
// flatten means everythin from root should be in bottom
// merge root and flattened list from root->next
Node* merge(Node* root,Node* next){
        if(root==NULL) return next;
        if(next==NULL) return root;
        Node* ans=new Node(-1);
        Node* temp=ans;
        while(root!=NULL && next!=NULL){
            if(root->data>next->data){
                temp->bottom=next;
                next=next->bottom;
                temp=temp->bottom;
            }
            else{
                temp->bottom=root;
                root=root->bottom;
                temp=temp->bottom;
            }
        }
        if(root) temp->bottom=root;
        else temp->bottom=next;
        
        return ans->bottom;
    }
Node *flatten(Node *root)
{
   if(root==NULL) return NULL;
   return merge(root,flatten(root->next));
}
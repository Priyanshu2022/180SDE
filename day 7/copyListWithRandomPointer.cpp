// **********************************************************************************************************
// clone linked list with next and random pointer
// firstly we can use a map of node , node and create copy node's and make , original node's value as copy node
// OR
// OPTIMAL
// 3 steps
// 1st step
// make copy of each node and link them side by side in single list i.e. 1->1'->2->2'->3->3'
// 2nd step
// assign random pointers for copy nodes
// 3rd step
// restore the original list, and extract copy list (by assigning correct next pointers)
Node* copyRandomList(Node* head) {
        Node* iter=head;
        Node* front=head;
        
        while(iter!=NULL){
            front=iter->next;
            Node*copy=new Node(iter->val);
            iter->next=copy;
            copy->next=front;
            iter=front;
        }
        
        iter=head;
        while(iter!=NULL){
            if(iter->random!=NULL){
                iter->next->random=iter->random->next;
            }
            iter=iter->next->next;
        }
        
        iter=head;
        Node* pseudoHead=new Node(0);
        Node* copy=pseudoHead;
        while(iter!=NULL){
            front=iter->next->next;
            
            copy->next=iter->next;
            
            iter->next=front;
            
            iter=iter->next;
            copy=copy->next;
        }
        return pseudoHead->next;
    }
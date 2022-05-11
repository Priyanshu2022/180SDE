// slow (move by 1) and fast (move by 2) at head
// if slow and fast are equal break and point fast to head 
// now move both one by one
ListNode *detectCycle(ListNode *head) {
        ListNode* slow=head;
        ListNode* fast=head;
        if(head==NULL || head->next==NULL) return NULL;
        while(fast->next && fast->next->next){
            slow=slow->next;
            fast=fast->next->next;
            if(slow==fast) break;
        }
        if(fast->next==NULL || fast->next->next==NULL) return NULL;
        fast=head;
        while(fast!=slow){
            slow=slow->next;
            fast=fast->next;
        }
        return slow;
    }
// while(fast->next && fast->next->next)
bool hasCycle(ListNode *head) {
        ListNode* slow=head;
        ListNode* fast=head;
        if(head==NULL || head->next==NULL) return false;
        while(fast->next && fast->next->next){
            if(slow->next==fast->next->next) return true;
            slow=slow->next;
            fast=fast->next->next;
        }
        return false;
    }
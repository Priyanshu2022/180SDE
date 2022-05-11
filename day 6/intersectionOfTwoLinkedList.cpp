// a=headA
// b=headB
// run till a!=b
// if a is null move it to b's start,else to the next of a
// if b is null move it to a's start,alse to the next of b
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* a=headA;
        ListNode* b=headB;
        while(a!=b){
            a= a==NULL?headB:a->next;
            b= b==NULL?headA:b->next;
        }
        return a;
    }
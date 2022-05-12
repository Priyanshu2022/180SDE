// calculate length
// k=k%len
// make cur point to last
// and connect last to first
// now k=len-k
// move cur k times
// head as cur's next and cur->next as NULL
ListNode* rotateRight(ListNode* head, int k) {
        if(head==NULL || head->next==NULL || k==0) return head;
        int len=1;
        ListNode* cur=head;
        while(cur->next){
            cur=cur->next;
            len++;
        }
        cur->next=head;
        k=k%len;
        k=len-k;
        while(k--) cur=cur->next;
        head=cur->next;
        cur->next=NULL;
        return head;
    }
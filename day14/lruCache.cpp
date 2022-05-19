// doubly linked list and unordered map
class LRUCache {
public:
    class node{
        public:
        int key;
        int value;
        node* next;
        node* prev;
        node(int _key,int _value){
            key=_key;
            value=_value;
        }
    };
    node* head=new node(-1,-1);
    node* tail=new node(-1,-1);
    int cap;
    unordered_map<int,node*> mp;
    LRUCache(int capacity) {
        cap=capacity;
        head->next=tail;
        tail->prev=head;
    }
    void addnode(node* newnode){
        node* temp=head->next;
        head->next=newnode;
        newnode->next=temp;
        temp->prev=newnode;
        newnode->prev=head;
    }
    
    void deletenode(node* delnode){
        node* delprev=delnode->prev;
        node* delnext=delnode->next;
        delprev->next=delnext;
        delnext->prev=delprev;
    }
    
    int get(int key_) {
        if(mp.find(key_)!=mp.end()){
            node* resnode=mp[key_];
            int res=resnode->value;
            mp.erase(key_);
            deletenode(resnode);
            addnode(resnode);
            mp[key_]=head->next;
            return res;
        }
        return -1;
    }
    
    void put(int key_, int val) {
        if(mp.find(key_)!=mp.end()){
            deletenode(mp[key_]);
            mp.erase(key_);
        }
        if(mp.size()==cap){
            mp.erase(tail->prev->key);
            deletenode(tail->prev);
        }
        addnode(new node(key_,val));
        mp[key_]=head->next;
    }
};

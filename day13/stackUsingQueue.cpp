// stack using queue
class MyStack {
public:
    // queue<int> q1;
    // queue<int> q2;
    queue<int> q;
    MyStack() {
        
    }
    
    void push(int x) {
        // q2.push(x);
        // while(!q1.empty()){
        //     q2.push(q1.front());
        //     q1.pop();
        // }
        // swap(q1,q2);
        q.push(x);
        for(int i=0;i<q.size()-1;i++){
            q.push(q.front());
            q.pop();
        }
    }
    
    int pop() {
        // int ans=q1.front();
        // q1.pop();
        // return ans;
        int ans=q.front();
        q.pop();
        return ans;
    }
    
    int top() {
        // return q1.front();
        return q.front();
    }
    
    bool empty() {
        // return q1.empty();
        return q.empty();
    }
};
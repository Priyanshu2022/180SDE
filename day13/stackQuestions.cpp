// some questions of stack
// reverse a string using a stack
// stack has a property of reversing
string str="hello"
stack <char> st;
for(int i=0;i<str.lengnt();i++){
	st.push(str[i]);
}
string ans="";
while(!st.empty()){
	ans+=st.top();
	st.pop();
}
return ans; 

// delete middle element from stack
// jaate huye leke jana num
void solve(stack<int> &s,int n,int count){
        if(count==n/2){
            s.pop();
            return ;
        }
        int num=s.top();
        s.pop();
        solve(s,n,count+1);
        s.push(num);
    }
    void deleteMid(stack<int>&s, int sizeOfStack)
    {
        solve(s,sizeOfStack,0);
    }


 // similarly , insert a element at a bottom


 // reverse a stack using recursion
 // store the top element
// call the recursive fucntion for left over stack
// then put the top element at the bottom



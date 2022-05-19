// the celibrity problem
// celibrity knows no one and everyone knows celibrity
// {{0 1 0},
//  {0 0 0}, 
//  {0 1 0}}
// Brute force -> n^2 -> for(0 to no of person) {for( for checking if all elements of row zero ) for( column should be one except diagonal )} 
// Optimal 
// put every person in stack
// while( stack size is not 1)
// A=st.top() then pop , B=st.top() then pop()
// if(A knows B) push B
// if(B knows A) push A
// the last element left is potential answer 
int celebrity(vector<vector<int> >& M, int n) 
    {
        stack<int> st;
        for(int i=0;i<n;i++) st.push(i);
        while(st.size()>1){
            int a=st.top();
            st.pop();
            int b=st.top();
            st.pop();
            if(M[a][b]){
                st.push(b);
            }
            else st.push(a);
        }
        int potentialCelibrity=st.top();
        int count0=0;
        int count1=0;
        for(int i=0;i<n;i++){
            if(M[potentialCelibrity][i]==0) count0++;
            if(M[i][potentialCelibrity]==1) count1++;
        }
        if(count0==n && count1==n-1) return potentialCelibrity;
        else return -1;
    }
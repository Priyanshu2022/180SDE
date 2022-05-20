// Maximum of minimum for every window size
// 10 20 30 50 10 70 30
// 50 ka effect kaha kaha hoga, kaha  kaha par woh minimum hoga (only one window size)
// find next smaller element in left and right
// 10->7 ,20->3 ,30->2 ,50->1 ,10->7, 70->1 30->2
// 10 answer ho sakta hai 7 aur usse chote window ka 
// we have to find maximum of every window size
// make a vector of window size (1 to n)
// and store in 7 th window size 10 (update if find greater)
// in 2 windwo size store 30
// after storing iterate from back and if back is i+1 is greater than i update
vector <int> maxOfMin(int arr[], int n)
    {
        vector<int> nextSmaller(n,n),prevSmaller(n,-1);
        stack<int> s,t;
        for(int i=0;i<n;i++){
            while(!s.empty() && arr[s.top()]>=arr[i]) s.pop();
            if(!s.empty()) prevSmaller[i]=s.top();
            s.push(i);
        }
        for(int i=n-1;i>=0;i--){
            while(!t.empty() && arr[t.top()]>=arr[i]) t.pop();
            if(!t.empty()) nextSmaller[i]=t.top();
            t.push(i);
        }
        vector<int> ans(n,0);
        for(int i=0;i<n;i++) ans[nextSmaller[i]-prevSmaller[i]-2]=max(ans[nextSmaller[i]-prevSmaller[i]-2],arr[i]);
        for(int i=n-2;i>=0;i--){
            if(ans[i+1]>ans[i]) ans[i]=ans[i+1];
        }    
        return ans;
    }
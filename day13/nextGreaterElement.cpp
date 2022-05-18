vector<long long> nextLargerElement(vector<long long> arr, int n){
        stack<long long > st;
        st.push(-1);
        vector<long long> ans;
        for(int i=n-1;i>=0;i--){
            while(st.top()!=-1 && st.top()<=arr[i]) st.pop();
            ans.push_back(st.top());
            st.push(arr[i]);
        }
        reverse(ans.begin(),ans.end());
        return ans;
    }
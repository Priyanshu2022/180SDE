StockSpanner() {
        
    }
    int i=0;
    stack<pair<int,int>> st;
    int next(int price) {
        while(!st.empty() && st.top().first<=price) st.pop();
        int ans;
        if(st.empty()) ans=i+1;
        else ans=i-st.top().second;
        st.push({price,i});
        i++;
        return ans;
    }
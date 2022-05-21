// count and say
// for n=1 answer is 1
    // for n=2 answer is 11
    // for n=3 answer is 21
    // for n=4 answer is 1211
    string countAndSay(int n) {
        // recursive
        // if(n==1) return "1";
        // string s=countAndSay(n-1);
        // int count=1;
        // string res="";
        // for(int i=1;i<s.length();i++){
        //     if(s[i]==s[i-1]){
        //         count++;
        //     }
        //     else{
        //         res=res+to_string(count)+s[i-1];
        //         count=1;
        //     }
        // }
        // res+=to_string(count)+s[s.length()-1];
        // return res;
        string ans="1";
        for(int i=1;i<n;i++){
            string res="";
            int count=1;
            for(int i=1;i<ans.length();i++){
                if(ans[i]==ans[i-1]){
                    count++;
                }
                else{
                    res=res+to_string(count)+ans[i-1];
                    count=1;
                }
            }
            res+=to_string(count)+ans[ans.length()-1];
            ans=res;
        }
        return ans;
    }
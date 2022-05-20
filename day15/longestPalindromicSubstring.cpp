// longest palindromic substring
// for every element check if it making , a odd palindrome or a even palindrome
string longestPalindrome(string s) {
        int n=s.length();
        int MaxLen=0;
        string ans="";
        for(int i=0;i<s.length();i++){
            int j=i-1;
            int k=i+1;
            int tempLen=1;
            while(j>=0 && k<n){
                if(s[j]==s[k]){
                    tempLen+=2;
                }
                else break;
                j--;
                k++;
            }
            if(MaxLen<tempLen){
                MaxLen=tempLen;
                ans=s.substr(j+1,k-j-1);
            }
            if(i<n-1){
                tempLen=0;
                j=i;
                k=i+1;
                while(j>=0 && k<n){
                    if(s[j]==s[k]){
                        tempLen+=2;
                    }
                    else break;
                    j--;
                    k++;
                }
                if(MaxLen<tempLen){
                    MaxLen=tempLen;
                    ans=s.substr(j+1,k-j-1);
                }
            }
        }
        return ans;
    }
// start from back
// if mp[s[i]]<mp[s[i+1]] then subtract mp[s[i]]
// else add
int romanToInt(string s) {
        map<char,int> mp={{'M',1000},{'D',500},{'C',100},{'L',50},{'X',10},{'V',5},{'I',1}};
        int ans=mp[s[s.length()-1]];
        for(int i=s.length()-2;i>=0;i--){
            if(mp[s[i]]<mp[s[i+1]]) ans-=mp[s[i]];
            else ans+=mp[s[i]];
        }
        return ans;
    }
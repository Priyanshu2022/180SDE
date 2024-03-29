// char array
// increment for one and decrement for the other
bool isAnagram(string s, string t) {
        if(s.length()!=t.length()) return false;
        vector<int> v(26,0);
        for(int i=0;i<s.length();i++) v[s[i]-'a']++;
        for(int i=0;i<t.length();i++) v[t[i]-'a']--;
        for(int i=0;i<26;i++) if(v[i]!=0) return false;
        return true;
    }
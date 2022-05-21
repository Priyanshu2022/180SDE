// z[i] represents length of string after i (including i) , which is also present in prefix
// brute -> n^2
// optimal -> n
vector<int> z_function(string s) {
    int n = (int) s.length();
    vector<int> z(n);
    for (int i = 1, l = 0, r = 0; i < n; ++i) {
        if (i <= r) z[i] = min (r - i + 1, z[i - l]); // i before r(we have already calculated)
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) ++z[i]; // add if further presetn
        if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1; // update l and r if found greater
    }
    return z;
}
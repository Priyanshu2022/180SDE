// KMP algo
// find pattern in string -> n+m
// for that we have to find lps (longest prefix which is a suffix)
vector<int> lps(string s){
	int n=s.size();
	vector<int> pi(n,0);
	for(int i=1;i<n;i++){
		int j=pi[i-1];// previous char tak jo match kar paya wo aa jayega
		while(j>0 && s[i]!=s[j]) j=pi[j-1]; // piche wale se match nahi hua toh aur piche jao		
		if(s[i]==s[j]) j++;
		pi[i]=j;
	}
	return pi;
}

kmp (t,s){
	while(i<t.size()){
		if(t[i]==s[i]){
			j++;
			i++;
		}
		else{
			if(j!=0) j=prefix[j-1];
			else i++;
		}
		if(j==s.size()){
			return i-s.size();
		}
	}
}
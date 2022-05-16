// find nth root of m
// which number should be multiplied n times , so that we can get m
int power(int mid,int n,int m){
	    long long int ans=1;
	    for(int i=1;i<=n;i++){
	        ans*=mid;
	        if(ans>m) return m+2; // can never be the answer
	    }
	    return (int) ans;
	}
	int NthRoot(int n, int m)
	{
	    int l=0;
	    int h=m;
	    while(l<=h){
	        int mid=(l+h)/2;
	        int p=power(mid,n,m);
	        if(p==m) return mid;
	        else if(p>m) h=mid-1;
	        else l=mid+1;
	    }
	    return -1;
	} 
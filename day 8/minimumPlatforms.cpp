// sort both arrival and departure
int findPlatform(int arr[], int dep[], int n)
    {
    	int cur=1;
    	int ans=1;
    	sort(arr,arr+n);
    	sort(dep,dep+n);
    	int i=1;
    	int j=0;
    	while(i<n && j<n){
    	    if(arr[i]<=dep[j]){
    	        cur++;
    	        i++;
    	    }
    	    else{
    	        cur--;
    	        j++;
    	    }
    	    ans=max(ans,cur);
    	}
    	return ans;
    }
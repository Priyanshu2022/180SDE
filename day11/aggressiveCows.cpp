#include <bits/stdc++.h>
using namespace std;
 
bool isPossible(int a[],int mid,int c,int n){
	int cowCount=1;
	int lastPos=a[0];
	for(int i=0;i<n;i++){
		if(a[i]-lastPos>=mid){
			cowCount++;
			if(cowCount==c) return true;
			lastPos=a[i];
		}
	}
	return false;
}
 
int main() {
	int t;
	cin>>t;
	while(t--){
		int n,c;
		cin>>n>>c;
		int a[n];
		for(int i=0;i<n;i++) cin>>a[i];
		sort(a,a+n);
		int l=0;
		int h=a[n-1];
		int ans=-1;
		while(l<=h){
			int mid=l+(h-l)/2;
			if(isPossible(a,mid,c,n)){
				ans=mid;
				l=mid+1;
			}
			else h=mid-1;
		}
		cout<<ans<<endl;
	}
	return 0;
}  
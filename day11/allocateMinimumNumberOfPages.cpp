bool isPossible(int a[],int n,int m,int mid){
        int sum=0;
        int student=1;
        for(int i=0;i<n;i++){
            if(sum+a[i]<=mid){
                sum+=a[i];
            }
            else{
                student++;
                if(student>m || a[i]>mid) return false;
                sum=a[i];
            }
        }
        return true;
    }
    int findPages(int A[], int N, int M) 
    {
        int ans=-1;
        int sum=0;
        for(int i=0;i<N;i++) sum+=A[i];
        int l=0;
        int h=sum;
        while(l<=h){
            int mid=l+(h-l)/2;
            if(isPossible(A,N,M,mid)){
                ans=mid;
                h=mid-1;
            }
            else l=mid+1;
        }
        return ans;
    }
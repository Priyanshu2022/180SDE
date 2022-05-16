// median in row wise sorted matrix
// answer can be in the range 0 to 1e9
// for each mid calculate how many numbers are smaller than or equal to this
int countSmallerThanEqualTo(vector<int> &v,int mid){
        int l=0;
        int h=v.size()-1;
        while(l<=h){
            int m=(l+h)/2;
            if(v[m]<=mid) l=m+1;
            else h=m-1;
        }
        return l;
    }
    int median(vector<vector<int>> &matrix, int r, int c){
        int n=matrix.size();
        int m=matrix[0].size();
        int low=0;
        int high=1e9;
        while(low<=high){
            int mid=(low+high)/2;
            int count=0;
            for(int i=0;i<n;i++){
                count+=countSmallerThanEqualTo(matrix[i],mid);
            }
            if(count<=(n*m)/2) low=mid+1;
            else high=mid-1;
        }
        return low;
    }
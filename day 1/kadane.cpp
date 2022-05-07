// cursum+arr[i]>maxsum update if cursum <0 cursum=0
long long int maxSubarraySum(int arr[], int n){
        long long int maxSum=INT_MIN;
        long long int currentSum=0;;
        for(int i=0;i<n;i++){
            currentSum=currentSum+arr[i];
            if(currentSum>maxSum){
                maxSum=currentSum;
            }
            if(currentSum<0)currentSum=0;
        }
        return maxSum;
        
    }
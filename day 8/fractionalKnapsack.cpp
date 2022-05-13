// sort according profit by weight
// and take elements 
    static bool cmp(struct Item i1,struct Item i2){
        return (double)(((double)i1.value/(double)i1.weight)>((double)i2.value/(double)i2.weight));
    }
    double fractionalKnapsack(int W, Item arr[], int n)
    {
        sort(arr,arr+n,cmp);
        int cur=0;
        double ans=0;
        for(int i=0;i<n;i++){
            if(arr[i].weight+cur<=W){
                ans+=arr[i].value;
                cur+=arr[i].weight;
            }
            else{
                ans+=(double)((double)arr[i].value/(double)arr[i].weight)*(double)(W-cur);
                break;
            }
        }
        return ans;
    }
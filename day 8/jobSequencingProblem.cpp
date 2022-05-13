// sort the job's according to the profit(descending) and try to the job 
// as late as possible by taking occupied array , marked with -1 at the start
static bool cmp(struct Job a1,struct Job a2){
        return a1.profit>a2.profit;
    }
    vector<int> JobScheduling(Job arr[], int n) 
    { 
        sort(arr,arr+n,cmp);
        int num=0;
        int profit=0;
        int occupied[101];
        for(int i=0;i<101;i++) occupied[i]=-1;
        for(int i=0;i<n;i++){
            for(int j=arr[i].dead;j>0;j--){
                if(occupied[j]==-1){
                    occupied[j]=i;
                    num++;
                    profit+=arr[i].profit;
                    break;
                }
            }
        }
        vector<int> ans;
        ans.push_back(num);
        ans.push_back(profit);
        return ans;
    } 

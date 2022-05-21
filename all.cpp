vector<vector<int>> generate(int numRows) {
        vector<vector<int>> r(numRows);

        for (int i = 0; i < numRows; i++) {
            r[i].resize(i + 1);
            r[i][0] = r[i][i] = 1;
  
            for (int j = 1; j < i; j++)
                r[i][j] = r[i - 1][j - 1] + r[i - 1][j];
        }
        
        return r;
    }

// find a i such that a[i+1]>a[i] then find index2 such that index2>i then swap both then reverse index1+1 to end
void nextPermutation(vector<int>& nums) {
        int index1=-1;
        for(int i=nums.size()-2;i>=0;i--){
            if(nums[i+1]>nums[i]){
                index1=i;
                break;
            }
        }
        int index2=-1;
        if (index1 < 0) {
            reverse(nums.begin(), nums.end());
        }else{
            for(int i=nums.size()-1;i>=0;i--){
            if(nums[i]>nums[index1]){
                index2=i;
                swap(nums[index1],nums[index2]);
                break;
            }
        }
        reverse(nums.begin()+index1+1,nums.end());
        }
        
    }

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

// take 3 pointers low , mid and high . where 0 to low-1 will be zero and high +1 end will be 2
void sortColors(vector<int>& nums) {
        int low=0;
        int mid=0;
        int high=nums.size()-1;
        while(mid<=high){
            if(nums[mid]==0){
                swap(nums[mid],nums[low]);
                low++;
                mid++;
            }
            else if(nums[mid]==1){
                mid++;
            }
            else{
                swap(nums[mid],nums[high]);
                high--;
            }
        }
    }

// minimum till now 
int maxProfit(vector<int>& prices) {
        int mini=INT_MAX;
        int ans=0;
        for(int i=0;i<prices.size();i++)
        {
            if(prices[i]>=mini) ans=max(ans,prices[i]-mini);
            mini=min(mini,prices[i]);
        }
        return ans;
    }

// set matrix zero
// Assuming all the elements in the matrix are non-negative. Traverse through the matrix and if you find an element with value 0
// , then change all the elements in its row and column to -1, except when an element is 0. The reason for not changing other 
// elements to 0, but -1, is because that might affect other columns and rows. Now traverse through the matrix again and if an 
// element is -1 change it to 0, which will be the answer.

// rotate matrix -> transpose then reverse
void rotate(vector<vector<int>>& matrix) {
        for(int i=0;i<matrix.size();i++){
            for(int j=0;j<i;j++){
                swap(matrix[i][j],matrix[j][i]);
            }
        }
        for(int i=0;i<matrix.size();i++){
            reverse(matrix[i].begin(),matrix[i].end());
        }

    }


// merge intervals
// iterate through intervals and make temp interval , check if cur's start is less than temp's end if yes make 
// temp end to max of both ends  , else push temp in ans and make temp = cur
vector<vector<int>> ans;
        sort(intervals.begin(),intervals.end());
        vector<int> temp=intervals[0];
        for(auto cur:intervals){
            if(cur[0]<=temp[1]){
                temp[1]=max(temp[1],cur[1]);
            }
            else{
                ans.push_back(temp);
                temp=cur;
            }
        }
        ans.push_back(temp);
        return ans;

// merge two sorted arrays
// GAP method
//Initially take the gap as (m+n)/2;
// Take as a pointer1 = 0 and pointer2 = gap.
// Run a oop from pointer1 &  pointer2 to  m+n and whenever arr[pointer2]<arr[pointer1], just swap those.
// After completion of the loop reduce the gap as gap=gap/2.
// Repeat the process until gap>0.

int findDuplicate(vector<int>& nums) {
        int slow=nums[0];
        int fast=nums[0];
        do{
            slow=nums[slow];
            fast=nums[nums[fast]];
        }while(slow!=fast);
        fast=nums[0];
        while(slow!=fast){
            slow=nums[slow];
            fast=nums[fast];
        }
        return slow;
    }

// Count Inversion
// Arr[i]>arr[j] and i<j
// count while merging if arr[i]>arr[j] and i< j , all elements in right of a[i] is greater than a[i] so add mid-i
int merge(int arr[],int temp[],int left,int mid,int right)
{
    int inv_count=0;
    int i = left;
    int j = mid;
    int k = left;
    while((i <= mid-1) && (j <= right)){
        if(arr[i] <= arr[j]){
            temp[k++] = arr[i++];
        }
        else
        {
            temp[k++] = arr[j++];
            inv_count = inv_count + (mid - i);
        }
    }

    while(i <= mid - 1)
        temp[k++] = arr[i++];

    while(j <= right)
        temp[k++] = arr[j++];

    for(i = left ; i <= right ; i++)
        arr[i] = temp[i];
    
    return inv_count;
}

int merge_Sort(int arr[],int temp[],int left,int right)
{
    int mid,inv_count = 0;
    if(right > left)
    {
        mid = (left + right)/2;

        inv_count += merge_Sort(arr,temp,left,mid);
        inv_count += merge_Sort(arr,temp,mid+1,right);

        inv_count += merge(arr,temp,left,mid+1,right);
    }
    return inv_count;
}

int main()
{
    int arr[]={5,3,2,1,4};
    int n=5;
    int temp[n];
    int ans = merge_Sort(arr,temp,0,n-1);
    cout<<"The total inversions are "<<ans<<endl; 


    return 0;
}

// find repeating and missing
// hashing
// 1+2+3...n =n*(n+1)/2
// 1^2+2^2...=n(n+1)(2n+1)/6
// x-y=
// x^2-y^2=
// or xor
// xor all numbers with 1 to n
// we will be remained with x^y
// now check last **set** bit of x^y 
// and now separate the given array in two buckets according the last bit which is set ,
// now separte 1 to n in the same bucket 
// xor both buckets



// if the rows are sorted separately and columns separately
// we can start from last element of first column 
// and move left if target is less and down if target is more
    
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int low=0;
        int high=matrix.size()*matrix[0].size()-1;
        while(low<=high){
            int mid=(low+high)/2;
            if(matrix[mid/matrix[0].size()][mid%matrix[0].size()]>target){
                high=mid-1;
            }
            else if(matrix[mid/matrix[0].size()][mid%matrix[0].size()]<target){
                low=mid+1;
            }
            else{
                return true;
            }
        }
        return false;
    }


// pow(x,n)
// if n is even n=n/2 x=x*2
// if n is odd  n=n-1 ans=ans*x
double myPow(double x, int n) {
        double ans=1.0;
        long long int nn=n;
        if(nn<0) nn=nn*-1;
        while(nn){
            if(nn%2==1){
                nn=nn-1;
                ans*=x;
            }
            else{
                nn=nn/2;
                x*=x;
            }
        }
        if(n<0){
            ans=(double)1.0/(double)ans;
        }
        return ans;
    }

// el=-1
// count =0
// if(count==0) el=a[i]
// if(el==a[i]) count++
// else count--;

// return el
int majorityElement(vector<int>& a) {
        int count=0;
        int element=-1;
        for(int i=0;i<a.size();i++){
            if(count==0) element=a[i];
            if(element==a[i]) count++;
            else count--;
        }
        return element;
    }

// recursive dp or iterative dp
// or simply total paths =(n-1)+(m-1)=n+m-2
// choose n-1 from total choices
// n+m-2Cn-1
int func(int i,int j,int m,int n,vector<vector<int>>& dp){
        if(i>=n || j>=m) return 0;
        if(i==n-1 && j==m-1) return 1;
        if(dp[i][j]!=-1) return dp[i][j];
        int down=func(i+1,j,m,n,dp);
        int right=func(i,j+1,m,n,dp);
        return dp[i][j]=right+down;
    }
    int uniquePaths(int m, int n) {
        int i=0;
        int j=0;
        vector<vector<int>> dp(m,vector<int> (n,-1));
        // return func(i,j,m,n,dp);
        dp[0][0]=1;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(i==0 && j==0) continue;
                if(i==0) dp[i][j]=dp[i][j-1];
                else if(j==0) dp[i][j]=dp[i-1][j];
                else dp[i][j]=dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }


int merge(vector<int>& nums,int start,int end,int mid){
        int count=0;
        int l=start,h=mid+1;
        while(l<=mid && h<=end){
            if((long long int)nums[l]>(long long int)2*nums[h]){
                count+=(mid-l+1);
                h++;
            }
            else l++;
        }
        int temp[end-start+1];
        int t=0;
        l=start,h=mid+1;
        while(l<=mid && h<=end){
            if(nums[l]>nums[h]){
                temp[t++]=nums[h++];
            }
            else temp[t++]=nums[l++];
        }
        while(l<=mid) temp[t++]=nums[l++];
        while(h<=end) temp[t++]=nums[h++];
        for(int i=0;i<end-start+1;i++){
            nums[start+i]=temp[i];
        }
        return count;
    }
    int mergeSort(vector<int>& nums,int start,int end){
        if(start>=end) return 0;
        int mid=(start+end)/2;
        int count=mergeSort(nums,start,mid);
        count+=mergeSort(nums,mid+1,end);
        count+=merge(nums,start,end,mid);
        return count;
    }
    int reversePairs(vector<int>& nums) {
        return mergeSort(nums,0,nums.size()-1);
    }



vector<int> majorityElement(vector<int>& nums) {
        int num1,num2;
        int c1=0;
        int c2=0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]==num1) c1++;
            else if(nums[i]==num2) c2++;
            else if(c1==0) num1=nums[i],c1++;
            else if(c2==0) num2=nums[i], c2++;
            else{
                c1--;
                c2--;
            }
        }
        vector<int> ans;
        int k=0,l=0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]==num1) k++;
            if(nums[i]==num2) l++;
        }
        if(k>nums.size()/3) ans.push_back(num1);
        if(l>nums.size()/3) ans.push_back(num2);
        
        return ans;   
    }


// 2 sum

// 4 sum

// check subarray sum 
// if it is update
// else is already present in map update length
// or put it in map
int maxLen(vector<int>&A, int n)
    {   
        unordered_map<int,int> mp;
        int sum=0;
        int ans=0;
        for(int i=0;i<A.size();i++){
            sum+=A[i];
            if(sum==0) ans=max(ans,i+1);
            else{
                if(mp[sum]==0){
                    mp[sum]=i+1;
                }
                else{
                    ans=max(ans,i+1-mp[sum]);
                }
            }
        }
        return ans;
    }

// check if cur-1 present in map
// if present , continue;
// else count from there
int longestConsecutive(vector<int>& nums) {
        unordered_map<int,int> mp;
        for(auto cur:nums){
            mp[cur]++;
        }
        int ans=0;
        for(auto cur:nums){
            if(mp[cur-1]!=0) continue;
            else{
                int count=0;
                int temp=cur;
                while(mp[temp]!=0){
                    count++;
                    temp++;
                }
                ans=max(count,ans);
            }
        }
        return ans;
    }


// left to right is our substring
// if s[right] is in map update left
// map stores char and its index
// ans=max(ans,right-left+1)
int lengthOfLongestSubstring(string s) {
         unordered_map<char,int> mp;
         int left=0;
         int right=0;
         int n =s.length();
         int ans=0;
        while(right<n){
            if(mp.find(s[right])!=mp.end()) left=max(left,mp[s[right]]+1);
            mp[s[right]]=right; 
            ans=max(ans,right-left+1);
            right++;
        }
        return ans;
    }


// subarray with given xor
// unordered_map<int,int> =>xor , no of time
// let the given xor be k
// y^k=x (...(y)...(k)==(x))
// y=k^x
// how many subarrays with xor y is present when xor till now is x


ListNode* reverseList(ListNode* head) {
        ListNode* newHead=NULL;
        while(head){
            ListNode* temp=head->next;
            head->next=newHead;
            newHead=head;
            head=temp;
        }
        return newHead;
    }


ListNode* middleNode(ListNode* head) {
        ListNode* slow=head;
        ListNode* fast=head;
        while(fast!=NULL && fast->next!=NULL){
            slow=slow->next;
            fast = fast->next->next;
        }
        return slow;
    }


ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* temp=new ListNode(-1);
        ListNode* ans=temp;

        while(list1 and list2){
            if((list1->val)>(list2->val)){
                temp->next=list2;
                list2=list2->next;  
            }
            else{
                temp->next=list1;
                list1=list1->next;
            }
            temp=temp->next;
        }
        while(list1){
            temp->next=list1;
            list1=list1->next;
            temp=temp->next;
        }
        while(list2){
            temp->next=list2;
            list2=list2->next;
            temp=temp->next;
        }
        return ans->next;
    } 


// run fast pointer for n times
// then run slow and fast untill fast->next
ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* start=new ListNode();
        start->next=head;
        ListNode* slow=start;
        ListNode* fast=start;
        for(int i=0;i<n;i++){
            fast=fast->next;
        }
        while(fast->next){
            fast=fast->next;
            slow=slow->next;
        }
        slow->next=slow->next->next;
        return start->next;
    } 


void deleteNode(ListNode* node) {
        node->val=node->next->val;
        node->next=node->next->next;
    }


// while((l1||l2)||carry)
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* temp=new ListNode();
        ListNode* dummy=temp;
        int carry=0;
        while((l1||l2) || carry){
            int sum=0;
            if(l1!=NULL){
                sum+=l1->val;
                l1=l1->next;
            }
            if(l2!=NULL){
                sum+=l2->val;
                l2=l2->next;
            }
            sum+=carry;
            ListNode* node=new ListNode(sum%10);
            carry=sum/10;
            temp->next=node;
            temp=temp->next;
        }
        return dummy->next;
     }


// a=headA
// b=headB
// run till a!=b
// if a is null move it to b's start,else to the next of a
// if b is null move it to a's start,alse to the next of b
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* a=headA;
        ListNode* b=headB;
        while(a!=b){
            a= a==NULL?headB:a->next;
            b= b==NULL?headA:b->next;
        }
        return a;
    }

// while(fast->next && fast->next->next)
bool hasCycle(ListNode *head) {
        ListNode* slow=head;
        ListNode* fast=head;
        if(head==NULL || head->next==NULL) return false;
        while(fast->next && fast->next->next){
            if(slow->next==fast->next->next) return true;
            slow=slow->next;
            fast=fast->next->next;
        }
        return false;
    }


// go to middle of ll(one before the middle) then reverse it from there
// make slow's next to reverseLL
// then check untill slow
ListNode* reverseList(ListNode* head) {
        ListNode* newHead=NULL;
        while(head){
            ListNode* temp=head->next;
            head->next=newHead;
            newHead=head;
            head=temp;
        }
        return newHead;
    }
    bool isPalindrome(ListNode* head) {
        ListNode* slow=head;
        ListNode* fast=head;
        while(fast->next && fast->next->next){
            slow=slow->next;
            fast=fast->next->next;
        }
        slow->next=reverseList(slow->next);
        slow=slow->next;
        while(slow){
            if(slow->val!=head->val) return false;
            slow=slow->next;
            head=head->next;
        }
        return true;
    }


// merge sort in linked list
// find mid point
// separate two linked list
// call recursive merge sort in left and right
// then merge two sorted linked list

// 1->2->3->4->5->6    k=2
// first check if from current head there are k element present or not
// reverse first k elements using counter
// check if not cur node is not null again call for recursion and make it head's next
// at last return prev (new head first reversed list)
ListNode* reverseKGroup(ListNode* head, int k) {
        if(head==NULL) return NULL;
        
        ListNode* temp=head;
        for(int i=0;i<k;i++){
            if(!temp) return head;
            temp=temp->next;
        }
        
        int count=0;
        ListNode* next=NULL;
        ListNode* cur=head;
        ListNode* prev=NULL;
        while(cur!=NULL && count<k){
            next=cur->next;
            cur->next=prev;
            prev=cur;
            cur=next;
            count++;
        }
        if(cur!=NULL){
            head->next=reverseKGroup(cur,k);
        }
        return prev;
    }


// flatten means everythin from root should be in bottom
// merge root and flattened list from root->next
Node* merge(Node* root,Node* next){
        if(root==NULL) return next;
        if(next==NULL) return root;
        Node* ans=new Node(-1);
        Node* temp=ans;
        while(root!=NULL && next!=NULL){
            if(root->data>next->data){
                temp->bottom=next;
                next=next->bottom;
                temp=temp->bottom;
            }
            else{
                temp->bottom=root;
                root=root->bottom;
                temp=temp->bottom;
            }
        }
        if(root) temp->bottom=root;
        else temp->bottom=next;
        
        return ans->bottom;
    }
Node *flatten(Node *root)
{
   if(root==NULL) return NULL;
   return merge(root,flatten(root->next));
}


// slow (move by 1) and fast (move by 2) at head
// if slow and fast are equal break and point fast to head 
// now move both one by one
ListNode *detectCycle(ListNode *head) {
        ListNode* slow=head;
        ListNode* fast=head;
        if(head==NULL || head->next==NULL) return NULL;
        while(fast->next && fast->next->next){
            slow=slow->next;
            fast=fast->next->next;
            if(slow==fast) break;
        }
        if(fast->next==NULL || fast->next->next==NULL) return NULL;
        fast=head;
        while(fast!=slow){
            slow=slow->next;
            fast=fast->next;
        }
        return slow;
    }


// trapping rain water
// brute can be , we can calculate water stored at each index
// min(max_in_left,max_in_right)-hight_of_current_index
// time -> n*n , space -> 1
// if we use prefix and suffix  max 
// time -> n , space -> n
// OPTIMAL
// l=0,r=n-1,res=0
// leftMax=0,rightMax=0
// if(a[l]<=a[r]) if(a[l]>=leftmax) leftmax=a[l] else res+=(leftMax-a[l]); l++(for both)
// else if(a[r]>=rightmax) rightmax=a[r] else res+=(rightmax-a[r]) ; r--
// basically we are doing min(leftmax,rightmax)-a[i]
int trap(vector<int>& a) {
        int l=0;
        int r=a.size()-1;
        int ans=0;
        int leftMax=0;
        int rightMax=0;
        while(l<=r){
            if(a[l]<=a[r]){
                if(a[l]<=leftMax) ans+=leftMax-a[l];
                else leftMax=a[l];
                l++;
            }
            else{
                if(a[r]<=rightMax) ans+=rightMax-a[r];
                else rightMax=a[r];
                r--;
            }
        }
        return ans;
    }


// calculate length
// k=k%len
// make cur point to last
// and connect last to first
// now k=len-k
// move cur k times
// head as cur's next and cur->next as NULL
ListNode* rotateRight(ListNode* head, int k) {
        if(head==NULL || head->next==NULL || k==0) return head;
        int len=1;
        ListNode* cur=head;
        while(cur->next){
            cur=cur->next;
            len++;
        }
        cur->next=head;
        k=k%len;
        k=len-k;
        while(k--) cur=cur->next;
        head=cur->next;
        cur->next=NULL;
        return head;
    }


// maximum consecutive one's
// if(nums[i]==1) count++ and update ans
// else count=0
int findMaxConsecutiveOnes(vector<int>& nums) {
        int ans=0;
        int count=0;
        for(int i=0;i<nums.size();i++)
        {
            if(nums[i]==1)
            {
                count++;
                ans=max(count,ans);
            }
            else
            {
                count=0;
            }
        }
        return ans;
    }



// 3 sum


// if( a[i]!=a[j]) i++ and a[i]=a[j];
int removeDuplicates(vector<int>& a) {
        if(a.size()==0) return 0;
        int i=0;
        for(int j=0;j<a.size();j++){
            if(a[i]!=a[j]){
                i++;
                a[i]=a[j];
            }
        }
        return i+1;
    }


// **********************************************************************************************************
// clone linked list with next and random pointer
// firstly we can use a map of node , node and create copy node's and make , original node's value as copy node
// OR
// OPTIMAL
// 3 steps
// 1st step
// make copy of each node and link them side by side in single list i.e. 1->1'->2->2'->3->3'
// 2nd step
// assign random pointers for copy nodes
// 3rd step
// restore the original list, and extract copy list (by assigning correct next pointers)
Node* copyRandomList(Node* head) {
        Node* iter=head;
        Node* front=head;
        
        while(iter!=NULL){
            front=iter->next;
            Node*copy=new Node(iter->val);
            iter->next=copy;
            copy->next=front;
            iter=front;
        }
        
        iter=head;
        while(iter!=NULL){
            if(iter->random!=NULL){
                iter->next->random=iter->random->next;
            }
            iter=iter->next->next;
        }
        
        iter=head;
        Node* pseudoHead=new Node(0);
        Node* copy=pseudoHead;
        while(iter!=NULL){
            front=iter->next->next;
            
            copy->next=iter->next;
            
            iter->next=front;
            
            iter=iter->next;
            copy=copy->next;
        }
        return pseudoHead->next;
    }



// sort the meeting according to end time
// take the first meeting in ans
// and mark it's end time as limit
// now iterate and check if currents start time is greater than limit if yes ans++
// and update limit as cur's end time
struct meeting{
        int s;
        int e;
        int p;
    };
    static bool cmp(struct meeting m1, struct meeting m2){
        if(m1.e<m2.e) return true;
        else if(m1.e>m2.e) return false;
        else if(m1.p<m2.p) return true;
        else return false;
    }
    int maxMeetings(int start[], int end[], int n)
    {
        struct meeting meet[n];
        for(int i=0;i<n;i++){
            meet[i].s=start[i];
            meet[i].e=end[i];
            meet[i].p=i+1;
        }
        sort(meet,meet+n,cmp);
        int ans=1;
        int limit=meet[0].e;
        for(int i=1;i<n;i++){
            if(meet[i].s>limit){
                limit=meet[i].e;
                ans++;
            }
        }
        return ans;
    }


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


// take the current and do not change i or leave the current and change the i
int solve(int i,int amount,vector<int>& coins,vector<vector<int>> &dp){
        if(amount==0) return 0;
        if(amount<0 || i>=coins.size()) return INT_MAX-1;
        if(dp[i][amount]!=-1) return dp[i][amount];
        return dp[i][amount]=min(1+solve(i,amount-coins[i],coins,dp),solve(i+1,amount,coins,dp));
    }
    int coinChange(vector<int>& coins, int amount) {
        vector<vector<int>> dp(coins.size(),vector<int>(amount+1,-1));
        int ans=solve(0,amount,coins,dp);
        if(ans==INT_MAX-1) return -1;
        else return ans;
    }



// print all subsequences
// take , not take
// time -> 2^n
// space -> n (at max n recursion calls will be waiting in the stack space)
void solve(vector<vector<int>> &ans,vector<int> ds,int index,vector<int> arr){
    if(index==arr.size()){
        ans.push_back(ds);
        return;
    }
    ds.push_back(arr[index]);
    solve(ans,ds,index+1,arr);
    ds.pop_back();
    solve(ans,ds,index+1,arr);
}

// if want just one subsequence of given sum
// by bool and returning , we are avoiding further recursion calls
bool solve(vector<vector<int>> &ans,vector<int> ds,int index,vector<int> arr,int sum){
    if(index==arr.size()){
        if(sum==0){
            ans.push_back(ds);
            return true;
        }
        return false;
    }
    ds.push_back(arr[index]);
    if(solve(ans,ds,index+1,arr,sum-arr[index])) return true;
    ds.pop_back();
    if(solve(ans,ds,index+1,arr)) return true;
}

// no of subsequence with sum, sum
// base case if condition satisfy return 1 else 0
// at last return solve(included)+solve(not included)
// tc=2^n sc=


// ***********************************
// print all subsequence using power set
// space complexity is constant
// time complexity is 2^n * n
// a b c
// 0 0 0 represent no char taken
// 0 0 1 represent a is taken
// for(num=0 to 2^n-1 (i.e (1<<n)-1)){
//  sub=""
//  for(i=0 to n-1){
//      if(num&(i<<1)){
//          sub+=s[i];
//      }
//  }
//  ans push_back or print
// }






void findcombination(int index,int target,vector<int>& candidates,vector<vector<int>>& ans,vector<int>& ds){
        if(index==candidates.size()){
            if(target==0){
                ans.push_back(ds);
            }
            return;
        }
        if(candidates[index]<=target){
            ds.push_back(candidates[index]);
            findcombination(index,target-candidates[index],candidates,ans,ds);
            ds.pop_back();
        }
        findcombination(index+1,target,candidates,ans,ds);
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> ans;
        vector<int> ds;
        findcombination(0,target,candidates,ans,ds);
        return ans;
    }




// tc=2^n*k (k  is average length)
// sc=k*x (x is total combinations) (ignorign auxiliary space)
// cannot pick a element more than once
    void combinationSum(int ind,int target,vector<int>& candidates,vector<vector<int>> &ans,vector<int> &ds){

        if(target==0){
            ans.push_back(ds);
            return;
        }
        
        // for the same index we will not element of same value
        // we are one by one pickinig our first element 
        // then second element
        for(int i=ind;i<candidates.size();i++){
            if(i>ind && candidates[i]==candidates[i-1]) continue;
            if(candidates[i]>target) break; // if not able to pick this , can't pick ahead (as sorted)
            ds.push_back(candidates[i]);
            combinationSum(i+1,target-candidates[i],candidates,ans,ds);
            ds.pop_back();
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        // in combination sum 2, combinations should not repeat
        vector<vector<int>> ans;
        vector<int> ds;
        sort(candidates.begin(),candidates.end()); // we want solution in lexiographic order
        // we will put subsequences
        combinationSum(0,target,candidates,ans,ds);
        return ans;
    }



// n queen
// iterate in every column
// try to put in every column of a row
// check if valid
// we can optimize this using hashing
// using if in same column a queen is present we can skip it and sum of i , j in lower diagonal is same so we can check 
// if present in lower diagonal 
// for upper diagonal n-1 + col -row is same
bool isValid(int row,int col,vector<string> &board,int n){
        int i=row;
        // checking all rows in that column
        while(i>=0){
            if(board[i][col]=='Q') return false;
            i--;
        }
        i=row;
        int j=col;
        while(i>=0 && j>=0){
            if(board[i][j]=='Q') return false;
            i--;
            j--;
        }
        i=row;
        j=col;
        while(i>=0 && j<n){
            if(board[i][j]=='Q') return false;
            i--;
            j++;
        }
        return true;
    }
    void solve(vector<vector<string>> &ans,vector<string> &board,int n,int row){
        if(row==n){
            ans.push_back(board);
            return;
        }
        for(int i=0;i<n;i++){
            if(isValid(row,i,board,n)){
                board[row][i]='Q';
                solve(ans,board,n,row+1); // place and try in next row
                board[row][i]='.';
            }
        }
    }
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> ans;
        vector<string> board(n);
        string s(n,'.');
        for(int i=0;i<n;i++){
            board[i]=s;
        }
        solve(ans,board,n,0);
        return ans;
    }




// print all permutations
// first we can use a map , which will store which element  i have taken
// tc=n! * n    sc=n + n
void recurPermute(vector < int > & ds, vector < int > & nums, vector < vector < int >> & ans, int freq[]) {
      if (ds.size() == nums.size()) {
        ans.push_back(ds);
        return;
      }
      for (int i = 0; i < nums.size(); i++) {
        if (!freq[i]) {
          ds.push_back(nums[i]);
          freq[i] = 1;
          recurPermute(ds, nums, ans, freq);
          freq[i] = 0;
          ds.pop_back();
        }
      }
    }
  public:
    vector < vector < int >> permute(vector < int > & nums) {
      vector < vector < int >> ans;
      vector < int > ds;
      int freq[nums.size()];
      for (int i = 0; i < nums.size(); i++) freq[i] = 0;
      recurPermute(ds, nums, ans, freq);
      return ans;
    }


// better answer
// tc n!*n
// sc -> only to store answer n! and stack space of n
void solve(int index,vector<vector<int>> &ans,vector<int> &nums){
        if(index==nums.size()){
            ans.push_back(nums);
            return;
        }
        // swapping index with i , so that we can have all elements possible at the start
        for(int i=index;i<nums.size();i++){
            swap(nums[index],nums[i]);
            solve(index+1,ans,nums);
            swap(nums[index],nums[i]);
        }
    }
    
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ans;
        solve(0,ans,nums);
        return ans;
    }



// recursion
// stack space contains funtion yet to be completed
// space complexity is number of funtion call waiting at maximum
// time complexity number of funtion calls

// fibonacci
// time complexity=2^n
// space complexity=



// subsets
void solve(int index,vector<int> nums,vector<vector<int>> &ans,vector<int> &temp){
        if(index==nums.size()){
            ans.push_back(temp);
            return;
        }
        solve(index+1,nums,ans,temp);
        temp.push_back(nums[index]);
        solve(index+1,nums,ans,temp);
        temp.pop_back();
    }
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> temp;
        solve(0,nums,ans,temp);
        return ans;
    }



// subsets 2
// The solution set must not contain duplicate subsets
void func(int index,vector<vector<int>> &ans,vector<int> &ds,vector<int>& nums){
        ans.push_back(ds);
        for(int i=index;i<nums.size();i++){
            if(i>index && nums[i]==nums[i-1]) continue;
            ds.push_back(nums[i]);
            func(i+1,ans,ds,nums);
            ds.pop_back();
        }
    }
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> ds;
        sort(nums.begin(),nums.end());
        func(0,ans,ds,nums);
        return ans;
    }


// sudoku solver
// bool function (we want just one answer)
// iterate and check where empty cell is present
// try to put char from 1 to 9 
// if valid put and call for next 
// if this returns true return true;
// else mark again empty
// if after putting 1 to 9 in empty it does not return true  , 
// then return false ** (means this config is not correct) backtrack
// if every cell is marked means at the end return true
bool isValid(int row,int col,char check,vector<vector<char>> &boards){
        for(int i=0;i<9;i++){
            if(boards[row][i]==check) return false;
            if(boards[i][col]==check) return false;
            if(boards[3*(row/3)+i/3][3*(col/3)+i%3]==check) return false;
        }
        return true;
    }
    bool solve(vector<vector<char>>&boards){
        for(int i=0;i<boards.size();i++){
            for(int j=0;j<boards[0].size();j++){
                if(boards[i][j]=='.'){
                    for(char c='1';c<='9';c++){
                        if(isValid(i,j,c,boards)){
                            boards[i][j]=c;
                            if(solve(boards)==true){
                                return true;
                            }
                            else{
                                boards[i][j]='.';
                            }
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }
    void solveSudoku(vector<vector<char>>& boards) {
        bool temp=solve(boards);
    }




// is it possible to colour the graph with m colours
// try to color current node with every color
// bool funtion because only check if we can
bool issafe(int node,int color[],bool graph[101][101],int n,int i){
    for(int k=0;k<n;k++){
        if(k!=node && graph[k][node]==1 && color[k]==i) return false;
    }
    return true;
}   
bool solve(int node,bool graph[101][101],int m,int n,int color[]){
    if(node==n) return true;
    for(int i=1;i<=m;i++){
        if(issafe(node,color,graph,n,i)){
            color[node]=i;
            if(solve(node+1,graph,m,n,color)) return true;
            color[node]=0;
        }
    }
    return false;
}
bool graphColoring(bool graph[101][101], int m, int V)
{
    int color[V]={0};
    return solve(0,graph,m,V,color);
}



// Given a string s, partition s such that every substring of the partition is a palindrome. 
// Return all possible palindrome partitioning of s.
bool isPalindrome(int start,int end,string &s){
        while(start<=end){
            if(s[start++]!=s[end--]) return false;
        }
        return true;
    }
    void solve(int index,string &s,vector<vector<string>> &ans,vector<string> &path){
        if(index==s.length()){
            ans.push_back(path);
            return ;
        }
        for(int i=index;i<s.length();i++){
            if(isPalindrome(index,i,s)){
                path.push_back(s.substr(index,i-index+1));
                solve(i+1,s,ans,path);
                path.pop_back();
            }
        }
    }
    vector<vector<string>> partition(string s) {
        vector<vector<string>> ans;
        vector<string> path;
        solve(0,s,ans,path);
        return ans;
    }




void solve(vector<string> &ans,string s,int i,int j,vector<vector<int>> &m,int n,vector<vector<int>> &vis){
        if(i==n-1 && j==n-1 ){
            ans.push_back(s);
            return;
        }
        if(i>=n || i<0 || j>=n || j<0 || m[i][j]==0 || vis[i][j]==1){
            return;
        }
        vis[i][j]=1;
        solve(ans,s+'D',i+1,j,m,n,vis);
        solve(ans,s+'U',i-1,j,m,n,vis);
        solve(ans,s+'R',i,j+1,m,n,vis);
        solve(ans,s+'L',i,j-1,m,n,vis);
        vis[i][j]=0;
    }
    public:
    vector<string> findPath(vector<vector<int>> &m, int n) {
        
        vector<string> ans;
        if(m[n-1][n-1]==0) return ans;
        string s="";
        vector<vector<int>> vis(n,vector<int>(n,0));
        solve(ans,s,0,0,m,n,vis);
        return ans;
    }



// word break 
// return true if we can break s , into space separated strings of words of wordDict
bool solve(int i,string s,unordered_map<string,int> &dict,vector<int> &dp){
        if(i==s.length()) return true;
        if(dp[i]!=-1) return dp[i];
        string temp;
        for(int j=i;j<s.length();j++){
            temp+=s[j];
            if(dict.find(temp)!=dict.end()){
                if(solve(j+1,s,dict,dp)) return dp[i]=true;
            }
        }
        return dp[i]=false;
    }
    bool wordBreak(string s, vector<string>& wordDict) {
        vector<int> dp(s.length(),-1);
        unordered_map<string,int> dict;
        for(int i=0;i<wordDict.size();i++) dict[wordDict[i]]++;
        return solve(0,s,dict,dp);
    }




// permutation sequence
// 1 2 3 4
// 1 {2 3 4} -> 6 -> (n-1)!
// 2 {1 3 4} -> 6
// 3 {1 2 4} ....
// 4 {1 2 3} ....
string getPermutation(int n, int k) {
        vector<int> numbers;
        int fact=1;
        // calculating (n-1)!
        for(int i=1;i<n;i++){
            fact=fact*i;
            numbers.push_back(i);
        }
        numbers.push_back(n);
        string ans="";
        k=k-1; // zero based indexing 
        while(true){
            ans+=to_string(numbers[k/fact]);
            numbers.erase(numbers.begin()+k/fact);
            if(numbers.size()==0) break;
            k=k%fact;
            fact=fact/numbers.size();
        }
        return ans;
    }



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



// kth element of two sorted arrays
int kthElement(int arr1[], int arr2[], int n, int m, int k)
    {
        if(n>m) return kthElement(arr2,arr1,m,n,k);
        int l=max(0,k-m);
        int h=min(n,k);
        while(l<=h){
            int cut1=(l+h)/2;
            int cut2=k-cut1;
            int l1=(cut1==0)?INT_MIN:arr1[cut1-1];
            int l2=(cut2==0)?INT_MIN:arr2[cut2-1];
            int r1=(cut1==n)?INT_MAX:arr1[cut1];
            int r2=(cut2==m)?INT_MAX:arr2[cut2];
            if(l1<=r2 && l2<=r1) return max(l1,l2);
            else if(l1>r2){
                h=cut1-1;
            }
            else l=cut1+1;
        }
        return -1;
    }





// median of two sorted arrays
// l=0, h=n1;
// cut1=(l+h)/2 , cut2=(n1+n2+1)/2-cut1
// if l1 <=r2 && l2<=r1 check if n1+n2 is odd or even
// if( l1>r2) then we will have to reduce therefore h=cut1-1
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int n1=nums1.size();
        int n2=nums2.size();
        if(n1>n2) return findMedianSortedArrays(nums2,nums1);
        int l=0;
        int h=n1;
        while(l<=h){
            int cut1=(l+h)/2;
            int cut2=(n1+n2+1)/2 -cut1;
            
            int l1=(cut1==0)?INT_MIN:nums1[cut1-1];
            int l2=(cut2==0)?INT_MIN:nums2[cut2-1];
            int r1=(cut1==n1)?INT_MAX:nums1[cut1];
            int r2=(cut2==n2)?INT_MAX:nums2[cut2];
            
            if(l1<=r2 && l2<=r1){
                if((n1+n2)%2==0){
                    return (max(l1,l2)+min(r1,r2))/2.0;
                }
                else return max(l1,l2);
            }
            else if(l1>r2){
                h=cut1-1;
            }
            else l=cut1+1;
        }
        return -1;
    }



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





int search(vector<int>& nums, int target) {
        int n=nums.size();
        int l=0;
        int h=n-1;
        int pivot=-1;
        while(l<h){
            int m=(l+h)/2;
            if(nums[m]>=nums[0]) l=m+1;
            else h=m;
        }
        pivot=l;
        if(nums[pivot]<=target && target<=nums[n-1]){
            l=pivot;
            h=n-1;
        }
        else{
            l=0;
            h=pivot-1;
        }
        while(l<=h){
            int m=(l+h)/2;
            if(nums[m]==target) return m;
            else if(nums[m]>target) h=m-1;
            else l=m+1;
        }
        return -1;
    }



// single element in sorted array
int singleNonDuplicate(vector<int>& nums) {
        int l=0;
        int h=nums.size()-1;
        while(l<h){
            int m=(l+h)/2;
            if((m%2==0 && nums[m]==nums[m+1]) || (m%2!=0 && nums[m]==nums[m-1])){
                l=m+1;
            }
            else{
                h=m;
            }
        }
        return nums[h];
    }




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


    



// trie 1
// insert
// search
// startsWith
// links[26] and flag
struct Node{
    Node* links[26];
    bool flag=false;
    bool containsKey(char ch){
        return (links[ch-'a']!=NULL);
    }
    void put(char ch,Node* node){
        links[ch-'a']=node;
    }
    Node* get(char ch){
        return links[ch-'a'];
    }
    void setEnd(){
        flag=true;
    }
    bool isEnd(){
        return flag;
    }
};
class Trie {
    Node* root;
public:
    Trie() {
        root=new Node();
    }
    
    void insert(string word) {
        Node* node=root;
        for(int i=0;i<word.length();i++){
            if(!node->containsKey(word[i])){
                node->put(word[i],new Node());
            }
            node=node->get(word[i]);
        }
        node->setEnd();
    }
    
    bool search(string word) {
        Node* node=root;
        for(int i=0;i<word.length();i++){
            if(!node->containsKey(word[i])){
                return false;
            }
            node=node->get(word[i]);
        }
         return node->isEnd();
    }
    
    bool startsWith(string prefix) {
        Node* node=root;
        for(int i=0;i<prefix.length();i++){
            if(!node->containsKey(prefix[i])){
                return false;
            }
            node=node->get(prefix[i]);
        }
        return true;
    }
};

// trie 2
// insert
// count words equal to
// count words starting with
// erase
// links[26],cw(count words) ,cp(count prefix)
struct Node {
  Node * links[26];
  int cntEndWith = 0;
  int cntPrefix = 0;

  bool containsKey(char ch) {
    return (links[ch - 'a'] != NULL);
  }
  Node * get(char ch) {
    return links[ch - 'a'];
  }
  void put(char ch, Node * node) {
    links[ch - 'a'] = node;
  }
  void increaseEnd() {
    cntEndWith++;
  }
  void increasePrefix() {
    cntPrefix++;
  }
  void deleteEnd() {
    cntEndWith--;
  }
  void reducePrefix() {
    cntPrefix--;
  }
  int getEnd() {
    return cntEndWith;
  }
  int getPrefix() {
    return cntPrefix;
  }
};
class Trie {
  private:
    Node * root;

  public:
    /** Initialize your data structure here. */
    Trie() {
      root = new Node();
    }

  /** Inserts a word into the trie. */
  void insert(string word) {
    Node * node = root;
    for (int i = 0; i < word.length(); i++) {
      if (!node -> containsKey(word[i])) {
        node -> put(word[i], new Node());
      }
      node = node -> get(word[i]);
      node -> increasePrefix();
    }
    node -> increaseEnd();
  }

 int countWordsEqualTo(string &word)
    {
        Node *node = root;
        for (int i = 0; i < word.length(); i++)
        {
            if (node->containsKey(word[i]))
            {
                node = node->get(word[i]);
            }
            else
            {
                return 0;
            }
        }
        return node->getEnd();
    }


  int countWordsStartingWith(string & word) {
    Node * node = root;
    for (int i = 0; i < word.length(); i++) {
      if (node -> containsKey(word[i])) {
        node = node -> get(word[i]);
      } else {
        return 0;
      }
    }
    return node -> getPrefix();
  }

  void erase(string & word) {
    Node * node = root;
    for (int i = 0; i < word.length(); i++) {
      if (node -> containsKey(word[i])) {
        node = node -> get(word[i]);
        node -> reducePrefix();
      } else {
        return;
      }
    }
    node -> deleteEnd();
  }
};

// Longest word with all prefix
// given vector of strings
// we have to tell longest string whose every prefix is present


// count distinct substrings
// given a string find number of distinct substrings


// two more problems



vector<long long> nextLargerElement(vector<long long> arr, int n){
        stack<long long > st;
        st.push(-1);
        vector<long long> ans;
        for(int i=n-1;i>=0;i--){
            while(st.top()!=-1 && st.top()<=arr[i]) st.pop();
            ans.push_back(st.top());
            st.push(arr[i]);
        }
        reverse(ans.begin(),ans.end());
        return ans;
    }



class MyQueue {
public:
    stack<int> s1;
    stack<int> s2;
    MyQueue() {
        
    }
    
    void push(int x) {
        s1.push(x);
    }
    
    int pop() {
        while(!s1.empty()){
            s2.push(s1.top());
            s1.pop();
        }
        int ans=s2.top();
        s2.pop();
        while(!s2.empty()){
            s1.push(s2.top());
            s2.pop();
        }
        return ans;
    }
    
    int peek() {
        while(!s1.empty()){
            s2.push(s1.top());
            s1.pop();
        }
        int ans=s2.top();
        while(!s2.empty()){
            s1.push(s2.top());
            s2.pop();
        }
        return ans;
    }
    
    bool empty() {
        return s1.empty();
    }
};


// pop the elements
// insert in sorted order
// means insert when top is smaller or empty
void sortedInsert(stack<int> &stack, int num) {
    //base case
    if(stack.empty() || (!stack.empty() && stack.top() < num) ) {
        stack.push(num);
        return;
    }
    
    int n = stack.top();
    stack.pop();
    
    //recusrive call
    sortedInsert(stack, num);
    
    stack.push(n);
}

void sortStack(stack<int> &stack)
{
        //base case
        if(stack.empty()) {
            return ;
        }
    
        int num = stack.top();
        stack.pop();
    
        //recursive call
        sortStack(stack);
    
        sortedInsert(stack, num);
}


// stack using arrays
#include<bits/stdc++.h>

using namespace std;
class Stack {
  int size;
  int * arr;
  int top;
  public:
    Stack() {
      top = -1;
      size = 1000;
      arr = new int[size];
    }
  void push(int x) {
    top++;
    arr[top] = x;
  }
  int pop() {
    int x = arr[top];
    top--;
    return x;
  }
  int Top() {
    return arr[top];
  }
  int Size() {
    return top + 1;
  }
};
int main() {

  Stack s;
  s.push(6);
  s.push(3);
  s.push(7);
  cout << "Top of stack is before deleting any element " << s.Top() << endl;
  cout << "Size of stack before deleting any element " << s.Size() << endl;
  cout << "The element deleted is " << s.pop() << endl;
  cout << "Size of stack after deleting an element " << s.Size() << endl;
  cout << "Top of stack after deleting an element " << s.Top() << endl;
  return 0;
}

// queue using arrays
#include<bits/stdc++.h>

using namespace std;
class Queue {
  int * arr;
  int start, end, currSize, maxSize;
  public:
    Queue() {
      arr = new int[16];
      start = -1;
      end = -1;
      currSize = 0;
    }

  Queue(int maxSize) {
    ( * this).maxSize = maxSize;
    arr = new int[maxSize];
    start = -1;
    end = -1;
    currSize = 0;
  }
  void push(int newElement) {
    if (currSize == maxSize) {
      cout << "Queue is full\nExiting..." << endl;
      exit(1);
    }
    if (end == -1) {
      start = 0;
      end = 0;
    } else
      end = (end + 1) % maxSize;
    arr[end] = newElement;
    cout << "The element pushed is " << newElement << endl;
    currSize++;
  }
  int pop() {
    if (start == -1) {
      cout << "Queue Empty\nExiting..." << endl;
    }
    int popped = arr[start];
    if (currSize == 1) {
      start = -1;
      end = -1;
    } else
      start = (start + 1) % maxSize;
    currSize--;
    return popped;
  }
  int top() {
    if (start == -1) {
      cout << "Queue is Empty" << endl;
      exit(1);
    }
    return arr[start];
  }
  int size() {
    return currSize;
  }

};

int main() {
  Queue q(6);
  q.push(4);
  q.push(14);
  q.push(24);
  q.push(34);
  cout << "The peek of the queue before deleting any element " << q.top() << endl;
  cout << "The size of the queue before deletion " << q.size() << endl;
  cout << "The first element to be deleted " << q.pop() << endl;
  cout << "The peek of the queue after deleting an element " << q.top() << endl;
  cout << "The size of the queue after deleting an element " << q.size() << endl;

  return 0;
}




// some questions of stack
// reverse a string using a stack
// stack has a property of reversing
string str="hello"
stack <char> st;
for(int i=0;i<str.lengnt();i++){
    st.push(str[i]);
}
string ans="";
while(!st.empty()){
    ans+=st.top();
    st.pop();
}
return ans; 

// delete middle element from stack
// jaate huye leke jana num
void solve(stack<int> &s,int n,int count){
        if(count==n/2){
            s.pop();
            return ;
        }
        int num=s.top();
        s.pop();
        solve(s,n,count+1);
        s.push(num);
    }
    void deleteMid(stack<int>&s, int sizeOfStack)
    {
        solve(s,sizeOfStack,0);
    }


 // similarly , insert a element at a bottom


 // reverse a stack using recursion
 // store the top element
// call the recursive fucntion for left over stack
// then put the top element at the bottom



// stack using queue
class MyStack {
public:
    // queue<int> q1;
    // queue<int> q2;
    queue<int> q;
    MyStack() {
        
    }
    
    void push(int x) {
        // q2.push(x);
        // while(!q1.empty()){
        //     q2.push(q1.front());
        //     q1.pop();
        // }
        // swap(q1,q2);
        q.push(x);
        for(int i=0;i<q.size()-1;i++){
            q.push(q.front());
            q.pop();
        }
    }
    
    int pop() {
        // int ans=q1.front();
        // q1.pop();
        // return ans;
        int ans=q.front();
        q.pop();
        return ans;
    }
    
    int top() {
        // return q1.front();
        return q.front();
    }
    
    bool empty() {
        // return q1.empty();
        return q.empty();
    }
};


bool isValid(string s) {
        stack<char> st;
        for(int i=0;i<s.length();i++)
        {
            if(s[i]=='(' || s[i]=='{' || s[i]=='[')
            {
                st.push(s[i]);
            }
            else if(st.empty()){
                return false;
            }
            if(s[i]==')')
            {
                if(st.top()!='(') return false;
                else st.pop();
            }
            if(s[i]=='}')
            {
                if(st.top()!='{') return false;
                else st.pop();
            }
            if(s[i]==']')
            {
                if(st.top()!='[') return false;
                else st.pop();
            }
        }
        return st.empty();

        
        
    }



// largest rectangle in histogram
vector<int> nextSmaller(vector<int> & heights,int n){
        stack<int> st;
        st.push(-1);
        vector<int> ans(n);
        for(int i=n-1;i>=0;i--){
            while(st.top()!=-1 && heights[st.top()]>=heights[i]) st.pop();
            ans[i]=st.top();
            st.push(i);
        }
        return ans;
    }
    vector<int> prevSmaller(vector<int> & heights,int n){
        stack<int> st;
        st.push(-1);
        vector<int> ans(n);
        for(int i=0;i<n;i++){
            while(st.top()!=-1 && heights[st.top()]>=heights[i]) st.pop();
            ans[i]=st.top();
            st.push(i);
        }
        return ans;
    }
    int largestRectangleArea(vector<int>& heights) {
        int n=heights.size();
        vector<int> next;
        next=nextSmaller(heights,n);
        vector<int> prev;
        prev=prevSmaller(heights,n);
        int ans=0;
        for(int i=0;i<n;i++){
            if(next[i]==-1) next[i]=n;
            ans=max(ans,heights[i]*(next[i]-prev[i]-1));
        }
        return ans;
    }



// doubly linked list and unordered map
class LRUCache {
public:
    class node{
        public:
        int key;
        int value;
        node* next;
        node* prev;
        node(int _key,int _value){
            key=_key;
            value=_value;
        }
    };
    node* head=new node(-1,-1);
    node* tail=new node(-1,-1);
    int cap;
    unordered_map<int,node*> mp;
    LRUCache(int capacity) {
        cap=capacity;
        head->next=tail;
        tail->prev=head;
    }
    void addnode(node* newnode){
        node* temp=head->next;
        head->next=newnode;
        newnode->next=temp;
        temp->prev=newnode;
        newnode->prev=head;
    }
    
    void deletenode(node* delnode){
        node* delprev=delnode->prev;
        node* delnext=delnode->next;
        delprev->next=delnext;
        delnext->prev=delprev;
    }
    
    int get(int key_) {
        if(mp.find(key_)!=mp.end()){
            node* resnode=mp[key_];
            int res=resnode->value;
            mp.erase(key_);
            deletenode(resnode);
            addnode(resnode);
            mp[key_]=head->next;
            return res;
        }
        return -1;
    }
    
    void put(int key_, int val) {
        if(mp.find(key_)!=mp.end()){
            deletenode(mp[key_]);
            mp.erase(key_);
        }
        if(mp.size()==cap){
            mp.erase(tail->prev->key);
            deletenode(tail->prev);
        }
        addnode(new node(key_,val));
        mp[key_]=head->next;
    }
};



// next smaller element
vector<int> help_classmate(vector<int> arr, int n) 
    { 
        vector<int> ans(n);
        stack<int> st;
        st.push(-1);
        for(int i=n-1;i>=0;i--){
            while(st.top()!=-1 && st.top()>=arr[i]) st.pop();
            ans[i]=st.top();
            st.push(arr[i]);
        }
        return ans;
    }



StockSpanner() {
        
    }
    int i=0;
    stack<pair<int,int>> st;
    int next(int price) {
        while(!st.empty() && st.top().first<=price) st.pop();
        int ans;
        if(st.empty()) ans=i+1;
        else ans=i-st.top().second;
        st.push({price,i});
        i++;
        return ans;
    }



int orangesRotting(vector<vector<int>>& grid) {
        int m=grid.size();
        int n=grid[0].size();
        int minutes=0,count=0,total=0;
        queue<pair<int,int>> q;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]!=0) total++;
                if(grid[i][j]==2) q.push({i,j});
            }
        }
        int dx[4] = {0, 0, 1, -1};
        int dy[4] = {1, -1, 0, 0};
        while(!q.empty()){
            int size=q.size();
            count+=size;
            while(size--){
                int x=q.front().first;
                int y=q.front().second;
                q.pop();
                for(int i=0;i<4;i++){
                    int nx=x+dx[i],ny=y+dy[i];
                    if(ny<0 || nx<0 || ny>=n || nx>=m || grid[nx][ny]!=1) continue;
                    grid[nx][ny]=2;
                    q.push({nx,ny});
                }
            }
            if(!q.empty()) minutes++;
        }
        return total==count?minutes:-1;
    }



    // sliding window maximum 
    // this approach -> tc= n , sc= k
    // storing in dq in decreasing order
    // storing index in dq
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ans;
        deque<int> dq;
        for(int i=0;i<nums.size();i++){
            if(!dq.empty() && dq.front()==i-k) dq.pop_front();// remove out of bound element
            while(!dq.empty() && nums[dq.back()]<=nums[i]) dq.pop_back();// removing all element less than i
            dq.push_back(i);
            if(i>=k-1) ans.push_back(nums[dq.front()]);// after we have checked for 1st subarray of size k
        }
        return ans;
    }



// the celibrity problem
// celibrity knows no one and everyone knows celibrity
// {{0 1 0},
//  {0 0 0}, 
//  {0 1 0}}
// Brute force -> n^2 -> for(0 to no of person) {for( for checking if all elements of row zero ) for( column should be one except diagonal )} 
// Optimal 
// put every person in stack
// while( stack size is not 1)
// A=st.top() then pop , B=st.top() then pop()
// if(A knows B) push B
// if(B knows A) push A
// the last element left is potential answer 
int celebrity(vector<vector<int> >& M, int n) 
    {
        stack<int> st;
        for(int i=0;i<n;i++) st.push(i);
        while(st.size()>1){
            int a=st.top();
            st.pop();
            int b=st.top();
            st.pop();
            if(M[a][b]){
                st.push(b);
            }
            else st.push(a);
        }
        int potentialCelibrity=st.top();
        int count0=0;
        int count1=0;
        for(int i=0;i<n;i++){
            if(M[potentialCelibrity][i]==0) count0++;
            if(M[i][potentialCelibrity]==1) count1++;
        }
        if(count0==n && count1==n-1) return potentialCelibrity;
        else return -1;
    }





// Maximum of minimum for every window size
// 10 20 30 50 10 70 30
// 50 ka effect kaha kaha hoga, kaha  kaha par woh minimum hoga (only one window size)
// find next smaller element in left and right
// 10->7 ,20->3 ,30->2 ,50->1 ,10->7, 70->1 30->2
// 10 answer ho sakta hai 7 aur usse chote window ka 
// we have to find maximum of every window size
// make a vector of window size (1 to n)
// and store in 7 th window size 10 (update if find greater)
// in 2 windwo size store 30
// after storing iterate from back and if back is i+1 is greater than i update
vector <int> maxOfMin(int arr[], int n)
    {
        vector<int> nextSmaller(n,n),prevSmaller(n,-1);
        stack<int> s,t;
        for(int i=0;i<n;i++){
            while(!s.empty() && arr[s.top()]>=arr[i]) s.pop();
            if(!s.empty()) prevSmaller[i]=s.top();
            s.push(i);
        }
        for(int i=n-1;i>=0;i--){
            while(!t.empty() && arr[t.top()]>=arr[i]) t.pop();
            if(!t.empty()) nextSmaller[i]=t.top();
            t.push(i);
        }
        vector<int> ans(n,0);
        for(int i=0;i<n;i++) ans[nextSmaller[i]-prevSmaller[i]-2]=max(ans[nextSmaller[i]-prevSmaller[i]-2],arr[i]);
        for(int i=n-2;i>=0;i--){
            if(ans[i+1]>ans[i]) ans[i]=ans[i+1];
        }    
        return ans;
    }




// integet to roman
// for every string , subtract from number uptill it is possible
string intToRoman(int num) {
        vector<pair<int,string>> v;
        v.push_back({1000,"M"});
        v.push_back({900,"CM"});
        v.push_back({500,"D"});
        v.push_back({400,"CD"});
        v.push_back({100,"C"});
        v.push_back({90,"XC"});
        v.push_back({50,"L"});
        v.push_back({40,"XL"});
        v.push_back({10,"X"});
        v.push_back({9,"IX"});
        v.push_back({5,"V"});
        v.push_back({4,"IV"});
        v.push_back({1,"I"});
        string ans="";
        
        for(int i=0;i<v.size();i++){
            while(num>=v[i].first){
                num-=v[i].first;
                ans+=v[i].second;
            }
        }
        return ans;
    }




string longestCommonPrefix(vector<string>& strs) {
        string ans="";
        for(int i=0;i<strs[0].size();i++){
            bool f=1;
            for(int j=1;j<strs.size();j++){
                if(i>=strs[j].size() || strs[0][i]!=strs[j][i]){
                    f=0;
                    break;
                }
            }
            if(f) ans+=strs[0][i];
            else break;
        }
        return ans;
    }



// longest palindromic substring
// for every element check if it making , a odd palindrome or a even palindrome
string longestPalindrome(string s) {
        int n=s.length();
        int MaxLen=0;
        string ans="";
        for(int i=0;i<s.length();i++){
            int j=i-1;
            int k=i+1;
            int tempLen=1;
            while(j>=0 && k<n){
                if(s[j]==s[k]){
                    tempLen+=2;
                }
                else break;
                j--;
                k++;
            }
            if(MaxLen<tempLen){
                MaxLen=tempLen;
                ans=s.substr(j+1,k-j-1);
            }
            if(i<n-1){
                tempLen=0;
                j=i;
                k=i+1;
                while(j>=0 && k<n){
                    if(s[j]==s[k]){
                        tempLen+=2;
                    }
                    else break;
                    j--;
                    k++;
                }
                if(MaxLen<tempLen){
                    MaxLen=tempLen;
                    ans=s.substr(j+1,k-j-1);
                }
            }
        }
        return ans;
    }



class Solution {
private:
    int BASE = 1000000;
public:
    int repeatedStringMatch(string A, string B) {
        if(A == B) return 1;
        int count = 1;
        string source = A;
        while(source.size() < B.size()){
            count++;
            source+=A;
        }
        if(source == B) return count;
        if(Rabin_Karp(source,B) != -1) return count;
        if(Rabin_Karp(source+A,B) != -1) return count+1;
        return -1;
    }
    int Rabin_Karp(string source, string target){
        if(source == "" or target == "") return -1;
        int m = target.size();
        int power = 1;
        for(int i = 0;i<m;i++){
            power = (power*31)%BASE;
        }
        int targetCode = 0;
        for(int i = 0;i<m;i++){
            targetCode = (targetCode*31+target[i])%BASE;
        }
        int hashCode = 0;
        for(int i = 0;i<source.size();i++){
            hashCode = (hashCode*31 + source[i])%BASE;
            if(i<m-1) continue;
            if(i>=m){
                hashCode = (hashCode-source[i-m]*power)%BASE;
            }
            if(hashCode<0)
                hashCode+=BASE;
            if(hashCode == targetCode){
                if(source.substr(i-m+1,m) == target)
                    return i-m+1;
            }
        }
        return -1;
    }
};



// reverse words in string
// skip the front spaces first
// make other pointer j , make this point to ending of the word
string reverseWords(string s) {
        string result;
        int i = 0;
        int n = s.length();

        while(i < n){
            while(i < n && s[i] == ' ') i++;
            if(i >= n) break;
            int j = i + 1;
            while(j < n && s[j] != ' ') j++;
            string sub = s.substr(i, j-i);
            if(result.length() == 0) result = sub;
            else result = sub + " " + result;
            i = j+1;
        }
        return result;
    }



// start from back
// if mp[s[i]]<mp[s[i+1]] then subtract mp[s[i]]
// else add
int romanToInt(string s) {
        map<char,int> mp={{'M',1000},{'D',500},{'C',100},{'L',50},{'X',10},{'V',5},{'I',1}};
        int ans=mp[s[s.length()-1]];
        for(int i=s.length()-2;i>=0;i--){
            if(mp[s[i]]<mp[s[i+1]]) ans-=mp[s[i]];
            else ans+=mp[s[i]];
        }
        return ans;
    }



// generate number between the dots
    int compareVersion(string version1, string version2) {
        int n1=version1.length();
        int n2=version2.length();
        int i=0;
        int j=0;
        while(i<n1  || j<n2){
            int number1=0,number2=0;
            while(i<n1 && version1[i]!='.'){
                number1=number1*10+(version1[i]-'0');
                i++;
            }
            while(j<n2 && version2[j]!='.'){
                number2=number2*10+(version2[j]-'0');
                j++;
            }
            if(number1>number2) return 1;
            if(number2>number1) return -1;
            i++;
            j++;
        }
        return 0;
    }




// count and say
// for n=1 answer is 1
    // for n=2 answer is 11
    // for n=3 answer is 21
    // for n=4 answer is 1211
    string countAndSay(int n) {
        // recursive
        // if(n==1) return "1";
        // string s=countAndSay(n-1);
        // int count=1;
        // string res="";
        // for(int i=1;i<s.length();i++){
        //     if(s[i]==s[i-1]){
        //         count++;
        //     }
        //     else{
        //         res=res+to_string(count)+s[i-1];
        //         count=1;
        //     }
        // }
        // res+=to_string(count)+s[s.length()-1];
        // return res;
        string ans="1";
        for(int i=1;i<n;i++){
            string res="";
            int count=1;
            for(int i=1;i<ans.length();i++){
                if(ans[i]==ans[i-1]){
                    count++;
                }
                else{
                    res=res+to_string(count)+ans[i-1];
                    count=1;
                }
            }
            res+=to_string(count)+ans[ans.length()-1];
            ans=res;
        }
        return ans;
    }



// KMP algo




// minimum insertions to make palindrom
// length-longest palindromic subsequence
int solve(int i,int j,string &text1,string &text2,vector<vector<int>> &dp){
        if(i==-1 || j==-1 ) return 0;
        if(dp[i][j]!=-1) return dp[i][j];
        if(text1[i]==text2[j]) return dp[i][j]=1+solve(i-1,j-1,text1,text2,dp);
        else 
        return dp[i][j]=max(solve(i-1,j,text1,text2,dp),solve(i,j-1,text1,text2,dp));
    }
    int longestPalindromeSubseq(string s) {
        string text1=s;
        reverse(s.begin(),s.end());
        string text2=s;
        vector<vector<int>> dp(text1.length(),vector<int>(text2.length(),-1));
        return solve(text1.length()-1,text2.length()-1,text1,text2,dp);
    }
    int minInsertions(string s) {
        return s.length()-longestPalindromeSubseq(s);
    }



// char array
// increment for one and decrement for the other
bool isAnagram(string s, string t) {
        if(s.length()!=t.length()) return false;
        vector<int> v(26,0);
        for(int i=0;i<s.length();i++) v[s[i]-'a']++;
        for(int i=0;i<t.length();i++) v[t[i]-'a']--;
        for(int i=0;i<26;i++) if(v[i]!=0) return false;
        return true;
    }




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



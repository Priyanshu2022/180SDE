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
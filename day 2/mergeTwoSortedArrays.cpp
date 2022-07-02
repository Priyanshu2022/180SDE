// merge two sorted arrays
// GAP method
//Initially take the gap as (m+n)/2;
// Take as a pointer1 = 0 and pointer2 = gap.
// Run a oop from pointer1 &  pointer2 to  m+n and whenever arr[pointer2]<arr[pointer1], just swap those.
// After completion of the loop reduce the gap as gap=gap/2.
// Repeat the process until gap>0.
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i1=m-1;
        int i2=n-1;
        int j=m+n-1;
        while(i1>=0 && i2>=0){
            if(nums1[i1]>nums2[i2]){
                nums1[j]=nums1[i1];
                i1--;
            }
            else{
                nums1[j]=nums2[i2];
                i2--;
            }
            j--;
        }
        while(i2>=0){
            nums1[j--]=nums2[i2--];
        }
    }
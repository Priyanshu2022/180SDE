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
// a b c
// 0 0 0 represent no char taken
// 0 0 1 represent a is taken
// for(num=0 to 2^n-1 (i.e (1<<n)-1)){
// 	sub=""
// 	for(i=0 to n-1){
// 		if(num&(i<<1)){
// 			sub+=s[i];
// 		}
// 	}
// 	ans push_back or print
// }


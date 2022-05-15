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
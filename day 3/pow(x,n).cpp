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
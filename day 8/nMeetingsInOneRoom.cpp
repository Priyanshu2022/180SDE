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
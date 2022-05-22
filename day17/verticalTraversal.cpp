// vertical traversal
// map will have hd and map of level and node's value
// queue will store node and pair of hd and level
vector<int> verticalOrder(Node *root)
    {
        // hd       level  node's value 
        map<int,map<int,vector<int>>> mp;
        vector<int> ans;
        if(root==NULL) return ans;
        
        //        node        hd  level
        queue<pair<Node*,pair<int,int>>> q;
        q.push({root,{0,0}});
        while(!q.empty()){
            auto temp=q.front();
            q.pop();
            int hd=temp.second.first;
            int lvl=temp.second.second;
            mp[hd][lvl].push_back(temp.first->data);
            if(temp.first->left) q.push({temp.first->left,{hd-1,lvl+1}});
            if(temp.first->right) q.push({temp.first->right,{hd+1,lvl+1}});
        }
        for(auto i:mp){
            for(auto j:i.second){
                for(auto cur:j.second){
                    ans.push_back(cur);
                }
            }
        }
        return ans;
    }
// run dfs, maintain a map
    Node* dfs(Node* node,unordered_map<Node*,Node*> &mp){
        vector<Node*> neighbor;
        Node* clone=new Node(node->val);
        mp[node]=clone;
        for(auto it:node->neighbors){
            if(mp.find(it)==mp.end()){
                neighbor.push_back(dfs(it,mp));
            }
            else{
                neighbor.push_back(mp[it]);
            }
        }
        clone->neighbors=neighbor;
        return clone;
    }
    Node* cloneGraph(Node* node) {
        unordered_map<Node*,Node*> mp;
        if(node==NULL) return NULL;
        if(node->neighbors.size()==0){
            Node* clone=new Node(node->val);
            return clone;
        }
        return dfs(node,mp);
    }
// floyd warshall
// all pair shortest path
// mark diagonal elements as zero
// rest all as int max
// now n^3 loop (1st one for pivot , i.e if we can go through this pivot with less distance)
void shortest_distance(vector<vector<int>>&matrix){
	    int n=matrix.size();
	    for(int i=0;i<n;i++){
	        for(int j=0;j<n;j++){
	            if(i==j) matrix[i][j]=0;
	            else if(matrix[i][j]==-1) matrix[i][j]=INT_MAX;
	        }
	    }
	    for(int k=0;k<n;k++){
	        for(int i=0;i<n;i++){
	            for(int j=0;j<n;j++){
	                if(matrix[i][k]!=INT_MAX && matrix[k][j]!=INT_MAX && (matrix[i][k]+matrix[k][j]<matrix[i][j])){
	                    matrix[i][j]=matrix[i][k]+matrix[k][j];
	                }
	            }
	        }
	    }
	    for(int i=0;i<n;i++){
	        for(int j=0;j<n;j++){
	            if(matrix[i][j]==INT_MAX) matrix[i][j]=-1;
	        }
	    }
	}
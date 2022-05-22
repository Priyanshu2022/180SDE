// if root is null return false
// push in path
// then match with x , if matched return true, else check if left or right return true, if yes return true
// other wise pop back and return false
bool getPath(node * root, vector < int > & arr, int x) {
  if (!root) return false;
  arr.push_back(root -> data);
  if (root -> data == x) return true;
  if (getPath(root -> left, arr, x) || getPath(root -> right, arr, x)) return true;  
  arr.pop_back();
  return false;
}
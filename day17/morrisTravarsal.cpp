// morris traversal
vector < int > inorderTraversal(node * root) {
  vector < int > inorder;
  node * cur = root;
  while (cur != NULL) {
    if (cur -> left == NULL) { // if curr's left is null, no left therefore root will be printed and cur will move right
      inorder.push_back(cur -> data);
      cur = cur -> right;
    } else { // if there exist a left
      node * prev = cur -> left;
      while (prev->right!=NULL && prev -> right != cur) { // find last guy in the left subtree , it should not point to cur
        prev = prev -> right;
      }

      if (prev -> right == NULL) { // link not made
        prev -> right = cur; // make link to cur
        cur = cur -> left; // move cur to left
      } else {
        prev -> right = NULL; // if already link present (prev->right ==cur), make it point to null
        inorder.push_back(cur -> data); // push cur as , left already visited
        cur = cur -> right; // move to right
      }
    }
  }
  return inorder;
}

// for preorder , instead of pushing after right , push while marking link
vector < int > preorderTraversal(node * root) {
  vector < int > inorder;
  node * cur = root;
  while (cur != NULL) {
    if (cur -> left == NULL) {
      inorder.push_back(cur -> data);
      cur = cur -> right;
    } else {
      node * prev = cur -> left;
      while (prev->right!=NULL && prev -> right != cur) {
        prev = prev -> right;
      }

      if (prev -> right == NULL) { 
        prev -> right = cur;
        inorder.push_back(cur -> data);
        cur = cur -> left;
      } else {
        prev -> right = NULL;
        cur = cur -> right;
      }
    }
  }
  return inorder;
}
import torch
def get_kernel_mat(a, b, r=1, p=2):
  # input: a and b vectors of shape [N]
  # returns kernel matrix K of shape [N, N]
  # kernel operation is: K_{ij} = (dot(x_i, x_j) + r)^p where r=1 and p={2, 4}

    res = torch.outer(a.T, b)
    return (res+r)**p

def get_normalized_kernel_mat(y):
  # input: kernel matrix of shape [N, N]
  # returns normalized kernel matrix of the same shape
  rows, cols = y.shape
  d = torch.sqrt(torch.diag(y))
  r = d.repeat(rows, 1)
  c = d.repeat(cols, 1).T


def get_mat_renyi_entropy(g, alpha):
    # returns the matrix based renyi entropy of a normalized kernel matrix g
  # according to eqn 2
  # g is of shape [N, N]
  # returns scalar
  print(f'inside get_mat_renyi_entropy: {type(g)} and g: {g}')
  res = torch.log2(torch.trace(g**alpha))
  return res/(1-alpha)

  return ((y/c)/r)/rows

def get_joint_entropy(g1, g2, alpha):
  # computes joint entropy according to eqn 3
  # g1 and g2 are of shape [N, N]
  # returns scalar
  val = torch.multiply(g1, g2)
  return get_mat_renyi_entropy(val/torch.trace(val), alpha)

def get_mutual_info(g1, g2, alpha):
  # computes mutual information according to eqn 4
  # g1 and g2 are two normalized kernel matrices of shape [N, N]
  # returns scalar
  return get_mat_renyi_entropy(g1, alpha) + get_mat_renyi_entropy(g2, alpha) - get_joint_entropy(g1, g2, alpha)

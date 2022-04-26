import torch
def get_kernel_mat(a, b, r=1, p=2):
  # input: a and b vectors of shape [N]
  # returns kernel matrix K of shape [N, N]
  # kernel operation is: K_{ij} = (dot(x_i, x_j) + r)^p where r=1 and p={2, 4}
  if len(a.shape)==1: # for vectors, compute an outer product
    res = torch.outer(a.T, b)
  else:
    res = torch.matmul(a,  a.T)
  return (res+r)**p

def get_normalized_kernel_mat(y):
  # input: kernel matrix of shape [N, N]
  # returns normalized kernel matrix of the same shape
  rows, cols = y.shape
  d = torch.sqrt(torch.diag(y))
  r = d.repeat(rows, 1)
  c = d.repeat(cols, 1).T

  return ((y/c)/r)/rows


def get_mat_renyi_entropy(g, alpha):
    # returns the matrix based renyi entropy of a normalized kernel matrix g
  # according to eqn 2
  # g is of shape [N, N]
  # returns scalar
  # print(f'inside get_mat_renyi_entropy: {type(g)} and g: {g}')
  res = torch.log2(torch.trace(g**alpha))
  return res/(1-alpha)

  return ((y/c)/r)/rows

def get_joint_entropy(g1, g2, alpha):
  # computes joint entropy according to eqn 3
  # g1 and g2 are of shape [N, N]
  # returns scalar
  val = torch.multiply(g1, g2)
  return get_mat_renyi_entropy(val/torch.trace(val), alpha)

def get_mutual_info(m1, m2, alpha):
  # computes mutual information according to eqn 4
  # returns scalar


  # g1 and g2 are two normalized kernel matrices of shape [N, N]

  g1 = get_kernel_mat(m1, m1)
  g2 = get_kernel_mat(m2, m2)
  # normalized_g1 and normalized_g2 are two normalized kernel matrices of shape [N, N]
  normalized_g1 = get_normalized_kernel_mat(g1)
  normalized_g2 = get_normalized_kernel_mat(g2)
  return get_mat_renyi_entropy(normalized_g1, alpha) + get_mat_renyi_entropy(normalized_g2, alpha) - get_joint_entropy(normalized_g1, normalized_g2, alpha)

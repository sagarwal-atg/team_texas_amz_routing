import cvxpy as cvx
import numpy as np
import timeit
from cvxpylayers.torch import CvxpyLayer
import torch


def subtour(B):
    """
    helper function: return subtour from a boolean matrix B
    """
    node = 0
    subt = [node]
    while True:
        for j in range(6):
            #print (B[subt[-1], j])
            if B[subt[-1],j] > 0.99:
                if j not in subt:
                    subt.append(j)
                else:
                    return subt
                
                
"""
Approach 1: MTZ subtour elimination constraint
"""
# distance matrix from one city to another
a = np.matrix([[1000, 60, 79, 37, 10, 61], 
               [60, 1000, 22, 48, 63, 54],
               [79, 22, 1000, 49, 70, 38],
               [37, 48, 49, 1000, 38, 45],
               [10, 63, 70, 38, 1000, 53],
               [61, 54, 38, 45, 53, 1000]]) 
a = torch.from_numpy(a)

A = cvx.Parameter((6, 6))

# boolean matrix, indicating the trip
B = cvx.Variable((6,6))

# exemplary matrix
C = np.array([1,1,1,1,1,1])

# auxiliary var
u = cvx.Variable((6, 1))

# objective
obj = cvx.Minimize(sum([A[i,:] @ B[:,i] for i in range(6)]))


# basic condition
constraints = []
# constraints = [(cvx.sum(B, axis=0) == C)]
# constraints.append((cvx.sum(B, axis=1) == C.transpose()))

for i in range(6):
    constraints.append((cvx.sum(B[i, :]) == 1 ))
    constraints.append((cvx.sum(B[:, i]) == 1 ))
    constraints.append((cvx.sum(B[i, i]) == 0 ))
    for j in range(6):
        constraints.append((0 <= B[i][j]))
        constraints.append((B[i][j] <= 1))

# subtour elimination
for i in range(1,6):
    for j in range(1,6):
        if i != j:
            constraints.append(u[i] - u[j] + 6*B[i,j] <= 5)
            
# # condition for u
for i in range(6):
    constraints.append(u[i] >= 0)
    
prob = cvx.Problem(obj, constraints)

assert prob.is_dpp()

cvxpylayer = CvxpyLayer(prob, parameters=[A], variables=[B])

# solve the problem
sol_x = cvxpylayer(a)
print(sol_x)

# Time performance:

# opt = prob.solve()

# Print results
# print ("Minimal time: ", opt)
# print ("Optimal tour: ", subtour(B.value))
# print ("Converge time: ", timeit.default_timer() - st)

"""
# Approach 2: Lazy subtour elimination
# """
# # distance matrix from one city to another
# # A = np.matrix([[1000, 60, 79, 37, 10, 61], 
# #                [60, 1000, 22, 48, 63, 54],
# #                [79, 22, 1000, 49, 70, 38],
# #                [37, 48, 49, 1000, 38, 45],
# #                [10, 63, 70, 38, 1000, 53],
# #                [61, 54, 38, 45, 53, 1000]])

# # boolean matrix, indicating the trip
# B = cvx.Bool(6,6)

# # exemplary matrix
# C = np.matrix('1,1,1,1,1,1')

# # objective
# obj = cvx.Minimize(sum([A[i,:]*B[:,i] for i in range(6)]))

# # basic condition
# constraints = [(cvx.sum_entries(B, axis=0) == C), (cvx.sum_entries(B, axis=1) == C.transpose())]

# # preliminary solution, which might involve subtours
# prob = cvx.Problem(obj, constraints)
# st = timeit.default_timer()
# opt = prob.solve()

# while True:
#     subt = subtour(B.value)
#     if len(subt) == 6:
#         print ("Minimal time: ", opt)
#         print ("Optimal tour: ", subt)
#         print ("Converge time: ", timeit.default_timer() - st)
#         break
#     else:
#         print ("Try: ", subt)
#         nots = [j for j in range(6) if j not in subt]
#         constraints.append(sum(B[i,j] for i in subt for j in nots) >= 1)
#         prob = cvx.Problem(obj, constraints)
#         opt = prob.solve()
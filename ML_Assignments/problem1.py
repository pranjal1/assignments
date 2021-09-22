import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

y = np.zeros(500)
x = np.zeros((2,500))

mean1 = [1, 1] #change this line
cov1 = [[2, 1], [1, 2]]  #change this line

mean2 = [-1, -1]  #change this line
cov2 = [[2, 1], [1, 2]]  #change this line

p1 = 0.5  #change this line
p2 = 0.5  #change this line

x_0min = -5
x_0max = 5

x_1min = -5
x_1max = 5

Label1 = [[],[]]
rv1 = multivariate_normal(mean1, cov1)
rv2 = multivariate_normal(mean2, cov2)
Label2 = [[],[]]
x0 = np.linspace(x_0min,x_0max,500)
x1 = np.linspace(x_1min,x_1max,500)

for i in x0:
  for j in x1:
    if rv1.pdf([i,j])*p1 > rv2.pdf([i,j])*p2:
      Label1[0].append(i)
      Label1[1].append(j)
    else:
      Label2[0].append(i)
      Label2[1].append(j)

plt.plot(Label1[0], Label1[1], 'rx')
plt.plot(Label2[0], Label2[1], 'o')
plt.tight_layout()

plt.show()
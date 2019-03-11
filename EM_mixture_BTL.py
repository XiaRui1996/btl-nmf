import numpy as np
import pandas as pd
import math

M = 14
N = 20
K = 2

b_m = np.zeros((M, N, N))
xls = pd.ExcelFile("C:/Users/admin/Desktop/FYP/Finals_male.xlsx")

tournaments ={ 'Australian Open':0,
               'Roland Garros':1,
               'Wimbledon':2,
               'US Open':3,
               'Indian Wells Master':4,
               'Madrid Open':5,
               'Miami Open':6,
               'Monte-Carlo Masters':7,
               'Paris Masters':8,
               'Italian Open':9,
               'Canada Masters':10,
               'Cincinnati Masters':11,
               'Shanghai Masters':12,
               'ATP Finals':13}

for tournament in list(tournaments.keys()):
    print(tournament)
    df = pd.read_excel(xls, tournament)
    for i in range(N):
        for j in range(N):
            value = df.iloc[i][j+1]
            b_m[tournaments[tournament]][i][j] = int(value)

b = np.zeros((N,N))
E = 0
for i in range(N):
    for j in range(N):
        b[i][j] = np.sum(b_m[:,i,j])
        if b[i][j] != 0:
            E += b[i][j]

final_l=[]
final_p=[]
final_a=[]
x = 1
while x <= 100:
    P = np.array([0.5,0.5])
    A = np.random.uniform(low=1, high=10.0, size = (K,N))

    normalize = np.sum(A)
    for i in range(N):
        for k in range(K):
            A[k][i] /= normalize

    loglikelihood = 0
    for i in range(N):
        for j in range(N):
            if b[i][j]!=0:
                term = 0
                for k in range(K):
                    term += P[k] * (A[k][i]/(A[k][i]+A[k][j]))
                loglikelihood += b[i][j] * math.log(term)
    new = loglikelihood
    old = 0

    skip = 0
    while True:
        new_P = np.array([0.0,0.0])
        new_A = np.zeros((K,N))


        # E-step, evaluate the posterior assignment probabilities
        ass = np.zeros((K,N,N))

        for i in range(N):
            for j in range(N):
                if b[i][j]!=0:
                    for k in range(K):
                        ass[k][i][j] = P[k] * (A[k][i]/(A[k][i]+A[k][j]))

        for i in range(N):
            for j in  range(N):
                norm = np.sum(ass[:,i,j])
                if norm!=0:
                    for k in range(K):
                        ass[k][i][j]/=norm

        # M-step, update the parameters

        for k in range(K):
            hello = 0
            for i in range(N):
                for j in range(N):
                    if b[i][j]!=0:
                        hello += ass[k][i][j]*b[i][j]
            new_P[k] = hello/E

        for k in range(K):
            for i in range(N):

                a = 0
                c = 0
                for j in range(N):
                    if b[i][j]!=0:
                        a += b[i][j] * ass[k][i][j]
                    if b[i][j]!=0 or b[j][i]!=0:
                        if A[k][i] + A[k][j] <1e-300:
                            print(A[k][i] + A[k][j], A[k][i], A[k][j],k, i,j)
                            skip= 1
                            break
                        c += (b[i][j] * ass[k][i][j] + b[j][i] * ass[k][j][i])/(A[k][i] + A[k][j])
                if skip == 1:
                    break
                new_A[k][i] = a/c
            if skip == 1:
                break
        if skip == 1:
            print("skip")
            break

        
        normalize = np.sum(new_A)
        for i in range(N):
            for k in range(K):
                new_A[k][i] /= normalize

        difference = max(np.amax(np.absolute(new_A-A)), np.amax(np.absolute(new_P-P)))
        P = new_P
        A = new_A

        loglikelihood = 0
        for i in range(N):
            for j in range(N):
                if b[i][j]!=0:
                    term = 0
                    for k in range(K):
                        term += P[k] * (A[k][i]/(A[k][i]+A[k][j]))
                    loglikelihood += b[i][j] * math.log(term)
        old = new
        new = loglikelihood
        if new < old:
            print("Warning: loglikelihood decreased!")
            break
        
        if difference < 1e-6:
            break
        
    if skip == 1:
        continue
    x += 1
    print(x, ' ', new)
    final_l.append(new)
    final_p.append(P)
    final_a.append(A)

best= np.argmax(final_l)
P = final_p[best]
A = final_a[best]
        

            

    

    


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

M = 14
N = 20
K = 2
epsilon = 1e-300
E = np.full((K,N), epsilon)

final_a = []
final_acc = {}

years = ['08','09','10','11','12','13','14','15','16','17']

for year in years:
    m = np.zeros((M, N, N))
    xls = pd.ExcelFile("C:/Users/admin/Desktop/FYP/Matches_male.xlsx")

    tournaments ={ 'Australian':0,
               'Roland':1,
               'USOpen':2,
               'Monte':3,
               'Wimbledon':4,
               'Paris':5,
               'Madrid':6,
               'Rome':7,
               'Indian':8,
               'Miami':9,
               'Canada':10,
               'Cincinnati':11,
               'Shanghai':12,
               'Finals':13}
    
    names = { 'Rafael Nadal':0,
          'Novak Djokovic':1,
          'David Ferrer':2,
          'Tomas Berdych':3,
          'Roger Federer':4,
          'Andy Murray':5,
          'Fernando Verdasco':6,
          'Philipp Kohlschreiber':7,
          'Richard Gasquet':8,
          'Gilles Simon':9,
          'Stan Wawrinka':10,
          'Jo-Wilfried Tsonga':11,
          'Marin Cilic':12,
          'Feliciano Lopez':13,
          'John Isner':14,
          'Nicolas Almagro':15,
          'Juan Martin del Potro':16,
          'Gael Monfils':17,
          'Milos Raonic':18,
          'Kei Nishikori':19 }

    for defender in list(names.keys()):
        if defender == 'Gael Monfils':
            continue
        df = pd.read_excel(xls, defender)
        num_column = len(df.columns)
        
        for player in list(df.index):
            for j in range(num_column):

                if type(df.loc[player][j]) == str:
                    result = df.loc[player][j]

                    if result[-2:] != year:

                        if 'Australian' in result:
                            if 'Australian0' in result:
                                m[0][names[player]][names[defender]] = int(m[0][names[player]][names[defender]])+1
                            else:
                                m[0][names[defender]][names[player]] = int(m[0][names[defender]][names[player]])+1
                        elif 'Roland' in result:
                            if 'Roland0' in result:
                                m[1][names[player]][names[defender]] = int(m[1][names[player]][names[defender]])+1
                            else:
                                m[1][names[defender]][names[player]] = int(m[1][names[defender]][names[player]])+1
                        elif 'USOpen' in result:
                            if 'USOpen0' in result:
                                m[2][names[player]][names[defender]] = int(m[2][names[player]][names[defender]])+1
                            else:
                                m[2][names[defender]][names[player]] = int(m[2][names[defender]][names[player]])+1
                        elif 'Monte' in result:
                            if 'Monte0' in result:
                                m[3][names[player]][names[defender]] = int(m[3][names[player]][names[defender]])+1
                            else:
                                m[3][names[defender]][names[player]] = int(m[3][names[defender]][names[player]])+1
                        elif 'Wimbledon' in result:
                            if 'Wimbledon0' in result:
                                m[4][names[player]][names[defender]] = int(m[4][names[player]][names[defender]])+1
                            else:
                                m[4][names[defender]][names[player]] = int(m[4][names[defender]][names[player]])+1
                        elif 'Paris' in result:
                            if 'Paris0' in result:
                                m[5][names[player]][names[defender]] = int(m[5][names[player]][names[defender]])+1
                            else:
                                m[5][names[defender]][names[player]] = int(m[5][names[defender]][names[player]])+1
                        elif 'Madrid' in result:
                            if 'Madrid0' in result:
                                m[6][names[player]][names[defender]] = int(m[6][names[player]][names[defender]])+1
                            else:
                                m[6][names[defender]][names[player]] = int(m[6][names[defender]][names[player]])+1
                        elif 'Rome' in result:
                            if 'Rome0' in result:
                                m[7][names[player]][names[defender]] = int(m[7][names[player]][names[defender]])+1
                            else:
                                m[7][names[defender]][names[player]] = int(m[7][names[defender]][names[player]])+1
                        elif 'Indian' in result:
                            if 'Indian0' in result:
                                m[8][names[player]][names[defender]] = int(m[8][names[player]][names[defender]])+1
                            else:
                                m[8][names[defender]][names[player]] = int(m[8][names[defender]][names[player]])+1
                        elif 'Miami' in result:
                            if 'Miami0' in result:
                                m[9][names[player]][names[defender]] = int(m[9][names[player]][names[defender]])+1
                            else:
                                m[9][names[defender]][names[player]] = int(m[9][names[defender]][names[player]])+1
                        elif 'Canada' in result:
                            if 'Canada0' in result:
                                m[10][names[player]][names[defender]] = int(m[10][names[player]][names[defender]])+1
                            else:
                                m[10][names[defender]][names[player]] = int(m[10][names[defender]][names[player]])+1
                        elif 'Cincinnati' in result:
                            if 'Cincinnati0' in result:
                                m[11][names[player]][names[defender]] = int(m[11][names[player]][names[defender]])+1
                            else:
                                m[11][names[defender]][names[player]] = int(m[11][names[defender]][names[player]])+1
                        elif 'Shanghai' in result:
                            if 'Shanghai0' in result:
                                m[12][names[player]][names[defender]] = int(m[12][names[player]][names[defender]])+1
                            else:
                                m[12][names[defender]][names[player]] = int(m[12][names[defender]][names[player]])+1
                        elif 'Finals' in result:
                            if 'Finals0' in result:
                                m[13][names[player]][names[defender]] = int(m[13][names[player]][names[defender]])+1
                            else:
                                m[13][names[defender]][names[player]] = int(m[13][names[defender]][names[player]])+1
    b = m                         
    final_l = []
    final_w = []
    final_h = []



    x = 1
    while x <= 10:
        W = np.random.uniform(low=1, high=10.0, size = (M,K))
        H = np.random.uniform(low=1, high=10.0, size = (K,N))

    ##    #row(W) = 1, all(H) = 1
    ##    for m in range(M):
    ##        normalizeW = np.sum(W[m,:])
    ##        for k in range(K):
    ##            W[m][k] /= normalizeW
    ##
    ##    normalizeH = np.sum(H)
    ##    alpha = (normalizeH + K*N*epsilon)/(1+K*N*epsilon)
    ##    for k in range(K):
    ##        for i in range(N):
    ##            H[k][i] = (H[k][i] + (1-alpha)*epsilon)/alpha
        #col(W) = 1, all(H) = 1
        for k in range(K):
            normalizeW = np.sum(W[:,k])
            for m in range(M):
                W[m][k] /= normalizeW
            for i in range(N):
                H[k][i] = H[k][i]*normalizeW + epsilon * (normalizeW-1)

        normalizeH = np.sum(H)
        alpha = (normalizeH + K*N*epsilon)/(1+K*N*epsilon)
        for k in range(K):
            for i in range(N):
                H[k][i] = (H[k][i] + (1-alpha)*epsilon)/alpha

        H_e = H + E
        loglikelihood = 0
        for m in range(M):
            for i in range(N):
                for j in range(N):
                    if b[m][i][j]!=0:
                        loglikelihood += b[m][i][j] * (-math.log(np.dot(W[m,:],H_e[:,i])) + math.log(np.dot(W[m,:],H_e[:,i])+np.dot(W[m,:],H_e[:,j])))
        new = loglikelihood
        old = 0


        while True:
            new_W = np.zeros((M,K))
            new_H = np.zeros((K,N))
            
            for m in range(M):
                for k in range(K):

                    a_mk = 0
                    c_mk = 0
                    for i in range(N):
                        for j in range(N):
                            if b[m][i][j] != 0:
                                a_mk += b[m][i][j] * W[m][k] * (H[k][i] + epsilon) / np.dot(W[m,:],H_e[:,i])
                                c_mk += b[m][i][j] * (H[k][i] + H[k][j] + 2*epsilon) / (np.dot(W[m,:],H_e[:,i]) + np.dot(W[m,:],H_e[:,j]))

                    new_W[m][k] = a_mk/c_mk


            for k in range(K):
                for i in range(N):

                    a_ki = 0
                    c_ki = 0
                    for m in range(M):
                        for j in range(N):
                            if j != i:
                                a_ki += b[m][i][j] * new_W[m][k] * (H[k][i] + epsilon) / np.dot(new_W[m,:],H_e[:,i])
                                c_ki += (b[m][i][j]+b[m][j][i]) * new_W[m][k] / (np.dot(new_W[m,:],H_e[:,i]) + np.dot(new_W[m,:],H_e[:,j]))

                    new_H[k][i] = a_ki/c_ki - epsilon
                    if new_H[k][i] < 0:
                        new_H[k][i]=0
                    
            #row(W)=1, all(H)=1
    ##        for m in range(M):
    ##            normalizeW = np.sum(new_W[m,:])
    ##            for k in range(K):
    ##                new_W[m][k] /= normalizeW
    ##
    ##        normalizeH = np.sum(new_H)
    ##        alpha = (normalizeH + K*N*epsilon)/(1+K*N*epsilon)
    ##        for k in range(K):
    ##            for i in range(N):
    ##                new_H[k][i] = (new_H[k][i] + (1-alpha)*epsilon)/alpha

            #col(W)=1, all(H)=1
            for k in range(K):
                normalizeW = np.sum(new_W[:,k])
                for m in range(M):
                    new_W[m][k] /= normalizeW
                for i in range(N):
                    new_H[k][i] = new_H[k][i]*normalizeW + epsilon * (normalizeW-1)

            normalizeH = np.sum(new_H)
            alpha = (normalizeH + K*N*epsilon)/(1+K*N*epsilon)
            for k in range(K):
                for i in range(N):
                    new_H[k][i] = (new_H[k][i] + (1-alpha)*epsilon)/alpha



            ###calculate difference
            difference_W = abs(new_W[0][0] - W[0][0])
            for m in range(M):
                for k in range(K):
                    if difference_W < abs(new_W[m][k] - W[m][k]):
                        difference_W = abs(new_W[m][k] - W[m][k])
            difference_H = abs(new_H[0][0] - H[0][0])
            for k in range(K):
                for i in range(N):
                    if difference_H < abs(new_H[k][i] - H[k][i]):
                        difference_H = abs(new_H[k][i] - H[k][i])
            difference = max(difference_W, difference_H)

            W = new_W
            H = new_H
            H_e = H + E
            
            loglikelihood = 0
            for m in range(M):
                for i in range(N):
                    for j in range(N):
                        if b[m][i][j]!=0:
                            loglikelihood += b[m][i][j] * (-math.log(np.dot(W[m,:],H_e[:,i])) + math.log(np.dot(W[m,:],H_e[:,i])+np.dot(W[m,:],H_e[:,j])))
                            
            old = new
            new = loglikelihood
            if new > old:
                print("Warning: loglikelihood increased!")
                break

            if difference < 1e-6:
                break


        print(x,' ', new)
        x += 1
        final_l.append(new)
        final_w.append(W)
        final_h.append(H)

    best = np.argmin(final_l)
    result_w = final_w[best]
    result_h = final_h[best]
    result_a = np.dot(result_w, result_h)
    final_a.append(result_a)

    #prediction_task
    total = 0
    correct = 0
    A = result_a

    for defender in list(names.keys()):
        if defender == 'Gael Monfils':
            continue
        df = pd.read_excel(xls, defender)
        num_column = len(df.columns)
        
        for player in list(df.index):
            for j in range(num_column):

                if type(df.loc[player][j]) == str:
                    result = df.loc[player][j]

                    if result[-2:] == year:
                        total += 1
                        if 'Australian' in result:
                            m = tournaments['Australian']
                            if 'Australian0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Roland' in result:
                            m = tournaments['Roland']
                            if 'Roland0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'USOpen' in result:
                            m = tournaments['USOpen']
                            if 'USOpen0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Monte' in result:
                            m = tournaments['Monte']
                            if 'Monte0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Wimbledon' in result:
                            m = tournaments['Wimbledon']
                            if 'Wimbledon0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Paris' in result:
                            m = tournaments['Paris']
                            if 'Paris0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Madrid' in result:
                            m = tournaments['Madrid']
                            if 'Madrid0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Rome' in result:
                            m = tournaments['Rome']
                            if 'Rome0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Indian' in result:
                            m = tournaments['Indian']
                            if 'Indian0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Miami' in result:
                            m = tournaments['Miami']
                            if 'Miami0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Canada' in result:
                            m = tournaments['Canada']
                            if 'Canada0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Cincinnati' in result:
                            m = tournaments['Cincinnati']
                            if 'Cincinnati0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Shanghai' in result:
                            m = tournaments['Shanghai']
                            if 'Shanghai0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1
                        elif 'Finals' in result:
                            m = tournaments['Finals']
                            if 'Finals0' in result:
                                true_y = 0
                            else:
                                true_y = 1

                            prob = A[m][names[defender]]/(A[m][names[defender]] + A[m][names[player]])
                            if prob > 0.5:
                                predict_y = 1
                            else:
                                predict_y = 0

                            if true_y == predict_y:
                                correct += 1

    print(year, "prediction accuracy: ",correct/total)
    final_acc[year] = correct/total

        
            
                        



    

import numpy as np
from FunctionCall import *

# Generate population function
def generate_population(LB, UB, pop_size, dim):
    return np.random.uniform(LB, UB,(pop_size, dim))

# The temperature profile around the huddle
def temp(Itr, MaxItr, R):
    if R >= 1:
        T = 0
    else:
        T = 1
    return T - Itr/(MaxItr - Itr)

# Fix bounds function
def fix_bounds(x, LB, UB):
    x_lenght = len(x)
    for i in range(x_lenght):
        if x[i] > UB or x[i] < LB:
            x[i] = np.random.uniform(LB, UB)
    return x

# Relocate penguin position
def Relocate(P, P_other, Itr, MaxItr, LB, UB, R, func1, CurEval, thershold = 0.5):
    # Compute the temperature profile around the huddle
    R = np.random.uniform(-1, 1)
    Temp = temp(Itr, MaxItr, R)
    # The polygon grid accuracy
    P_grid = abs(P - P_other)
    # Calculate avoid collision parameter
    A = 1.5 * (Temp + P_grid) * np.random.uniform() - Temp
    # The social forces of emperor penguins
    S = np.sqrt(np.random.uniform(3, 15) * np.exp(-Itr/np.random.uniform(3, 15)) - np.exp(-Itr))
    # Relocate penguin
    D = abs(S * P - np.random.uniform() * P_other)
    modified = fix_bounds(P - A * D, LB, UB)
    newposition = np.array([])
    for i in range(len(modified)):
        if np.random.uniform() > thershold:
            newposition = np.append(newposition,P[i])
        else:
            newposition = np.append(newposition, modified[i])
    Solutions = [P_other, newposition, modified]
    Evals = [func1(i) for i in Solutions]
    newposition = Solutions[np.argmin(Evals)]
    # print(min(Evals), 1)
    return newposition, func1(newposition)

# Emperor penguin optimizer
def PCA(LB, UB, dim, pop_size, MaxItr, R, fn, T = 0.4):
    try:
        func1 = func(fn)
        func1(np.random.uniform(LB, UB, dim))
    except Exception as e:
        func1 = lambda x:CEC(x, fn, dim)
    # Initialize the emperor penguins population
    population = generate_population(LB, UB, pop_size, dim)
    # Calculate the fitness value of each search agent
    fitness = [func1(x) for x in population]
    best = min(fitness)
    Gbest = population[np.argmin(fitness),:]
    
    # k = 1
    # P_vector_1 = np.array([])
    # P_vector_2 = np.arange(MaxItr)
    # i belongs to [0, MaxItr -1]
    for i in range(MaxItr):
        for j in range(pop_size):
            # Update position
            Current_Position = population[j]
            Relocating = Relocate(Gbest, Current_Position, i, MaxItr, LB, UB, R, func1,fitness[j], T)
            Current_Position = Relocating[0]
            fitness[j] = Relocating[1]
            # print(fitness[j], 2)
            # if Eval_modified <= fitness[j]:
            #     Eval_new_position = Eval_modified
            #     Current_Position = modified
            #     fitness[j] = Eval_modified
            # else:
            #     Eval_new_position = fitness[j]
            # Update the position of the optimal solution
            if fitness[j] <= best:
                best = fitness[j]
                Gbest = Current_Position
        # P_vector_1 = np.append(P_vector_1,best)

    return best, Gbest

# Relocate penguin position
def Relocate_classic(P, P_other, Itr, MaxItr, LB, UB, R):
    # Compute the temperature profile around the huddle
    R = np.random.uniform(0, R)
    Temp = temp(Itr, MaxItr, R)
    # The polygon grid accuracy
    P_grid = abs(P - P_other)
    # Calculate avoid collision parameter
    A = 2 * (Temp + P_grid) * np.random.uniform() - Temp
    # The social forces of emperor penguins
    S = np.sqrt(np.random.uniform(5, 10) * np.exp(-Itr/np.random.uniform(5, 50)) - np.exp(-Itr))
    # Relocate penguin
    D = abs(S * P - np.random.uniform() * P_other)
    return fix_bounds(P - A * D, LB, UB)

# Emperor penguin optimizer
def PCA_classic(LB, UB, dim, pop_size, MaxItr, R, fn):
    try:
        func1 = func(fn)
        func1(np.random.uniform(LB, UB, dim))
    except Exception as e:
        func1 = lambda x:CEC(x, fn, dim)
    # Initialize the emperor penguins population
    population = generate_population(LB, UB, pop_size, dim)
    # Calculate the fitness value of each search agent
    fitness = [func1(x) for x in population]
    Gbest = population[fitness.index(min(fitness)),:]
    best = min(fitness)
    # k = 1
    # P_vector_1 = np.array([])
    # P_vector_2 = np.arange(MaxItr)
    for i in range(MaxItr):
        for j in range(pop_size):
            # Update position
            Current_Position = population[j]
            Current_Position = Relocate_classic(Gbest, Current_Position, i, MaxItr, LB, UB, R)
            Eval_new_position = func1(Current_Position)
            # Update the position of the new optimal solution
            if Eval_new_position <= best:
                best = Eval_new_position
                Gbest = Current_Position
        # P_vector_1 = np.append(P_vector_1,best)
        

    # plt.plot(P_vector_2, P_vector_1)
    # plt.ylabel('Fitness')
    # plt.xlabel('Iteration')
    return best, Gbest

# Relocate penguin position
def RelocateWeighted(P, P_other, Itr, MaxItr, LB, UB, R, func1, thershold = 0.5):
    # Compute the temperature profile around the huddle
    R = np.random.uniform(0, R)
    Temp = temp(Itr, MaxItr, R)
    # The polygon grid accuracy
    P_grid = abs(P - P_other)
    # Calculate avoid collision parameter
    A = 2 * (Temp + P_grid) * np.random.uniform() - Temp
    # The social forces of emperor penguins
    S = np.sqrt(np.random.uniform(5, 50) * np.exp(-Itr/np.random.uniform(5, 50)) - np.exp(-Itr))
    # Relocate penguin
    D = abs(S * P - np.random.uniform() * P_other)
    modified = fix_bounds(P - A * D, LB, UB)
    EvalP = func1(P)
    EvalOther = func1(P_other)
    EvalModified = func1(modified)
    TotalEval = EvalP + EvalModified
    w1, w2 = EvalP/TotalEval, EvalModified/TotalEval
    newposition = weightedSum(P, modified,  w1, w2)
    # newposition = np.array([])
    # for i in range(len(modified)):
    #     if np.random.uniform() > thershold:
    #         newposition = np.append(newposition,P[i])
    #     else:
    #         newposition = np.append(newposition, modified[i])
    allSolutions = [P_other, modified, newposition]
    Fitness = [EvalOther, EvalModified, func1(newposition)]
    best = allSolutions[np.argmin(Fitness)]

    return best, min(Fitness)

def PCAWeighted(LB, UB, dim, pop_size, MaxItr, R, fn, T = 0.4):
    try:
        func1 = func(fn)
        func1(np.random.uniform(LB, UB, dim))
    except Exception as e:
        func1 = lambda x:CEC(x, fn, dim)
    # Initialize the emperor penguins population
    population = np.random.uniform(LB, UB, (pop_size, dim))
    # Calculate the fitness value of each search agent
    fitness = [func1(x) for x in population]
    Gbest = population[np.argmin(fitness),:]
    best = min(fitness)
    k = 1
    P_vector_1 = np.array([])
    # i belongs to [0, MaxItr -1]
    for i in range(MaxItr):
        for j in range(pop_size):
            # Update position
            Current_Position = population[j]
            Relocating = RelocateWeighted(Gbest, Current_Position, i, MaxItr, LB, UB, R, func1)
            Current_Position = Relocating[0]
            fitness[j] = Relocating[1]
            if fitness[j] <= best:
                best = fitness[j]
                Gbest = Current_Position
        P_vector_1 = np.append(P_vector_1,best)

    return best, Gbest

def weightedSum(sol1 = 0, sol2 = 0, sol3 = 0,  w1 = 0, w2 = 0, w3 = 0):
    return w1 * sol1 + w2 * sol2 + w3 * sol3

# Modified EPO based weighted sum and information vector
def EPOWIV(LB, UB, dim, pop_size, MaxItr, R, fn, T = 0.4):
    try:
        func1 = func(fn)
        func1(np.random.uniform(LB, UB, dim))
    except Exception as e:
        func1 = lambda x:CEC(x, fn, dim)
    # Initialize the emperor penguins population
    population = np.random.uniform(LB, UB, (pop_size, dim))
    # Calculate the fitness value of each search agent
    fitness = [func1(x) for x in population]
    Gbest = population[np.argmin(fitness),:]
    best = min(fitness)
    k = 1
    P_vector_1 = np.array([])
    # i belongs to [0, MaxItr -1]
    for i in range(MaxItr):
        for j in range(pop_size):
            # Update position
            Current_Position = population[j]
            Relocating = RelocateWIV(Gbest, Current_Position, i, MaxItr, LB, UB, R, func1)
            Current_Position = Relocating[0]
            fitness[j] = Relocating[1]
            if fitness[j] <= best:
                best = fitness[j]
                Gbest = Current_Position
        P_vector_1 = np.append(P_vector_1,best)

    return best, Gbest

def RelocateWIV(P, P_other, Itr, MaxItr, LB, UB, R, func1, thershold = 0.5):
    # Compute the temperature profile around the huddle
    R = np.random.uniform(-1, 1)
    Temp = temp(Itr, MaxItr, R)
    # The polygon grid accuracy
    P_grid = abs(P - P_other)
    # Calculate avoid collision parameter
    A = 1.5 * (Temp + P_grid) * np.random.uniform() - Temp
    # The social forces of emperor penguins
    S = np.sqrt(np.random.uniform(3, 15) * np.exp(-Itr/np.random.uniform(3, 15)) - np.exp(-Itr))
    # Relocate penguin
    D = abs(S * P - np.random.uniform() * P_other)
    modified = fix_bounds(P - A * D, LB, UB)
    # Weigthed sum process
    EvalP = func1(P)
    EvalOther = func1(P_other)
    EvalModified = func1(modified)
    TotalEval = EvalP + EvalOther + EvalModified
    # w1, w2, w3 = EvalP/TotalEval, EvalOther/TotalEval, EvalModified/TotalEval
    w1, w2 = EvalP/TotalEval, EvalModified/TotalEval
    newpositionw = fix_bounds(weightedSum(P, modified,  w1, w2), LB, UB)
    # Infromation vector process
    newpositionIV = np.array([])
    for i in range(len(modified)):
        if np.random.uniform() > thershold:
            newpositionIV = np.append(newpositionIV,P[i])
        else:
            newpositionIV = np.append(newpositionIV, newpositionw[i])
    Solutions = [P_other, newpositionIV, modified]
    Evals = [func1(i) for i in Solutions]
    newposition = Solutions[np.argmin(Evals)]
    # print(min(Evals), 1)
    return newposition, func1(newposition)

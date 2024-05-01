import numpy as np
class ressure_vessel_design:
    def __init__(self):
        pass


    def constraints(self):
        g1 = -self.z1 + 0.0193 * self.z3 <= 0
        g2 = -self.z2 + 0.00954 * self.z3 <= 0
        g3 = -np.pi * self.z3 ** 2 * self.z4 - (4/3) * np.pi * self.z3 ** 2 + 1296000 <= 0
        g4 = self.z4 - 240 <= 0
        if g1 and g2 and g3 and g4:
            return self.objectiveFn()
        else:
            return float('Inf')
    
    def objectiveFn(self):
        self.vector = np.array([self.z1, self.z2, self.z3, self.z4])
        return 0.6224 * (self.z1 * self.z3 * self.z4)\
        + 1.7781 * (self.z2 * self.z3 ** 2) + 3.1661 * self.z1 ** 2\
        * self.z4 + 19.84 * self.z1 ** 2 * self.z3
    
    def InitialSolution(self):
        self.z1 = np.random.randint(1, 99) * 0.0625
        self.z2 = np.random.randint(1, 99) * 0.0625
        self.z3 = np.random.uniform(10, 200)
        self.z4 = np.random.uniform(10, 200)
        
    
    def Fix(self):
        self.z1 = np.trunc(self.z1/0.0625) * 0.0625
        self.z2 = np.trunc(self.z2/0.0625) * 0.0625
        self.z3, self.z4 = np.clip([self.z3, self.z4], 10, 200)
    
    def returnPosition(self):
        return np.array([self.z1, self.z2, self.z3, self.z4])

    def receive(self, z1, z2, z3, z4):
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        self.z4 = z4


def PVD(sol):
    z1, z2, z3, z4 = Fix(sol)
    g1 = -z1 + 0.0193 * z3 <= 0
    g2 = -z3 + 0.00954 * z3 <= 0
    g3 = -np.pi * z3 ** 2 * z4 - (4/3) * np.pi * z3 ** 3 + 1296000 <= 0
    g4 = z4 - 240 <= 0
    g5 = 0.0625 <= z1 <= 99 * 0.0625
    g6 = 0.0625 <= z2 <= 99 * 0.0625
    g7 = 10 <= z3 <= 200
    g8 = 10 <= z4 <= 200
    if g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8:
        return 0.6224 * (z1 * z3 * z4)\
        + 1.7781 * (z2 * z3 ** 2) + 3.1661 * z1 ** 2\
        * z4 + 19.84 * z1 ** 2 * z3
    else:
        return float('Inf')

def generateSol():
    z1 = np.random.randint(1, 99) * 0.0625
    z2 = np.random.randint(1, 99) * 0.0625
    z3 = np.random.uniform(10, 200)
    z4 = np.random.uniform(10, 200)
    return np.array([z1, z2, z3, z4])

def Fix(sol):
    z1, z2, z3, z4 = sol
    z1 = np.trunc(z1/0.0625) * 0.0625
    z2 = np.trunc(z2/0.0625) * 0.0625
    z3, z4 = np.clip([z3, z4], 10, 200)
    return np.array([z1, z2, z3, z4])

def objFunc(z1, z2, z3, z4):
    return 0.6224 * (z1 * z3 * z4)\
    + 1.7781 * (z2 * z3 ** 2) + 3.1661 * z1 ** 2\
    * z4 + 19.84 * z1 ** 2 * z3
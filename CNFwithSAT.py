import random
def negate(literal):
    # Negate a literal
    return -literal

def is_unit(clause):
    # Check if a clause is unit
    return len(clause) == 1

def unit_propagate(clauses, model):
    # Perform unit propagation on a set of clauses
    # Return a new set of clauses and an updated model
    new_clauses = []
    for clause in clauses:
        if is_unit(clause):
            literal = clause[0]
            if literal not in model:
                model[literal] = True
            elif not model[literal]:
                return None, None  # Conflict detected
            for c in clauses:
                if literal in c and not len(c) == 1:
                    clauses.remove(c)
        new_clause = [l for l in clause if negate(l) not in model]
        if len(new_clause) == 0:
            return None, None  # Conflict detected
        new_clauses.append(new_clause)
    return new_clauses, model

def Pure_literal_elimination(clauses, model, items):

    for literal in items:
        index = 0
        for clause in clauses:
            if literal in clause and negate(literal) in clause:
                return None, None
            if literal in clause:
                index += 1
                flag = 1
                sat_clause = clause
            elif negate(literal) in clause:
                index += 1
                flag = -1
                sat_clause = clause

        if index == 1:
            if flag == 1:
                if literal in model:
                    if not model[literal]:
                        return None, None
                model[literal] = True

            else:
                if negate(literal) in model:
                    if not model[negate(literal)]:
                        return None, None
                model[negate(literal)] = True
            clauses.remove(sat_clause)
    return clauses, model


def choose_literal(clauses):
    # Choose a literal from a set of clauses using some heuristic
    # Here we use the simplest heuristic: pick the first literal we see
    for clause in clauses:
        for literal in clause:
            return literal

def dpll(clauses, model):
    # Apply DPLL algorithm recursively to find a satisfying assignment or report unsatisfiability
    for item in model:
        if model[item]:
            for c in clauses:
                if item in c:
                    clauses.remove(c)
                    continue
                elif negate(item) in c:
                    c.remove(negate(item))
    items = []
    for clause in clauses:
        for literal in clause:
            if literal not in items and negate(literal) not in items:
                items.append(literal)
    clauses, model = unit_propagate(clauses, model)
    if clauses is None:
        return False  # Unsatisfiable
    if len(clauses) == 0:
        return True, model  # Satisfiable
    clauses, model = Pure_literal_elimination(clauses, model, items)
    if clauses is None:
        return False  # Unsatisfiable
    if len(clauses) == 0:
        return True, model  # Satisfiable

    literal = choose_literal(clauses)

    new_model = dict(model)
    new_model[literal] = True

    result = dpll(clauses, new_model)

    if result is not None:
        return result  # Satisfiable with literal=True

    new_model[literal] = False

    result = dpll(clauses, new_model)

    if result is not None:
        return result  # Satisfiable with literal=False

def random_clause(k):
    size = random.randint(1,int(k/2))
    clause = []
    for i in range(size):
        sign = 1
        if random.randint(1,2) == 2:
            sign = -1
        literal = sign*random.randint(1,k)
        while literal in clause or -1*literal in clause:
            sign = 1
            if random.randint(1, 2) == 2:
                sign = -1
            literal = sign * random.randint(1, k)
        clause.append(literal)
    return clause

def verification(clauses, model):
    t_clauses = clauses.copy()
    t_model = dict(model)
    for item in t_model:
        if not model[item]:
            model[negate(item)] = True
    for item in model:
        if model[item]:
            for c in t_clauses:
                if item in c:
                    t_clauses.remove(c)
                    continue
                elif negate(item) in c:
                    if len(c) == 1:
                        return False


                    c.remove(negate(item))

    if len(t_clauses) == 0:
        return True
    else:
        return False

def save_CNF(PATH, clauses):

    with open(PATH, "w") as f:
        for clause in clauses:
            print(*clause, file=f)
    f.close()
    print('The CNF is saved in the path:' + PATH)

def load_CNF(PATH):
    clauses_n = []
    with open(PATH, "r") as f:
        content = f.read().split('\n')
        for item in content:
            if len(item) == 0:
                continue
            temp = item.split(' ')
            clauses_n.append([int(q) for q in temp])
    f.close()
    return clauses_n

def test_CNF(clauses, k):
    score = 0
    m = 500*k
    for j in range(m):
        config = [random.randint(0,1) for _ in range(k)]
        model = dict()
        for i in range(1,k+1):
            if config[i-1] == 1:
                model[i] = True
            else:
                model[i] = False
        flag = verification(clauses, model)
        # print('config:', config, flag)
        if flag:
            score+= 1
    return score/m

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import csv
    data = pd.read_csv("sqlite/sqlite.csv")
    X = np.array(data)[:,:-1]
    Y = np.array(data)[:,-1]
    name_list = list(data)
    clauses = load_CNF('sqlite/sqlite_CNF.txt')

    file1 = open("sqlite/sqlite_CNF.csv", "a+", newline="")
    content = csv.writer(file1)

    content.writerow(name_list)


    for j in range(len(Y)):
        model = {}
        for i in range(1,23):
            if X[j][i-1] == 1:
                model[i] = True
            else:
                model[i] = False
        if verification(clauses, model):
            content.writerow(np.append(list(X[j]), Y[j]))
        else:
            content.writerow(np.append(list(X[j]), -1))

    file1.close()






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 20:42:45 2025

"""

import gurobipy as gp
from gurobipy import GRB

# Data
demand = [300, 600, 300, 400, 800, 900, 1600, 1800, 1600, 800, 500, 400]
months = len(demand)

# Parameters
initial_inventory = 0
initial_workforce = 0
monthly_output_reg_worker = 50 
# monthly_output_ot_worker = 70
max_workforce_per_month = 20
max_inventory_per_month = 1000

cost_regular = 40
cost_overtime = 60  
cost_hiring = 0
cost_firing = 0
cost_holding = 2

# Model
model = gp.Model("aggregate_planning")

# Decision Variables
P = model.addVars(months, name="Regular_Production", lb=0)
O = model.addVars(months, name="Overtime_Production", lb=0)
I = model.addVars(months, name="Inventory", lb=0)
W = model.addVars(months, name="Workforce", lb=0)
H = model.addVars(months, vtype=GRB.INTEGER,name="Hiring", lb=0)
F = model.addVars(months, vtype=GRB.INTEGER,name="Firing", lb=0)
R = model.addVars(months, vtype=GRB.INTEGER,name="Regular_Time, lb=0")
S = model.addVars(months, vtype=GRB.INTEGER,name="Over_Time, lb=0")

# Objective Function
model.setObjective(
    gp.quicksum(
        cost_regular * P[t] + cost_overtime * O[t] + cost_hiring * H[t] + cost_firing * F[t] + cost_holding * I[t]
        for t in range(months)
    ),
    GRB.MINIMIZE,
)

# Constraints

# Inventory Balance
for t in range(months):
    if t == 0:
        model.addConstr(initial_inventory + P[t] + O[t] - I[t] == demand[t])
    else:
        model.addConstr(I[t-1] + P[t] + O[t] - I[t] == demand[t])
model.addConstrs(I[t] <= max_inventory_per_month
                 for t in range(months))

#Production Capacity
model.addConstrs(P[t] == monthly_output_reg_worker * R[t] 
                 for t in range(months))
model.addConstrs(O[t] <= 0.4 * P[t] 
                 for t in range(months))

# Workforce Balance
for t in range(months):
    if t == 0:
        model.addConstr(W[t] == initial_workforce + H[t] - F[t])
    else:
        model.addConstr(W[t] == W[t-1] + H[t] - F[t])
model.addConstrs(W[t] == R[t] + S[t] 
                 for t in range(months))
model.addConstrs(W[t] <= max_workforce_per_month
                 for t in range(months))

# Solve
model.optimize()

# Display results
if model.status == GRB.OPTIMAL:
    print('\nOptimal Solution:')
    for t in range(months):
        print(f"Month {t+1}:")
        print(f"  Regular Production: {P[t].x:.2f}")
        print(f"  Overtime Production: {O[t].x:.2f}")
        print(f"  Inventory: {I[t].x:.2f}")
        print(f"  Workforce: {W[t].x:.2f}")
        print(f"  Hiring: {H[t].x:.2f}")
        print(f"  Firing: {F[t].x:.2f}")
else:
    print('No optimal solution found.')

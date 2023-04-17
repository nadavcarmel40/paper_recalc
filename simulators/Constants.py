import numpy as np
from qutip import *

#################################################################################
##############                       constants                   ################
#################################################################################

h=6.626e-34
hbar=1.05e-34
freq=6e9

sigmaForError = np.pi/100
meanForError = 0

decoherence_mode = 0
num_counting = 3 #number of counting qubits
num_measuredOp = 1
beta = 0.02

gate_time_dict_SuperConducting = {'i':1,'X':1,'Y':1,'Z':1,'S':1,'T':1,'H':1,'CNOT':3,'CZ':3,'Rx':1,'Ry':1,'Rz':1,'SingleQubitOperator':1}
gate_time_dict_Ions = {'i':1,'X':1,'Y':1,'Z':1,'S':1,'T':1,'H':1,'CNOT':25,'CZ':25,'Rx':1,'Ry':1,'Rz':1,'SingleQubitOperator':1}
gate_time_dict_NeutralAtoms = {'i':1,'X':1,'Y':1,'Z':1,'S':1,'T':1,'H':1,'CNOT':5,'CZ':5,'Rx':1,'Ry':1,'Rz':1,'SingleQubitOperator':1}



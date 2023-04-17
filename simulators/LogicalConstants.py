from qutip import *
from simulators.Constants import *
from simulators.BigStepSimulation import EfficientQuantumRegister
from simulators.SmallStepSimulation import InCoherentQuantumRegister


##########################################     defining the logical states      ########################################
z1 = tensor([basis(2,0) for i in range(5)])
z2 = tensor([basis(2,0) if i in {1,2,4} else basis(2,1) for i in range(5)])
z3 = tensor([basis(2,0) if i in {0,2,3} else basis(2,1) for i in range(5)])
z4 = tensor([basis(2,0) if i in {1,3,4} else basis(2,1) for i in range(5)])
z5 = tensor([basis(2,0) if i in {0,2,4} else basis(2,1) for i in range(5)])
z6 = tensor([basis(2,0) if i in {2} else basis(2,1) for i in range(5)])
z7 = tensor([basis(2,0) if i in {0,1,4} else basis(2,1) for i in range(5)])
z8 = tensor([basis(2,0) if i in {2,3,4} else basis(2,1) for i in range(5)])
z9 = tensor([basis(2,0) if i in {3} else basis(2,1) for i in range(5)])
z10 = tensor([basis(2,0) if i in {0,1,2} else basis(2,1) for i in range(5)])
z11 = tensor([basis(2,0) if i in {4} else basis(2,1) for i in range(5)])
z12 = tensor([basis(2,0) if i in {0} else basis(2,1) for i in range(5)])
z13 = tensor([basis(2,0) if i in {1,2,3} else basis(2,1) for i in range(5)])
z14 = tensor([basis(2,0) if i in {0,3,4} else basis(2,1) for i in range(5)])
z15 = tensor([basis(2,0) if i in {1} else basis(2,1) for i in range(5)])
z16 = tensor([basis(2,0) if i in {0,1,3} else basis(2,1) for i in range(5)])

o1 = tensor([basis(2,1) for i in range(5)])
o2 = tensor([basis(2,1) if i in {1,2,4} else basis(2,0) for i in range(5)])
o3 = tensor([basis(2,1) if i in {0,2,3} else basis(2,0) for i in range(5)])
o4 = tensor([basis(2,1) if i in {1,3,4} else basis(2,0) for i in range(5)])
o5 = tensor([basis(2,1) if i in {0,2,4} else basis(2,0) for i in range(5)])
o6 = tensor([basis(2,1) if i in {2} else basis(2,0) for i in range(5)])
o7 = tensor([basis(2,1) if i in {0,1,4} else basis(2,0) for i in range(5)])
o8 = tensor([basis(2,1) if i in {2,3,4} else basis(2,0) for i in range(5)])
o9 = tensor([basis(2,1) if i in {3} else basis(2,0) for i in range(5)])
o10 = tensor([basis(2,1) if i in {0,1,2} else basis(2,0) for i in range(5)])
o11 = tensor([basis(2,1) if i in {4} else basis(2,0) for i in range(5)])
o12 = tensor([basis(2,1) if i in {0} else basis(2,0) for i in range(5)])
o13 = tensor([basis(2,1) if i in {1,2,3} else basis(2,0) for i in range(5)])
o14 = tensor([basis(2,1) if i in {0,3,4} else basis(2,0) for i in range(5)])
o15 = tensor([basis(2,1) if i in {1} else basis(2,0) for i in range(5)])
o16 = tensor([basis(2,1) if i in {0,1,3} else basis(2,0) for i in range(5)])

logical_0 = 1/4*(z1+z2+z3+z4+z5-z6-z7-z8-z9-z10-z11-z12-z13-z14-z15+z16)
logical_1 = 1/4*(o1+o2+o3+o4+o5-o6-o7-o8-o9-o10-o11-o12-o13-o14-o15+o16)
logic_plus = (logical_0+logical_1)/np.sqrt(2)
logic_plus_dm = logic_plus*logic_plus.dag()

# all single qubit errors
iiiii=tensor([qeye(2) for i in range(5)])
iiiix=tensor([qeye(2) if i!=4 else sigmax() for i in range(5)])
iiixi=tensor([qeye(2) if i!=3 else sigmax() for i in range(5)])
iixii=tensor([qeye(2) if i!=2 else sigmax() for i in range(5)])
ixiii=tensor([qeye(2) if i!=1 else sigmax() for i in range(5)])
xiiii=tensor([qeye(2) if i!=0 else sigmax() for i in range(5)])
iiiiz=tensor([qeye(2) if i!=4 else sigmaz() for i in range(5)])
iiizi=tensor([qeye(2) if i!=3 else sigmaz() for i in range(5)])
iizii=tensor([qeye(2) if i!=2 else sigmaz() for i in range(5)])
iziii=tensor([qeye(2) if i!=1 else sigmaz() for i in range(5)])
ziiii=tensor([qeye(2) if i!=0 else sigmaz() for i in range(5)])
iiiiy=tensor([qeye(2) if i!=4 else sigmay() for i in range(5)])
iiiyi=tensor([qeye(2) if i!=3 else sigmay() for i in range(5)])
iiyii=tensor([qeye(2) if i!=2 else sigmay() for i in range(5)])
iyiii=tensor([qeye(2) if i!=1 else sigmay() for i in range(5)])
yiiii=tensor([qeye(2) if i!=0 else sigmay() for i in range(5)])

################################################          defining registers               #########################

plus_temp = 1/np.sqrt(2)*(basis(2,0)+basis(2,1))
plus = plus_temp*plus_temp.dag()
initial_state_for_NoisyRegister_plus = tensor([fock_dm(2,0), plus])
initial_state_for_NoisyLogicalRegister_plus = tensor([fock_dm(2,0),fock_dm(2,0),fock_dm(2,0),fock_dm(2,0),fock_dm(2,0),plus])

# vector simulators

initial_state_for_PerfectLogicalRegister = lambda e: tensor([basis(2,0),basis(2,0),basis(2,0),basis(2,0),basis(2,0),basis(2,int(e))])

initial_state_for_PerfectRegister = lambda e: tensor([basis(2,0),basis(2,int(e))])

# noisy registers
initial_state_for_NoisyLogicalRegister = lambda e: tensor([fock_dm(2,0),fock_dm(2,0),fock_dm(2,0),fock_dm(2,0),fock_dm(2,0),fock_dm(2,int(e))])

initial_state_for_NoisyRegister = lambda e: tensor([fock_dm(2,0), fock_dm(2,int(e))])


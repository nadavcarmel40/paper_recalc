"""
This file has functions to create the data of figure 2 in the letter titled
"Hybrid Logical-Physical Qubit Interaction for Quantum Metrology"
re-creation of code from year 2021 for paper GitHub
Written by Nadav Carmel 17/04/2023


"""

#########################################################################################
######################################### imports #######################################
#########################################################################################

from simulators.LogicalConstants import *
import qiskit
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
import time
from simulators.Utils import bin2num
from qiskit.visualization import plot_histogram
from simulators.Utils import map_decohere_to_worst_gate_fidelity
from scipy.io import savemat
from simulators.Utils import round_sig

print('all packages imported')


#########################################################################################
########################## constants to control this file ###############################
#########################################################################################

# plot related variables
rotation_type = 'Rx'
noise_type = 'T2'
perfect_syndrome_extraction = False
name = 'recalc_paper'

# data creation related variables - default from paper is precision = 9, T_list = None, angle = 1/np.sqrt(3), e = '0'
precision = 9
T_list = None
# T_list = [10,100,1000,10000]
angle = 1/np.sqrt(3)
e = '0'

generate_data = True

plot_data_new = True

show_small_histograms = False

plot_data_from_2021 = False


#########################################################################################
################################### data creation #######################################
#########################################################################################


def debugLogical(state):
    """
    :param state: register state for the 5 qubit code+sensor as density matrix
    :return: logical density matrix
    """
    zLz = tensor([logical_0,basis(2,0)])
    oLz = tensor([logical_1,basis(2,0)])
    zLo = tensor([logical_0,basis(2,1)])
    oLo = tensor([logical_1,basis(2,1)])

    a00 = (zLz.dag()*state*zLz).data[0,0]
    a01 = (zLz.dag()*state*zLo).data[0,0]
    a02 = (zLz.dag()*state*oLz).data[0,0]
    a03 = (zLz.dag()*state*oLo).data[0,0]
    a10 = (zLo.dag()*state*zLz).data[0,0]
    a11 = (zLo.dag()*state*zLo).data[0,0]
    a12 = (zLo.dag()*state*oLz).data[0,0]
    a13 = (zLo.dag()*state*oLo).data[0,0]
    a20 = (oLz.dag()*state*zLz).data[0,0]
    a21 = (oLz.dag()*state*zLo).data[0,0]
    a22 = (oLz.dag()*state*oLz).data[0,0]
    a23 = (oLz.dag()*state*oLo).data[0,0]
    a30 = (oLo.dag()*state*zLz).data[0,0]
    a31 = (oLo.dag()*state*zLo).data[0,0]
    a32 = (oLo.dag()*state*oLz).data[0,0]
    a33 = (oLo.dag()*state*oLo).data[0,0]

    return Qobj(np.array([[a00,a01,a02,a03],[a10,a11,a12,a13],[a20,a21,a22,a23],[a30,a31,a32,a33]]), dims=[[2,2],[2,2]])

def applyOperator(reg,noisy,Operator):
    """
    applies operator on register
    :param reg: quantum register
    :param noisy: boolean - True for noisy register
    :param Operator: the operator to apply
    :return: None
    """
    if not noisy:
        reg.state = Operator*reg.state # apply permutation
    else:
        reg.state = Operator*reg.state*Operator.dag() # apply permutation

def H_for_LogicalRegister(reg, noisy):
    """
    runs logical hadamard for logical register with sensor qubit
    :param reg: quantum register object
    :param noisy: boolean - True for noisy register
    :return: None
    """
    perm = qiskit.circuit.library.Permutation(5,pattern=[3,0,2,4,1])
    operator = Qobj(qiskit.quantum_info.operators.Operator(perm).data, dims = [[2 for i in range(5)],[2 for i in range(5)]])
    reg.run([[('H',0,None,None),('H',1,None,None),('H',2,None,None),('H',3,None,None),('H',4,None,None)]])
    applyOperator(reg,noisy,-tensor([operator,qeye(2)]))

def prepare_for_LogicalRegister(reg,noisy):
    """
    runs logical state preperation for logical register with sensor qubit
    :param reg: quantum register object
    :param noisy: boolean - True for noisy register
    :return: None
    """
    # prepare |+++++>
    reg.run([[('H',0,None,None),('H',1,None,None),('H',2,None,None),('H',3,None,None),('H',4,None,None)]])
    # do black actions
    reg.run([[('CZ',0,1,None),('CZ',2,3,None)],[('CZ',1,2,None),('CZ',3,4,None)],[('CZ',0,4,None)]])
    # repair from |-> to |+>
    H_for_LogicalRegister(reg, noisy)
    reg.run([[('X',0,None,None),('X',1,None,None),('X',2,None,None),('X',3,None,None),('X',4,None,None)]])

def EC_for_LogicalRegister(reg, noisy, perfect=False, LPS=True,U='Rz'):
    """
    runs full error correction (syndrome measurement+correction) for logical register with sensor qubit.
    the error correction here is not fault-tolerant.
    :param reg: quantum register
    :param noisy: boolean - True for noisy register
    :param perfect: boolean - True if we want the qubit not to decohere when performing EC
    :return: None
    assuming qubits 0-4 are logicl and qubit 5 is sensor
    """
    # print('entering error correction')
    if U=='Rz':
        ## use sensor qubit as measurement qubit - set it in the |0> state assuming it is in |1>
        if e=='1':
            applyOperator(reg,noisy,reg.Sx[5]) # update mes qubit to be in state 0
    elif U=='Rx': #change state from |+> to |0> by applying instant Hadamard
        applyOperator(reg,noisy,(-1j * np.pi / 2 * 1 / np.sqrt(2) * (reg.Sx[5] + reg.Sz[5])).expm())

    # extract syndrome
    syndrome = ''

    def messureANDcollapse(LPS):
        if LPS:
            project = reg.qI

            ########## do logical post selection by forcing each generator to measure trivially, ##########
            ##########         and dont normalize the state to gather lost information           ##########

            project *= (reg.qI+reg.Sz[5])/2 #IIIII|0><0|
            applyOperator(reg,noisy,project) #    sensor
            return '0'
        else:
            mes_qubit_state = reg.state.ptrace([5])
            p0 = mes_qubit_state[0,0]
            p1 = mes_qubit_state[1,1]
            project = reg.qI
            normalize = reg.state.tr()
            # print(normalize)
            if np.random.rand() < p0:
                project *= (reg.qI+reg.Sz[5])/2 #I..I|0><0|I..I
                applyOperator(reg,noisy,project)
                reg.state = reg.state.unit()*normalize
                return '0'
            else:
                project *= (reg.qI-reg.Sz[5])/2 #I..I|1><1|I..I
                applyOperator(reg,noisy,project)
                reg.state = reg.state.unit()*normalize
                applyOperator(reg,noisy,reg.Sx[5]) # update mes qubit to be in state 0
                return '1'


    dephase = reg.dephase
    amp = reg.amplitude_damp
    if (noisy and perfect):
        reg.setError(dephase=False,amplitude_damp=False)

    # g1 syndrome extraction
    # gates
    reg.run([[('H',5,None,None)]])
    reg.run([[('CNOT',0,5,None)]])
    reg.run([[('CZ',1,5,None)]])
    reg.run([[('CZ',2,5,None)]])
    reg.run([[('CNOT',3,5,None)]])
    reg.run([[('H',5,None,None)]])
    syndrome += messureANDcollapse(LPS)

    # g2 syndrome extraction
    # gates
    reg.run([[('H',5,None,None)]])
    reg.run([[('CNOT',1,5,None)]])
    reg.run([[('CZ',2,5,None)]])
    reg.run([[('CZ',3,5,None)]])
    reg.run([[('CNOT',4,5,None)]])
    reg.run([[('H',5,None,None)]])
    syndrome += messureANDcollapse(LPS)

    # g3 syndrome extraction
    # gates
    reg.run([[('H',5,None,None)]])
    reg.run([[('CNOT',0,5,None)]])
    reg.run([[('CNOT',2,5,None)]])
    reg.run([[('CZ',3,5,None)]])
    reg.run([[('CZ',4,5,None)]])
    reg.run([[('H',5,None,None)]])
    syndrome += messureANDcollapse(LPS)

    # g4 syndrome extraction
    # gates
    reg.run([[('H',5,None,None)]])
    reg.run([[('CZ',0,5,None)]])
    reg.run([[('CNOT',1,5,None)]])
    reg.run([[('CNOT',3,5,None)]])
    reg.run([[('CZ',4,5,None)]])
    reg.run([[('H',5,None,None)]])
    syndrome += messureANDcollapse(LPS)

    # print(syndrome)

    # do recovery
    recovery = tensor([qeye(2) for i in range(6)])
    if syndrome == '0001':
        recovery = tensor([qeye(2) if i!=0 else sigmax() for i in range(6)])
    elif syndrome == '0010':
        recovery = tensor([qeye(2) if i!=2 else sigmaz() for i in range(6)])
    elif syndrome == '0011':
        recovery = tensor([qeye(2) if i!=4 else sigmax() for i in range(6)])
    elif syndrome == '0100':
        recovery = tensor([qeye(2) if i!=4 else sigmaz() for i in range(6)])
    elif syndrome == '0101':
        recovery = tensor([qeye(2) if i!=1 else sigmaz() for i in range(6)])
    elif syndrome == '0110':
        recovery = tensor([qeye(2) if i!=3 else sigmax() for i in range(6)])
    elif syndrome == '0111':
        recovery = tensor([qeye(2) if i!=4 else sigmay() for i in range(6)])
    elif syndrome == '1000':
        recovery = tensor([qeye(2) if i!=1 else sigmax() for i in range(6)])
    elif syndrome == '1001':
        recovery = tensor([qeye(2) if i!=3 else sigmaz() for i in range(6)])
    elif syndrome == '1010':
        recovery = tensor([qeye(2) if i!=0 else sigmaz() for i in range(6)])
    elif syndrome == '1011':
        recovery = tensor([qeye(2) if i!=0 else sigmay() for i in range(6)])
    elif syndrome == '1100':
        recovery = tensor([qeye(2) if i!=2 else sigmax() for i in range(6)])
    elif syndrome == '1101':
        recovery = tensor([qeye(2) if i!=1 else sigmay() for i in range(6)])
    elif syndrome == '1110':
        recovery = tensor([qeye(2) if i!=2 else sigmay() for i in range(6)])
    elif syndrome == '1111':
        recovery = tensor([qeye(2) if i!=3 else sigmay() for i in range(6)])

    applyOperator(reg,noisy,recovery)

    ## return sensor qubit to eigenstate
    if U=='Rz':
        if e=='1':
            applyOperator(reg,noisy,reg.Sx[5]) # update mes qubit to be in state 1
    elif U=='Rx': #change state from |+> to |0> by applying instant Hadamard
        applyOperator(reg,noisy,(-1j * np.pi / 2 * 1 / np.sqrt(2) * (reg.Sx[5] + reg.Sz[5])).expm())


    if (noisy and perfect):
        reg.setError(dephase=dephase,amplitude_damp=amp)

def rotation_for_LogicalRegister(reg,theta):
    """
    runs logical Z rotation for logical register with sensor qubit
    :param reg: quantum register object
    :param noisy: boolean - True for noisy register
    :param theta: the angle for rotation
    :return: None
    """
    K = Qobj(1/np.sqrt(2)*np.array([[1,1],[1j,-1j]]))
    reg.run([[('SingleQubitOperator',0,None,K),('Y',2,None,None),('SingleQubitOperator',4,None,K)]])
    reg.run([[('CNOT', 2, 4, None)]])
    reg.run([[('CNOT', 2, 0, None)]])
    reg.run([[('Rz', 2, None, theta)]])
    reg.run([[('CNOT', 2, 0, None)]])
    reg.run([[('CNOT', 2, 4, None)]])
    reg.run([[('SingleQubitOperator',0,None,K.dag()),('Y',2,None,None),('SingleQubitOperator',4,None,K.dag())]])

def IPEA_for_simpleRegister(angle, k, omega_k, noisy=False, reg_params = None, U = 'Rz', decay='T2'):
    """
    this function runs a single iteration of traditional IPEA.
    :param angle: the angle of the measured operator, with phi = 2pi*(angle) and 0<=angle<1
    :param k: int, the iteration, starting from m (number of desired digits) and ending in 1
    :param omega_k: the parameter as to which the phase kickback happens.
    :param noisy: True for noisy register, with only T2 noise
    :param reg_params: (T1,T2,T1s,T2s,dt,Tgate) as defined for the simulators (documentation there).
    :return: state of register right before the measurement.
    """

    if U=='Rz':
        initial = initial_state_for_NoisyRegister(e)
        Entangling_gate = 'CNOT'
    elif U=='Rx':
        initial = initial_state_for_NoisyRegister_plus
        Entangling_gate = 'CZ'
    if noisy:
        (T1,T2,T1s,T2s,dt,Tgate) = reg_params
        register = InCoherentQuantumRegister(2,initial,T1,T2,T1s=T1s,T2s=T2s,dt=dt,Tgate=Tgate)
        if decay == 'T2':
            register.setError(dephase=True,amplitude_damp=False)
        else:
            register.setError(dephase=False,amplitude_damp=True)
    else:
        (T1,T2,T1s,T2s,dt,Tgate) = reg_params
        register = InCoherentQuantumRegister(2,initial,T1,T2,T1s=T1s,T2s=T2s,dt=dt,Tgate=Tgate)
        register.setError(dephase=False,amplitude_damp=False)

    command = [[('H',0,None,None)]]
    register.run(command)

    for i in range(2**(k-1)):
        command = [[(U,1,None,2*np.pi*angle/4)]]
        register.run(command)

    command = [[(Entangling_gate,1,0,None)]]
    digit = register.run(command)

    for i in range(2**(k-1)):
        command = [[(U,1,None,-2*np.pi*angle/2)]]
        register.run(command)
        register.run(command)

    command = [[(Entangling_gate,1,0,None)]]
    register.run(command)

    for i in range(2**(k-1)):
        command = [[(U,1,None,2*np.pi*angle/4)]]
        register.run(command)

    command = [[('Rz',0,None,omega_k)]]
    register.run(command)

    command = [[('H',0,None,None)]]
    register.run(command)

    return register.state

def IPEA_for_logicalRegister(angle, k, omega_k, noisy=False, reg_params = None, U = 'Rz', decay='T2'):
    """
    this function runs a single iteration of traditional IPEA.
    :param angle: the angle of the measured operator, with phi = 2pi*(angle) and 0<=angle<1
    :param k: int, the iteration, starting from m (number of desired digits) and ending in 1
    :param omega_k: the parameter as to which the phase kickback happens.
    :param noisy: True for noisy register, with only T2 noise
    :param reg_params: (T1,T2,T1s,T2s,dt,Tgate) as defined for the simulators (documentation there).
    :return: state of register right before the measurement.
    """
    if U=='Rz':
        initial = initial_state_for_NoisyLogicalRegister(e)
        Entangling_gate = 'CNOT'
    elif U=='Rx':
        initial = initial_state_for_NoisyLogicalRegister_plus
        Entangling_gate = 'CZ'
    if noisy:
        (T1,T2,T1s,T2s,dt,Tgate) = reg_params
        register = InCoherentQuantumRegister(6,initial,T1,T2,T1s=T1s,T2s=T2s,dt=dt,Tgate=Tgate)
        if decay == 'T2':
            register.setError(dephase=True,amplitude_damp=False)
        else:
            register.setError(dephase=False,amplitude_damp=True)
    else:
        (T1,T2,T1s,T2s,dt,Tgate) = reg_params
        register = InCoherentQuantumRegister(6,initial,T1,T2,T1s=T1s,T2s=T2s,dt=dt,Tgate=Tgate)
        register.setError(dephase=False,amplitude_damp=False)

    prepare_for_LogicalRegister(register,True)

    H_for_LogicalRegister(register, True)

    # first sensor rotation
    for i in range(2**(k-1)):
        command = [[(U,5,None,2*np.pi*angle/4)]]
        register.run(command)
        # register.run([[('SingleQubitOperator',5,None,(-1j*2*np.pi*angle/8*sigmaz()).expm())]])

    # controlled not
    register.run([[(Entangling_gate, 5, 0, None),(Entangling_gate, 5, 1, None),(Entangling_gate, 5, 2, None),(Entangling_gate, 5, 3, None),(Entangling_gate, 5, 4, None)]])

    # second rotation
    for i in range(2**(k-1)):
        command = [[(U,5,None,-2*np.pi*angle/2)]]
        register.run(command)
        register.run(command)
        # register.run([[('SingleQubitOperator',5,None,(1j*2*np.pi*angle/8*sigmaz()).expm())]])
        # register.run([[('SingleQubitOperator',5,None,(1j*2*np.pi*angle/8*sigmaz()).expm())]])

    # controlled not
    register.run([[(Entangling_gate, 5, 0, None),(Entangling_gate, 5, 1, None),(Entangling_gate, 5, 2, None),(Entangling_gate, 5, 3, None),(Entangling_gate, 5, 4, None)]])

    # third and last
    for i in range(2**(k-1)):
        command = [[(U,5,None,2*np.pi*angle/4)]]
        register.run(command)
        # register.run([[('SingleQubitOperator',5,None,(-1j*2*np.pi*angle/8*sigmaz()).expm())]])

    rotation_for_LogicalRegister(register,omega_k)

    H_for_LogicalRegister(register, True)

    EC_for_LogicalRegister(register, True, LPS=True, U=U, perfect=perfect_syndrome_extraction)

    return register.state

def createData(T2, angle, precision, algorithm):
    """
    creates and saves all data connected to measureing the 'angle' up to a desired 'precision', with 'algorithm'
    being 'ideal', 'traditional' or 'logical'.
    """

    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\'+name+'\\Rz\\T2\\'+str(angle))
    else:
        folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\Rz\\T2\\' + str(angle))
    try:
        os.makedirs(folder)
    except:
        pass
    angle = 2*angle
    for k in range(precision, 0, -1):
        # create all possible rotations
        temp_omegas = [i for i in range(2**(precision-k))]
        omegas = [-2*np.pi/(2**(precision-k+1))*temp_omegas[i] for i in range(len(temp_omegas))]
        for i,omega_k in enumerate(omegas):
            if algorithm == 'ideal':
                reg_params = (1,T2,[1,1],[T2,T2],1/20,1)
                final_state = IPEA_for_simpleRegister(angle, k, omega_k, noisy=False, reg_params = reg_params)
            elif algorithm == 'traditional':
                reg_params = (1,T2,[1,1],[T2,T2],1/20,1)
                final_state = IPEA_for_simpleRegister(angle, k, omega_k, noisy=True, reg_params = reg_params)
            elif algorithm == 'logical':
                reg_params = (1,T2,[1,1,1,1,1,1],[T2,T2,T2,T2,T2,T2],1/20,1)
                final_state = IPEA_for_logicalRegister(angle, k, omega_k, noisy=True, reg_params = reg_params)
                final_state = debugLogical(final_state)
            try:
                np.save(os.path.join(folder, algorithm + '_' + str(k) + '_' + str(temp_omegas[i]/(2**(precision-k+1))) + '_T2_'+str(T2)),final_state)
            except:
                pass

def createData_Rx(T2, angle, precision, algorithm):
    """
    creates and saves all data connected to measureing the 'angle' up to a desired 'precision', with 'algorithm'
    being 'ideal', 'traditional' or 'logical'.
    same as createData, but activates Rx and ancillas have much better T2 then sensor
    """

    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\'+name+'\\Rx\\T2\\'+str(angle))
    else:
        folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\Rx\\T2\\' + str(angle))
    try:
        os.makedirs(folder)
    except:
        pass
    angle = 2*angle
    for k in range(precision, 0, -1):
        s = time.time()
        # create all possible rotations
        temp_omegas = [i for i in range(2**(precision-k))]
        omegas = [-2*np.pi/(2**(precision-k+1))*temp_omegas[i] for i in range(len(temp_omegas))]
        for i,omega_k in enumerate(omegas):
            if algorithm == 'ideal':
                reg_params = (1,T2,[1,1],[T2,T2],1/20,1)
                final_state = IPEA_for_simpleRegister(angle, k, omega_k, noisy=False, reg_params = reg_params, U='Rx')
            elif algorithm == 'traditional':
                reg_params = (1,T2,[1,1],[1e10*T2,T2],1/20,1)
                final_state = IPEA_for_simpleRegister(angle, k, omega_k, noisy=True, reg_params = reg_params, U='Rx')
            elif algorithm == 'logical':
                reg_params = (1,T2,[1,1,1,1,1,1],[1e10*T2,1e10*T2,1e10*T2,1e10*T2,1e10*T2,T2],1/20,1)
                final_state = IPEA_for_logicalRegister(angle, k, omega_k, noisy=True, reg_params = reg_params, U='Rx')
                final_state = debugLogical(final_state)
            try:
                np.save(os.path.join(folder, algorithm + '_' + str(k) + '_' + str(temp_omegas[i]/(2**(precision-k+1))) + '_T2_'+str(T2)),final_state)
            except:
                pass
        e = time.time()
        # print('finished iteration in ' + str(e-s) + ' seconds')

def createData_RxT1(T1, angle, precision, algorithm):
    """
    creates and saves all data connected to measureing the 'angle' up to a desired 'precision', with 'algorithm'
    being 'ideal', 'traditional' or 'logical'.
    same as createData, but activates Rx and ancillas have much better T2 then sensor
    """

    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\'+name+'\\Rx\\T1\\'+str(angle))
    else:
        folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\Rx\\T1\\' + str(angle))

    try:
        os.makedirs(folder)
    except:
        pass
    angle = 2*angle
    for k in range(precision, 0, -1)[:]:
        s = time.time()
        # create all possible rotations
        temp_omegas = [i for i in range(2**(precision-k))]
        omegas = [-2*np.pi/(2**(precision-k+1))*temp_omegas[i] for i in range(len(temp_omegas))]
        for i,omega_k in enumerate(omegas[:]):
            if algorithm == 'ideal':
                reg_params = (T1,1,[T1,T1],[1,1],1/20,1)
                final_state = IPEA_for_simpleRegister(angle, k, omega_k, noisy=False, reg_params = reg_params, U='Rx', decay='T1')
            elif algorithm == 'traditional':
                reg_params = (T1,1,[1e10*T1,T1],[1,1],1/20,1)
                final_state = IPEA_for_simpleRegister(angle, k, omega_k, noisy=True, reg_params = reg_params, U='Rx', decay='T1')
            elif algorithm == 'logical':
                reg_params = (T1,1,[1e10*T1,1e10*T1,1e10*T1,1e10*T1,1e10*T1,T1],[1,1,1,1,1,1],1/20,1)
                final_state = IPEA_for_logicalRegister(angle, k, omega_k, noisy=True, reg_params = reg_params, U='Rx', decay='T1')
                final_state = debugLogical(final_state)
            try:
                np.save(os.path.join(folder, algorithm + '_' + str(k) + '_' + str(temp_omegas[i]/(2**(precision-k+1))) + '_T2_'+str(T1)),final_state)
            except:
                pass
        e = time.time()
        # print('finished iteration in ' + str(e-s) + ' seconds')

def createData_RzT1(T1, angle, precision, algorithm):
    """
    creates and saves all data connected to measureing the 'angle' up to a desired 'precision', with 'algorithm'
    being 'ideal', 'traditional' or 'logical'.
    same as createData, but activates Rx and ancillas have much better T2 then sensor
    """

    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\Rz\\T1\\' + str(angle))
    else:
        folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\Rz\\T1\\' + str(angle))

    try:
        os.makedirs(folder)
    except:
        pass
    angle = 2*angle
    for k in range(precision, 0, -1)[:]:
        s = time.time()
        # create all possible rotations
        temp_omegas = [i for i in range(2**(precision-k))]
        omegas = [-2*np.pi/(2**(precision-k+1))*temp_omegas[i] for i in range(len(temp_omegas))]
        for i,omega_k in enumerate(omegas[:]):
            if algorithm == 'ideal':
                reg_params = (T1,1,[T1,T1],[1,1],1/20,1)
                final_state = IPEA_for_simpleRegister(angle, k, omega_k, noisy=False, reg_params = reg_params, U='Rz', decay='T1')
            elif algorithm == 'traditional':
                reg_params = (T1,1,[1e10*T1,T1],[1,1],1/20,1)
                final_state = IPEA_for_simpleRegister(angle, k, omega_k, noisy=True, reg_params = reg_params, U='Rz', decay='T1')
            elif algorithm == 'logical':
                reg_params = (T1,1,[1e10*T1,1e10*T1,1e10*T1,1e10*T1,1e10*T1,T1],[1,1,1,1,1,1],1/20,1)
                final_state = IPEA_for_logicalRegister(angle, k, omega_k, noisy=True, reg_params = reg_params, U='Rz', decay='T1')
                final_state = debugLogical(final_state)
            try:
                np.save(os.path.join(folder, algorithm + '_' + str(k) + '_' + str(temp_omegas[i]/(2**(precision-k+1))) + '_T2_'+str(T1)),final_state)
            except:
                pass
        e = time.time()
        # print('finished iteration in ' + str(e-s) + ' seconds')

def create_data_chooser(U,decay):
    if U == 'Rx':
        if decay == 'T2':
            return createData_Rx
        elif decay == 'T1':
            return createData_RxT1
        else:
            return None
    elif U == 'Rz':
        if decay == 'T2':
            return createData
        elif decay == 'T1':
            return createData_RzT1
        else:
            return None
    else:
        return None

def createHistogram(T2,angle,precision,algorithm,U='Rz',decay='T2'):
    """
    creates the histogram from previously saved data, by doing a sort of monta carlo simulation with num_trials
    """

    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\'+U+'\\'+decay+'\\' + str(angle))
    else:
        folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\'+U+'\\'+decay+'\\' + str(angle))

    if algorithm=='logical':
        dim = 6
    else:
        dim = 2

    d = {} #dictionary of results for histogram
    d_li = {}
    for i in range(2**(precision)):
        binNum = bin(i)[2:]
        while len(binNum)<precision:
            binNum = '0' + binNum

        P = 1
        theta = 0.0
        traces = []
        for k in range(precision,0,-1):

            state = Qobj(np.load(folder+'\\' + algorithm + '_' + str(k) + '_' + str(theta) + '_T2_'+str(T2) + '.npy'), dims = [[2 for i in range(dim)],[2 for i in range(dim)]])
            traces.append(state.tr())
            # if k == 9:
            #     print(state)

            if algorithm == 'logical':
                state = state/state.tr()
            P0 = state[0,0]+state[1,1]
            P1 = 1-P0

            correct_digit = binNum[k-1]
            if correct_digit == '0':
                P*=P0
            else:
                P*=P1

            # update theta
            theta = fracbin2num(binNum[k-1:])/2

        #calculate lost information from this run
        li_k = 0
        for i in range(len(traces)):
            li_iteration = 1
            for k in range(i):
                li_iteration*=traces[k]
            li_iteration*=(1-traces[i])
            li_k += li_iteration

        d_li[binNum] = li_k #update lost info
        d[binNum] = np.real(P) #update probability

    return d, np.sum(list(d_li.values()))/len(list(d.values()))

def fracbin2num(binary):
    return bin2num(binary)/2**len(binary)

def getSTD(d,li):
    results = []
    # create a list of all experiment results, with repetitions
    for key in d.keys():
        angle = fracbin2num(key)
        for i in range(int(d[key]*1e4)):
            results.append(angle)
    # get the STD
    std = np.std(results)
    return std/np.sqrt(1-li), std

def num2bin(angle, percision):
    """
    :param angle: number between 0 and 1
    :param percision: number of digits in result
    :return: the string for binary fraction for num
    """
    basis=[2**-(i+1) for i in range(percision)]
    result='0.'
    for i in basis:
        if i<=angle:
            result += '1'
            angle-=i
        else:
            result +='0'
    return result[:]

def createAllData(angle, precision, U='Rz', decay='T2', T_list=None):
    # creating data
    if T_list is None:
        T_list = list(np.geomspace(10, 1e3 + 100, 20, endpoint=False)) + list(np.geomspace(1e3, 1e7, 50, endpoint=True))
    start = time.time()
    if decay == 'T2':
        if U=='Rz':
            for T2 in T_list:
                createData(T2,angle,precision,'traditional')
                createData(T2,angle,precision,'ideal')
                createData(T2,angle,precision,'logical')
                print('created data for T2/Tgate = ' + str(T2) + ' in ' + str(time.time()-start)+' seconds')
            print(' ----------------     created all data        --------------')
        else:
            for T2 in T_list:
                createData_Rx(T2,angle,precision,'traditional')
                createData_Rx(T2,angle,precision,'logical')
                createData_Rx(T2,angle,precision,'ideal')
                print('created data for T2/Tgate = ' + str(T2) + ' in ' + str(time.time()-start)+' seconds')
            print(' ----------------     created all data        --------------')
    else:
        if U=='Rz':
            for T2 in T_list:
                createData_RzT1(T2,angle,precision,'traditional')
                createData_RzT1(T2,angle,precision,'ideal')
                createData_RzT1(T2,angle,precision,'logical')
                print('created data for T2/Tgate = ' + str(T2) + ' in ' + str(time.time()-start)+' seconds')
            print(' ----------------     created all data        --------------')
        else:
            for T2 in T_list:
                createData_RxT1(T2,angle,precision,'traditional')
                createData_RxT1(T2,angle,precision,'logical')
                createData_RxT1(T2,angle,precision,'ideal')
                print('created data for T2/Tgate = ' + str(T2) + ' in ' + str(time.time()-start)+' seconds')
            print(' ----------------     created all data        --------------')

def calc_STDs(angle, precision,U='Rz', decay='T2', T_list=None):
    # getting histograms:
    if T_list is None:
        T_list = list(np.geomspace(10,1e3+100,20, endpoint=False)) + list(np.geomspace(1e3,1e7,50, endpoint=True))
    STD_l = []
    STD_l_noLostInfo = []
    STD_t = []
    STD_t_noLostInfo = []
    STD_i = []
    STD_i_noLostInfo = []
    s = time.time()
    for T2 in T_list:
        d_i,li_i = createHistogram(T2,angle,precision,'ideal',U=U, decay=decay)
        STD_i.append(getSTD(d_i,li_i)[0])
        STD_i_noLostInfo.append(getSTD(d_i,li_i)[1])

        d_l,li_l = createHistogram(T2,angle,precision,'logical',U=U, decay=decay)
        d_t,li_t = createHistogram(T2,angle,precision,'traditional',U=U, decay=decay)

        STD_l.append(getSTD(d_l,li_l)[0])
        STD_l_noLostInfo.append(getSTD(d_l,li_l)[1])
        STD_t.append(getSTD(d_t,li_t)[0])
        STD_t_noLostInfo.append(getSTD(d_t,li_t)[1])

        print('created STDs for T2/Tgate = ' + str(T2) + ' in ' + str(time.time()-s)+' seconds')
    print(' ----------------     created all data        --------------')

    return STD_i,STD_t,STD_l,STD_i_noLostInfo,STD_t_noLostInfo,STD_l_noLostInfo

def save_all_precisions(angle, precision,U='Rz', decay='T2', T_list=None):
    STD_i0, STD_t0, STD_l0, STD_i_noLI0, STD_t_noLI0, STD_l_noLI0 = [], [], [], [], [], []
    for precision_int in range(precision):
        STD_i_it, STD_t_it, STD_l_it, STD_i_noLI_it, STD_t_noLI_it, STD_l_noLI_it = calc_STDs(angle, precision,
                                                                                         U=U, decay=decay,T_list=T_list)
        STD_i0.append(STD_i_it)
        STD_t0.append(STD_t_it)
        STD_l0.append(STD_l_it)
        STD_i_noLI0.append(STD_i_noLI_it)
        STD_t_noLI0.append(STD_t_noLI_it)
        STD_l_noLI0.append(STD_l_noLI_it)

    # save data
    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\'+U+'\\'+decay+'\\' + str(angle))
    else:
        folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\'+U+'\\'+decay+'\\' + str(angle))

    try:
        f_worst_1q = map_decohere_to_worst_gate_fidelity(T_list, 1, Tgate=1, decohere="2", save=False)
        f_worst_2q = map_decohere_to_worst_gate_fidelity(T_list, 2, Tgate=1, decohere="2", save=False)

        np.save(os.path.join(folder, 'logical_STDs'),STD_l0)
        np.save(os.path.join(folder, 'logical_STDs_noLI'),STD_l_noLI0)
        np.save(os.path.join(folder, 'traditional_STDs'),STD_t0)
        np.save(os.path.join(folder, 'traditional_STDs_noLI'),STD_t_noLI0)
        np.save(os.path.join(folder, 'ideal_STDs'),STD_i0)
        np.save(os.path.join(folder, 'ideal_STDs_noLI'),STD_i_noLI0)
        np.save(os.path.join(folder, 'f_worst_1q'), f_worst_1q)
        np.save(os.path.join(folder, 'f_worst_2q'), f_worst_2q)
        np.save(os.path.join(folder, 'T_list'), T_list)

        # savemat('IPEA_STDs_'+name + '_'+U+'_'+decay+'_' + str(round_sig(angle))+'_all.mat',
        #         dict(f_worst_1q=f_worst_1q, f_worst_2q=f_worst_2q, T_list=T_list, logical_STDs=STD_l0,
        #              logical_STDs_noLI=STD_l_noLI0, traditional_STDs=STD_t0, traditional_STDs_noLI=STD_t_noLI0,
        #              ideal_STDs=STD_i0, ideal_STDs_noLI=STD_i_noLI0))

    except:
        pass


if generate_data:
    createAllData(angle, precision, U=rotation_type, decay=noise_type, T_list=T_list)
    save_all_precisions(angle, precision, U=rotation_type, decay=noise_type, T_list=T_list)

if show_small_histograms:
    print("The binary representation of 1/sqrt2 to 10 binary digits is:", num2bin(1 / np.sqrt(2), 10))

    create_data_func = create_data_chooser(rotation_type,noise_type)

    precision = 3
    print('starting to create data for logical algorithm')
    start = time.time()
    create_data_func(1000, 1 / np.sqrt(2), precision, 'logical')
    print('ended in ' + str(time.time() - start) + ' seconds')

    print('starting to create data for traditional algorithm')
    start = time.time()
    create_data_func(1000, 1 / np.sqrt(2), precision, 'traditional')
    print('ended in ' + str(time.time() - start) + ' seconds')

    print('starting to create data for ideal algorithm')
    start = time.time()
    create_data_func(1000, 1 / np.sqrt(2), precision, 'ideal')
    print('ended in ' + str(time.time() - start) + ' seconds')


    precision = 3
    print('starting to create histogram for logical algorithm')
    d, li = createHistogram(1000, 1 / np.sqrt(2), precision, 'logical', U=rotation_type, decay=noise_type)
    print('lost information is: ' + str(li))
    plot_histogram(d)
    plt.title('logical_'+rotation_type+'_'+noise_type)
    plt.show()

    print('starting to create histogram for traditional algorithm')
    d, li = createHistogram(1000, 1 / np.sqrt(2), precision, 'traditional', U=rotation_type, decay=noise_type)
    print('lost information is: ' + str(li))
    plot_histogram(d)
    plt.title('traditional_'+rotation_type+'_'+noise_type)
    plt.show()

    print('starting to create histogram for ideal algorithm')
    d, li = createHistogram(1000, 1 / np.sqrt(2), precision, 'ideal', U=rotation_type,
                            decay=noise_type)
    print('lost information is: ' + str(li))
    plot_histogram(d)
    plt.title('ideal_'+rotation_type+'_'+noise_type)
    plt.show()


#########################################################################################
########### load data from folder for data generated now  ###############################
#########################################################################################

def load_IPEA_new(rotation_type,noise_type):
    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\'+rotation_type+'\\'+noise_type+'\\' + str(angle))
    else:
        folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\'+rotation_type+'\\'+noise_type+'\\' + str(angle))

    logical_STDs = np.load(os.path.join(folder, 'logical_STDs.npy'))
    logical_STDs_noLI = np.load(os.path.join(folder, 'logical_STDs_noLI.npy'))
    traditional_STDs = np.load(os.path.join(folder, 'traditional_STDs.npy'))
    traditional_STDs_noLI = np.load(os.path.join(folder, 'traditional_STDs_noLI.npy'))
    ideal_STDs = np.load(os.path.join(folder, 'ideal_STDs.npy'))
    ideal_STDs_noLI = np.load(os.path.join(folder, 'ideal_STDs_noLI.npy'))
    f_worst_1q_IPEA = np.load(os.path.join(folder, 'f_worst_1q.npy'))
    f_worst_2q_IPEA = np.load(os.path.join(folder, 'f_worst_2q.npy'))
    T_list_for_IPEA = np.load(os.path.join(folder, 'T_list.npy'))

    return ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA

if plot_data_new:
    ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA = load_IPEA_new(
        rotation_type, noise_type)

#########################################################################################
########### load data from folder for data generated in 2021  ###########################
#########################################################################################

def load_IPEA(rotation_type,noise_type):

    file = os.path.join('data_2021\\IPEA_Fisher', 'IPEASTDs' + noise_type + '_' + rotation_type + '_all')

    mat = loadmat(file)
    logical_STDs = mat['logical_STDs']
    logical_STDs_noLI = mat['logical_STDs_noLI']
    traditional_STDs = mat['traditional_STDs']
    traditional_STDs_noLI = mat['traditional_STDs_noLI']
    ideal_STDs = mat['ideal_STDs']
    ideal_STDs_noLI = mat['ideal_STDs_noLI']
    f_worst_1q_IPEA = mat['f_worst_1q'][0,:]
    f_worst_2q_IPEA = mat['f_worst_2q'][0,:]
    T_list_for_IPEA = mat['T2_list'][0,:]

    return ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA

def load_WCGF():

    f_worst_1q = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_1_T2.npy')
    f_worst_2q = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_2_T2.npy')
    f_worst_1q_T1 = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_1_T1.npy')
    f_worst_2q_T1 = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_2_T1.npy')
    f_worst_1q_CNOTs = np.load('data_2021\\WCGF\\N_1_decohere_mode_2_for_CNOTs_explore.npy')
    f_worst_2q_CNOTs = np.load('data_2021\\WCGF\\N_2_decohere_mode_2_for_CNOTs_explore.npy')
    return f_worst_1q,f_worst_2q,f_worst_1q_T1,f_worst_2q_T1,f_worst_1q_CNOTs,f_worst_2q_CNOTs

if plot_data_from_2021:

    ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA = load_IPEA(
        rotation_type, noise_type)

    f_worst_1q,f_worst_2q,f_worst_1q_T1,f_worst_2q_T1,_,_ = load_WCGF()

#########################################################################################
############################## plot #####################################################
#########################################################################################
def set_ticks_size(ax, x, y):
    # Set tick font size
    for label in (ax.get_xticklabels()):
        label.set_fontsize(x)
    # Set tick font size
    for label in (ax.get_yticklabels()):
        label.set_fontsize(y)


def plot_IPEA(precisions, withLI=True, x_points='single', save=False):
    fig, axes = plt.subplots(len(precisions))
    if withLI:
        ideal = ideal_STDs
        traditional = traditional_STDs
        logical = logical_STDs
    else:
        ideal = ideal_STDs_noLI
        traditional = traditional_STDs_noLI
        logical = logical_STDs_noLI

    if x_points == 'single':
        x = f_worst_1q_IPEA
        plt.xlabel('Worst-Case Single Gate Fidelity', fontsize=15)
        xstart = 0.9975
        xend = 1
    elif x_points == 'two':
        x = f_worst_2q_IPEA
        plt.xlabel('Worst-Case Entangling Gate Fidelity', fontsize=15)
        xstart = 0.995
        xend = 1
    else:
        x = T_list_for_IPEA
        plt.xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
        xstart = 10
        xend = 500
        # plt.xscale('log')

    print(x.shape)

    for i in range(len(axes)):
        j = precisions[i] - 1
        trad_data = traditional[j, :]
        ideal_data = ideal[j, :]
        logical_data = logical[j, :]

        axes[i].plot(np.NaN, np.NaN, label='precision=' + str(j + 1), color='black')
        if i == 0:
            axes[i].plot(x, trad_data - ideal_data, label='traditional', color='red')
            axes[i].plot(x, logical_data - ideal_data, label='logical 1LPS', color='blue')
        else:
            axes[i].plot(x, trad_data - ideal_data, color='red')
            axes[i].plot(x, logical_data - ideal_data, color='blue')
        axes[i].set_ylabel('$\sqrt{N} \left( \sigma - \sigma_{ideal} \\right)$')
        axes[i].set_xlim((xstart, xend))
        axes[i].set_ylim((0, 0.1))
        axes[i].legend()
        for label in (axes[i].get_xticklabels()):
            label.set_fontsize(13)

        if i != len(axes) - 1:
            axes[i].get_xaxis().set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig('images\\IPEA_paper')
    plt.show()


def plot_IPEA_infidelity(precisions, withLI=True, x_points='single', save=False):
    fig, axes = plt.subplots(len(precisions))
    if withLI:
        ideal = ideal_STDs
        traditional = traditional_STDs
        logical = logical_STDs
    else:
        ideal = ideal_STDs_noLI
        traditional = traditional_STDs_noLI
        logical = logical_STDs_noLI

    if x_points == 'single':
        x = 1 - np.array(f_worst_1q_IPEA)
        plt.xlabel('1 - Fidelity(single qubit gate)', fontsize=15)
        xstart = 1e-4
        xend = 1e-2
    elif x_points == 'two':
        print("not implemented yet")
    else:
        print("not implemented yet")

    print(x.shape)

    for i in range(len(axes)):
        j = precisions[i] - 1
        trad_data = traditional[j, :]
        ideal_data = ideal[j, :]
        logical_data = logical[j, :]

        axes[i].plot(np.NaN, np.NaN, label='precision=' + str(j + 1), color='black')
        if i == 0:
            axes[i].plot(x, trad_data - ideal_data, label='traditional', color='red')
            axes[i].plot(x, logical_data - ideal_data, label='logical 1LPS', color='blue')
        else:
            axes[i].plot(x, trad_data - ideal_data, color='red')
            axes[i].plot(x, logical_data - ideal_data, color='blue')
        axes[i].set_ylabel('$\sqrt{N} \left( \sigma - \sigma_{ideal} \\right)$')
        axes[i].set_xlim((xstart, xend))
        axes[i].set_ylim((1e-4, 1e1))
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].legend()
        for label in (axes[i].get_xticklabels()):
            label.set_fontsize(13)

        if i != len(axes) - 1:
            axes[i].get_xaxis().set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig('images\\IPEA_3_paper')
    plt.show()


def plot_IPEA_single_fig(precisions, withLI=False, x_points='single', save=True):
    fig, axes = plt.subplots()
    if withLI:
        ideal = ideal_STDs
        traditional = traditional_STDs
        logical = logical_STDs
    else:
        ideal = ideal_STDs_noLI
        traditional = traditional_STDs_noLI
        logical = logical_STDs_noLI

    if x_points == 'single':
        x = f_worst_1q_IPEA
        plt.xlabel('Worst-Case Single Gate Fidelity', fontsize=15)
        xstart = 0.9975
        # xstart = 0.9965
        xend = 1
    elif x_points == 'two':
        x = f_worst_2q_IPEA
        plt.xlabel('Worst-Case Entangling Gate Fidelity', fontsize=15)
        xstart = 0.995
        xend = 1
    else:
        x = T_list_for_IPEA
        plt.xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
        plt.xscale('log')
        xstart = 10
        xend = 2000

    print(x.shape)

    for i in range(len(precisions)):
        if i == 0:
            c = 'red'
        elif i == 1:
            c = 'blue'
        else:
            c = 'green'

        j = precisions[i] - 1
        trad_data = traditional[j, :]
        ideal_data = ideal[j, :]
        logical_data = logical[j, :]

        axes.plot(np.NaN, np.NaN, label='precision=' + str(j + 1), color=c)
        axes.plot(x, trad_data - ideal_data, '-', color=c)
        axes.plot(x, logical_data - ideal_data, '--', color=c)

        if i == 0:
            axes.scatter(0.999926, 0.0016, color=c)
        elif i == 1:
            axes.scatter(0.9979292, 0.073019, color=c)
        elif i == 2:
            axes.scatter(0.9989193, 0.04115, color=c)

        # if i == 0:
        #     axes.scatter(0.999926,0.0016,color=c)
        # elif i == 1:
        #     axes.scatter(0.99724, 0.07337, color=c)
        # elif i == 2:
        #     axes.scatter(0.99935,0.026,color=c)

    axes.set_ylabel('$\sqrt{N} \left( \sigma - \sigma_{ideal} \\right)$')
    axes.plot(np.NaN, np.NaN, '--', color='black', label='logical ancilla LPS')
    axes.plot(np.NaN, np.NaN, '-', color='black', label='physical ancilla')
    for label in (axes.get_xticklabels()):
        label.set_fontsize(13)
    axes.set_xlim((xstart, xend))
    axes.set_ylim((0, 0.1))
    # axes.set_yscale('log')
    axes.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig('images\\IPEA_2_supp')
    plt.show()


def plot_IPEA_single_fig_infidelity(precisions, withLI=False, x_points='single', save=True):
    fig, axes = plt.subplots()
    if withLI:
        ideal = ideal_STDs
        traditional = traditional_STDs
        logical = logical_STDs
    else:
        ideal = ideal_STDs_noLI
        traditional = traditional_STDs_noLI
        logical = logical_STDs_noLI

    if x_points == 'single':
        x = 1 - np.array(f_worst_1q_IPEA)
        plt.xlabel('1 - Fidelity(single qubit gate)', fontsize=15)
        xstart = 0.0001
        xend = 0.01
        plt.ylim((1e-4, 1e1))
        plt.yscale('log')
        plt.xscale('log')
    elif x_points == 'two':
        print("not implemented yet)")

    else:
        print("not implemented yet)")

    print(x.shape)

    for i in range(len(precisions)):
        if i == 0:
            c = 'red'
        elif i == 1:
            c = 'blue'
        else:
            c = 'green'

        j = precisions[i] - 1
        trad_data = traditional[j, :]
        ideal_data = ideal[j, :]
        logical_data = logical[j, :]

        axes.scatter(np.NaN, np.NaN, label='precision=' + str(j + 1), color=c)
        axes.plot(x, trad_data - ideal_data, '+', color=c)
        axes.plot(x, logical_data - ideal_data, '*', color=c)

        if i == 0:
            axes.scatter(0.999926, 0.0016, color=c)
        elif i == 1:
            axes.scatter(0.9979292, 0.073019, color=c)
        elif i == 2:
            axes.scatter(0.9989193, 0.04115, color=c)

    axes.set_ylabel('$\sqrt{N} \left( \sigma - \sigma_{ideal} \\right)$')
    axes.plot(np.NaN, np.NaN, '*', color='black', label='logical 1LPS')
    axes.plot(np.NaN, np.NaN, '+', color='black', label='traditional')
    for label in (axes.get_xticklabels()):
        label.set_fontsize(13)
    axes.set_xlim((xstart, xend))
    axes.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    set_ticks_size(axes, 13, 12)
    if save:
        plt.savefig('images\\IPEA_2_paper')
    plt.show()

if (plot_data_new or plot_data_from_2021):
    plot_IPEA_single_fig([1,2,3],withLI=True,x_points='single',save=False)
    # plot_IPEA([5,6,9],withLI=True,x_points='single',save=False)
    # plot_IPEA_infidelity([5,6,9],withLI=True,x_points='single',save=False)
    # plot_IPEA_single_fig([5,6,9],withLI=True,x_points='single',save=False)
    # plot_IPEA_single_fig_infidelity([5,6,9],withLI=True,x_points='single',save=False)


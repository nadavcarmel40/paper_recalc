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
from matplotlib.pyplot import cm
import pandas
import seaborn as sns
import operator as op
from functools import reduce
import heapq
from scipy.interpolate import interp1d

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

generate_data = False

plot_data_new = True

show_small_histograms = True

plot_data_from_2021 = False

plot_histogram_STD_for_ideal_simulation = False

generate_MEANs_for_2021 = False


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

def createData(T2, angle, precision, algorithm,folder_to_save=None):
    """
    creates and saves all data connected to measureing the 'angle' up to a desired 'precision', with 'algorithm'
    being 'ideal', 'traditional' or 'logical'.
    """

    path = os.getcwd()
    if folder_to_save is None:
        if perfect_syndrome_extraction:
            folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\'+name+'\\Rz\\T2\\'+str(angle))
        else:
            folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\Rz\\T2\\' + str(angle))
    else:
        folder = os.path.join(path,folder_to_save)
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

def createData_Rx(T2, angle, precision, algorithm,folder_to_save=None):
    """
    creates and saves all data connected to measureing the 'angle' up to a desired 'precision', with 'algorithm'
    being 'ideal', 'traditional' or 'logical'.
    same as createData, but activates Rx and ancillas have much better T2 then sensor
    """

    path = os.getcwd()
    if folder_to_save is None:
        if perfect_syndrome_extraction:
            folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\'+name+'\\Rx\\T2\\'+str(angle))
        else:
            folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\Rx\\T2\\' + str(angle))
    else:
        folder = os.path.join(path,folder_to_save)
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

def createData_RxT1(T1, angle, precision, algorithm,folder_to_save=None):
    """
    creates and saves all data connected to measureing the 'angle' up to a desired 'precision', with 'algorithm'
    being 'ideal', 'traditional' or 'logical'.
    same as createData, but activates Rx and ancillas have much better T2 then sensor
    """

    path = os.getcwd()
    if folder_to_save is None:
        if perfect_syndrome_extraction:
            folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\'+name+'\\Rx\\T1\\'+str(angle))
        else:
            folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\Rx\\T1\\' + str(angle))
    else:
        folder = os.path.join(path,folder_to_save)
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

def createData_RzT1(T1, angle, precision, algorithm,folder_to_save=None):
    """
    creates and saves all data connected to measureing the 'angle' up to a desired 'precision', with 'algorithm'
    being 'ideal', 'traditional' or 'logical'.
    same as createData, but activates Rx and ancillas have much better T2 then sensor
    """

    path = os.getcwd()
    if folder_to_save is None:
        if perfect_syndrome_extraction:
            folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\Rz\\T1\\' + str(angle))
        else:
            folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\Rz\\T1\\' + str(angle))
    else:
        folder = os.path.join(path,folder_to_save)
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

def createHistogram(T2,angle,precision,algorithm,U='Rz',decay='T2',new_data=True,folder_to_save=None):
    """
    creates the histogram from previously saved data, by doing a sort of monta carlo simulation with num_trials
    """
    path = os.getcwd()

    if folder_to_save is None:
        if new_data:
            data = 'data'
            if perfect_syndrome_extraction:
                folder = os.path.join(path, data+'\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\'+U+'\\'+decay+'\\' + str(angle))
            else:
                folder = os.path.join(path, data+'\\IPEA\\NoisySyndromeExtraction\\' + name + '\\'+U+'\\'+decay+'\\' + str(angle))
        else:
            data = 'data_2021'
            if perfect_syndrome_extraction:
                folder = os.path.join(path, data+'\\IPEA_Fisher\\' +U+'\\'+decay+'\\' + str(angle))
            else:
                folder = os.path.join(path, data+'\\IPEA_Fisher\\' +U+'\\'+decay+'\\' + str(angle))
    else:
        folder = os.path.join(path, folder_to_save)

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
    sum_x = 0
    sum_y = 0
    for key in d.keys():
        theta = bin2num(key)/2**len(key)
        p = d[key]
        sum_x += p*np.cos(theta)
        sum_y += p*np.sin(theta)
    R = np.sqrt(sum_x ** 2 + sum_y ** 2)
    std = np.sqrt(-2*np.log(R))
    return std/np.sqrt(1-li), std

def getMEAN(d):
    sum_x = 0
    sum_y = 0
    for key in d.keys():
        theta = 2*np.pi*bin2num(key)/2**len(key)
        p = d[key]
        sum_x += p*np.cos(theta)
        sum_y += p*np.sin(theta)
    if sum_y > 0:
        if sum_x > 0:
            theta_avg = np.arctan(np.abs(sum_y)/np.abs(sum_x))/np.pi
    if sum_y > 0:
        if sum_x < 0:
            theta_avg = 0.5+np.arctan(np.abs(sum_y)/np.abs(sum_x))/np.pi
    if sum_y < 0:
        if sum_x < 0:
            theta_avg = 1+np.arctan(np.abs(sum_y)/np.abs(sum_x))/np.pi
    if sum_y < 0:
        if sum_x > 0:
            theta_avg = 1.5+np.arctan(np.abs(sum_y)/np.abs(sum_x))/np.pi

    return theta_avg/2

def getSTD_regular(d,li):
    results = []
    # create a list of all experiment results, with repetitions
    for key in d.keys():
        angle = fracbin2num(key)
        for i in range(int(d[key]*1e4)):
            results.append(angle)
    # get the STD
    std = np.std(results)
    return std/np.sqrt(1-li), std

def getMEAN_regular(d):
    results = []
    # create a list of all experiment results, with repetitions
    for key in d.keys():
        angle = fracbin2num(key)
        for i in range(int(d[key]*1e4)):
            results.append(angle)
    # get the MEAN
    mean = np.mean(results)
    return mean

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

def calc_STDs(angle, precision,U='Rz', decay='T2', T_list=None, noisy_too=True,folder_to_save=None,new_data=True):
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
        d_i,li_i = createHistogram(T2,angle,precision,'ideal',U=U, decay=decay,folder_to_save=folder_to_save,new_data=new_data)
        STD_i.append(getSTD(d_i,li_i)[0])
        STD_i_noLostInfo.append(getSTD(d_i,li_i)[1])

        if noisy_too:
            d_l,li_l = createHistogram(T2,angle,precision,'logical',U=U, decay=decay,new_data=new_data)
            d_t,li_t = createHistogram(T2,angle,precision,'traditional',U=U, decay=decay,new_data=new_data)

            STD_l.append(getSTD(d_l,li_l)[0])
            STD_l_noLostInfo.append(getSTD(d_l,li_l)[1])
            STD_t.append(getSTD(d_t,li_t)[0])
            STD_t_noLostInfo.append(getSTD(d_t,li_t)[1])

        print('created STDs for T2/Tgate = ' + str(T2) + ' in ' + str(time.time()-s)+' seconds')
    print(' ----------------     created all data        --------------')

    return STD_i,STD_t,STD_l,STD_i_noLostInfo,STD_t_noLostInfo,STD_l_noLostInfo

def calc_MEANs(angle, precision,U='Rz', decay='T2', T_list=None, noisy_too=True,folder_to_save=None,new_data=True):
    # getting histograms:
    if T_list is None:
        T_list = list(np.geomspace(10,1e3+100,20, endpoint=False)) + list(np.geomspace(1e3,1e7,50, endpoint=True))
    MEAN_l = []
    MEAN_t = []
    MEAN_i = []
    s = time.time()
    for T2 in T_list:
        d_i,li_i = createHistogram(T2,angle,precision,'ideal',U=U, decay=decay,folder_to_save=folder_to_save,new_data=new_data)
        MEAN_i.append(getMEAN(d_i))

        if noisy_too:
            d_l,li_l = createHistogram(T2,angle,precision,'logical',U=U, decay=decay,new_data=new_data)
            d_t,li_t = createHistogram(T2,angle,precision,'traditional',U=U, decay=decay,new_data=new_data)

            MEAN_l.append(getMEAN(d_l))
            MEAN_t.append(getMEAN(d_t))

        print('created MEANs for T2/Tgate = ' + str(T2) + ' in ' + str(time.time()-s)+' seconds')
    print(' ----------------     created all data        --------------')

    return MEAN_i,MEAN_t,MEAN_l

def save_all_precisions(angle, precision,U='Rz', decay='T2', T_list=None,new_data=True, N=None):
    STD_i0, STD_t0, STD_l0, STD_i_noLI0, STD_t_noLI0, STD_l_noLI0 = [], [], [], [], [], []
    MEAN_i0, MEAN_t0, MEAN_l0, MEAN_i_noLI0, MEAN_t_noLI0, MEAN_l_noLI0 = [], [], [], [], [], []
    P_i0, P_t0, P_l0 = [], [], []
    if T_list is None:
        T_list = list(np.geomspace(10, 1e3 + 100, 20, endpoint=False)) + list(np.geomspace(1e3, 1e7, 50, endpoint=True))
    start = time.time()
    for precision_int in range(precision):
        if N is None:
            STD_i_it, STD_t_it, STD_l_it, STD_i_noLI_it, STD_t_noLI_it, STD_l_noLI_it = calc_STDs(angle, precision_int,
                                                                                             U=U, decay=decay,T_list=T_list,
                                                                                                  new_data=new_data)
            print(time.time()-start)
            MEAN_i_it, MEAN_t_it, MEAN_l_it = calc_MEANs(angle, precision_int, U=U,
                                                         decay=decay,T_list=T_list,new_data=new_data)
            print(time.time() - start)

        else:
            STD_i_it, STD_t_it, STD_l_it, STD_i_noLI_it, STD_t_noLI_it, STD_l_noLI_it, MEAN_i_it, MEAN_t_it, MEAN_l_it,\
            P_i_it,P_t_it,P_l_it = calc_STD_MEANs_N(N, angle, precision_int, U=U,
                                                         decay=decay,T_list=T_list,new_data=new_data)

        STD_i0.append(STD_i_it)
        STD_t0.append(STD_t_it)
        STD_l0.append(STD_l_it)
        STD_i_noLI0.append(STD_i_noLI_it)
        STD_t_noLI0.append(STD_t_noLI_it)
        STD_l_noLI0.append(STD_l_noLI_it)

        MEAN_i0.append(MEAN_i_it)
        MEAN_t0.append(MEAN_t_it)
        MEAN_l0.append(MEAN_l_it)

        P_i0.append(P_i_it)
        P_t0.append(P_t_it)
        P_l0.append(P_l_it)

    print(time.time() - start)
    # save data
    path = os.getcwd()
    if new_data:
        data = 'data'
        if perfect_syndrome_extraction:
            folder = os.path.join(path, data+'\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\'+U+'\\'+decay+'\\' + str(angle)+'\\' + str(N))
        else:
            folder = os.path.join(path, data+'\\IPEA\\NoisySyndromeExtraction\\' + name + '\\'+U+'\\'+decay+'\\' + str(angle)+'\\' + str(N))
    else:
        data = 'data_2021'
        if perfect_syndrome_extraction:
            folder = os.path.join(path, data+'\\IPEA_Fisher\\' +U+'\\'+decay+'\\' + str(angle))
        else:
            folder = os.path.join(path, data+'\\IPEA_Fisher\\' +U+'\\'+decay+'\\' + str(angle))

    f_worst_1q = map_decohere_to_worst_gate_fidelity(T_list, 1, Tgate=1, decohere="2", save=False)
    f_worst_2q = map_decohere_to_worst_gate_fidelity(T_list, 2, Tgate=1, decohere="2", save=False)

    try:
        os.makedirs(folder)
    except:
        pass

    np.save(os.path.join(folder, 'logical_STDs'),STD_l0)
    np.save(os.path.join(folder, 'logical_STDs_noLI'),STD_l_noLI0)
    np.save(os.path.join(folder, 'traditional_STDs'),STD_t0)
    np.save(os.path.join(folder, 'traditional_STDs_noLI'),STD_t_noLI0)
    np.save(os.path.join(folder, 'ideal_STDs'),STD_i0)
    np.save(os.path.join(folder, 'ideal_STDs_noLI'),STD_i_noLI0)
    np.save(os.path.join(folder, 'logical_MEANs'),MEAN_l0)
    np.save(os.path.join(folder, 'traditional_MEANs'),MEAN_t0)
    np.save(os.path.join(folder, 'ideal_MEANs'),MEAN_i0)
    np.save(os.path.join(folder, 'f_worst_1q'), f_worst_1q)
    np.save(os.path.join(folder, 'f_worst_2q'), f_worst_2q)
    np.save(os.path.join(folder, 'T_list'), T_list)
    np.save(os.path.join(folder, 'P_i'), P_i0)
    np.save(os.path.join(folder, 'P_t'), P_t0)
    np.save(os.path.join(folder, 'P_l'), P_l0)

def createHistogram_N(N,T2,angle,precision,algorithm,U='Rz',decay='T2',new_data=True,folder_to_save=None):
    """
    creates the histogram from previously saved data, by doing a sort of monta carlo simulation with num_trials
    """
    path = os.getcwd()

    if folder_to_save is None:
        if new_data:
            data = 'data'
            if perfect_syndrome_extraction:
                folder = os.path.join(path, data+'\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\'+U+'\\'+decay+'\\' + str(angle))
            else:
                folder = os.path.join(path, data+'\\IPEA\\NoisySyndromeExtraction\\' + name + '\\'+U+'\\'+decay+'\\' + str(angle))
        else:
            data = 'data_2021'
            if perfect_syndrome_extraction:
                folder = os.path.join(path, data+'\\IPEA_Fisher\\' +U+'\\'+decay+'\\' + str(angle))
            else:
                folder = os.path.join(path, data+'\\IPEA_Fisher\\' +U+'\\'+decay+'\\' + str(angle))
    else:
        folder = os.path.join(path, folder_to_save)

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

            P0_temp = state[0,0]+state[1,1]
            P1_temp = 1-P0_temp

            def ncr(n, r):
                r = min(r, n - r)
                numer = reduce(op.mul, range(n, n - r, -1), 1)
                denom = reduce(op.mul, range(1, r + 1), 1)
                return numer // denom

            P0 = 0
            P1 = 0
            j = 0
            while j <= N/2:
                P0 += ncr(N,j)*P1_temp**j*P0_temp**(N-j)
                P1 += ncr(N,j)*P0_temp**j*P1_temp**(N-j)
                j += 1

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

def calc_STD_MEANs_N(N,angle, precision,U='Rz', decay='T2', T_list=None, noisy_too=True,folder_to_save=None,new_data=True):
    # getting histograms:
    if T_list is None:
        T_list = list(np.geomspace(10,1e3+100,20, endpoint=False)) + list(np.geomspace(1e3,1e7,50, endpoint=True))
    STD_l = []
    STD_l_noLostInfo = []
    STD_t = []
    STD_t_noLostInfo = []
    STD_i = []
    STD_i_noLostInfo = []
    MEAN_l = []
    MEAN_t = []
    MEAN_i = []
    P_i = []
    P_t = []
    P_l = []
    s = time.time()
    for T2 in T_list:
        d_i,li_i = createHistogram_N(N,T2,angle,precision,'ideal',U=U, decay=decay,folder_to_save=folder_to_save,new_data=new_data)
        STD_i.append(getSTD(d_i,li_i)[0])
        STD_i_noLostInfo.append(getSTD(d_i,li_i)[1])
        MEAN_i.append(getMEAN(d_i))
        P_i.append(np.sum(heapq.nlargest(2, d_i.values())))

        if noisy_too:
            d_l,li_l = createHistogram_N(N,T2,angle,precision,'logical',U=U, decay=decay,new_data=new_data)
            d_t,li_t = createHistogram_N(N,T2,angle,precision,'traditional',U=U, decay=decay,new_data=new_data)

            STD_l.append(getSTD(d_l,li_l)[0])
            STD_l_noLostInfo.append(getSTD(d_l,li_l)[1])
            STD_t.append(getSTD(d_t,li_t)[0])
            STD_t_noLostInfo.append(getSTD(d_t,li_t)[1])
            MEAN_l.append(getMEAN(d_l))
            MEAN_t.append(getMEAN(d_t))
            P_t.append(np.sum(heapq.nlargest(2, d_t.values())))
            P_l.append(np.sum(heapq.nlargest(2, d_l.values())))

        print('created STDs for T2/Tgate = ' + str(T2) + ' in ' + str(time.time()-s)+' seconds')
    print(' ----------------     created all data        --------------')

    return STD_i,STD_t,STD_l,STD_i_noLostInfo,STD_t_noLostInfo,STD_l_noLostInfo, MEAN_i, MEAN_t, MEAN_l, P_i, P_t, P_l



def plot_histogram_STD_VS_num_digits(angle,max_precision,U,decay,folder_to_save,create_data=True,show=True):
    # create data
    if create_data:
        data_creator = create_data_chooser(U,decay)
        data_creator(1,angle,max_precision,'ideal',folder_to_save=folder_to_save)
    # calculate STDs for precisions
    def saver_in_folder():
        STD_i0, STD_i_noLI0 = [], []
        for precision_int in range(max_precision):
            STD_i_it, STD_t_it, STD_l_it, STD_i_noLI_it, STD_t_noLI_it, STD_l_noLI_it = calc_STDs(angle, precision_int,
                                                                                                  U=U, decay=decay,
                                                                                                  T_list=[1],noisy_too=False,
                                                                                                  folder_to_save=folder_to_save)
            STD_i0.append(STD_i_it)
            STD_i_noLI0.append(STD_i_noLI_it)

        # save data
        path = os.getcwd()
        folder = os.path.join(path,folder_to_save)
        try:
            np.save(os.path.join(folder, 'ideal_STDs'), STD_i0)
            np.save(os.path.join(folder, 'ideal_STDs_noLI'), STD_i_noLI0)
        except:
            pass

        return np.array(STD_i0), np.array(STD_i_noLI0)
    STD_i0, STD_i_noLI0 = saver_in_folder()
    stds = STD_i0[:,0]
    precisions = np.array(list(range(max_precision)))
    plt.scatter(precisions,stds,label=str(round_sig(angle,sig=precision+1)))
    plt.xlabel('desired precision - number of digits')
    plt.ylabel('histogram STD')
    plt.legend()
    if show:
        plt.show()

if generate_data:
    # createAllData(angle, precision, U=rotation_type, decay=noise_type, T_list=T_list)
    Ns = [1,5,10,20,30,40,50,60,70,80,90,100,120,150,200,300,500,1000,1500,2000,4000,10000,20000,50000,100000,500000,1000000]
    for N in Ns:
        save_all_precisions(angle, precision, U=rotation_type, decay=noise_type, T_list=T_list, N=N)


if generate_MEANs_for_2021:
    save_all_precisions(angle,precision,U=rotation_type,decay=noise_type,T_list=None,new_data=True)

if show_small_histograms:
    print("The binary representation of angle to 4 binary digits is:", num2bin(1 / np.sqrt(3), 4))
    print("The binary representation of angle to 4 binary digits is:", num2bin(1 / np.sqrt(3) + 2 ** (-4), 4))

    create_data_func = create_data_chooser(rotation_type, noise_type)


    def set_ticks_size(ax, x, y):
        # Set tick font size
        for label in (ax.get_xticklabels()):
            label.set_fontsize(x)
        # Set tick font size
        for label in (ax.get_yticklabels()):
            label.set_fontsize(y)


    def create_data(precision, type, T, angle):
        print('starting to create data for logical algorithm')
        start = time.time()
        create_data_func(T, angle, precision, type)
        print('ended in ' + str(time.time() - start) + ' seconds')


    # create_data(4,'logical',40,1/np.sqrt(3))
    # create_data(4,'traditional',40,1/np.sqrt(3))
    # create_data(4,'ideal',40,1/np.sqrt(3))

    def plot_histogram_wrapped(N,precision, T, angle, circular=True):
        print('starting to create histogram for logical algorithm')
        d_l, li = createHistogram_N(N,T, angle, precision, 'logical', U=rotation_type, decay=noise_type)
        d_t, li = createHistogram_N(N,T, angle, precision, 'traditional', U=rotation_type, decay=noise_type)
        d_i, li = createHistogram_N(N,T, angle, precision, 'ideal', U=rotation_type, decay=noise_type)

        keys_l = list(d_l.keys())
        vals_l = [d_l[k] for k in keys_l]
        keys_t = list(d_t.keys())
        vals_t = [d_t[k] for k in keys_t]
        keys_i = list(d_i.keys())
        vals_i = [d_i[k] for k in keys_i]
        keys_i = [bin2num(k) / 2 ** len(k) for k in d_i.keys()]
        print('lost information is: ' + str(li))
        df_periodic = pandas.DataFrame({
            'Factor': keys_i,
            '\n physical ancilla, \n $T_2=\infty$ \n $\\bar{\\theta}=$' + str(round(getMEAN(d_i), 3)) +
            ',\n $\sigma=$' + str(round(getSTD(d_i, 0)[0], 2)): vals_i,
            '\n logical ancilla, \n $T_2=40 T_{gate}$ \n $\\bar{\\theta}=$' + str(round(getMEAN(d_l), 3)) +
            ',\n $\sigma=$' + str(round(getSTD(d_l, 0)[0], 3)): vals_l,
            '\n physical ancilla, \n $T_2=40 T_{gate}$ \n $\\bar{\\theta}=$' + str(round(getMEAN(d_t), 3)) +
            ',\n $\sigma=$' + str(round(getSTD(d_t, 0)[0], 3)): vals_t,
        })
        # get values in the same order as keys, and parse percentage values

        df_regular = pandas.DataFrame({
            'Factor': keys_i,
            '\n physical ancilla, \n $T_2=\infty$ \n $\\bar{\\theta}=$' + str(round(getMEAN_regular(d_i), 3)) +
            ',\n $\sigma=$' + str(round(getSTD_regular(d_i, 0)[0], 2)): vals_i,
            '\n logical ancilla, \n $T_2=40 T_{gate}$ \n $\\bar{\\theta}=$' + str(round(getMEAN_regular(d_l), 3)) +
            ',\n $\sigma=$' + str(round(getSTD_regular(d_l, 0)[0], 3)): vals_l,
            '\n physical ancilla, \n $T_2=40 T_{gate}$ \n $\\bar{\\theta}=$' + str(round(getMEAN_regular(d_t), 3)) +
            ',\n $\sigma=$' + str(round(getSTD_regular(d_t, 0)[0], 3)): vals_t,
        })

        if circular:
            df = df_periodic
        else:
            df = df_regular

        fig, ax1 = plt.subplots(figsize=(20, 10))
        tidy = df.melt(id_vars='Factor').rename(columns=str.title)
        sns.barplot(x='Factor', y='Value', hue='Variable', data=tidy, ax=ax1)
        ax1.set_xlabel('Experiment Result', fontsize=30)
        ax1.set_ylabel('Probability', fontsize=30)
        plt.legend(fontsize=25, loc='upper left')
        sns.despine(fig)
        ax1.set_title('$\\theta=1/3$                                                       ', fontsize=25)
        plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])
        set_ticks_size(ax1, 22, 25)
        plt.savefig('images\\latest_histogram_'+str(N)+'.jpg')
        plt.show()

    def plot_histogram_wrapped_alg(precision, T, angle):
        print('starting to create histogram for logical algorithm')
        d_1, li = createHistogram_N(1,T, angle, precision, 'ideal', U=rotation_type, decay=noise_type)
        d_3, li = createHistogram_N(3,T, angle, precision, 'ideal', U=rotation_type, decay=noise_type)
        d_5, li = createHistogram_N(5,T, angle, precision, 'ideal', U=rotation_type, decay=noise_type)

        keys_1 = list(d_1.keys())
        vals_1 = [d_1[k] for k in keys_1]
        keys_3 = list(d_3.keys())
        vals_3 = [d_3[k] for k in keys_3]
        keys_5 = list(d_5.keys())
        vals_5 = [d_5[k] for k in keys_5]
        keys_1 = [bin2num(k) / 2 ** len(k) for k in d_1.keys()]
        print('lost information is: ' + str(li))
        df = pandas.DataFrame({
            'Factor': keys_1,
            '\n $n=5$,\n $P_{error}=$'+str(round(1-np.sum(heapq.nlargest(2, d_5.values())),3)): vals_5,
            '\n $n=3$, \n $P_{error}=$'+str(round(1-np.sum(heapq.nlargest(2, d_3.values())),3)): vals_3,
            '\n $n=1$, \n $P_{error}=$'+str(round(1-np.sum(heapq.nlargest(2, d_1.values())),3)): vals_1,
        })
        # get values in the same order as keys, and parse percentage values

        fig, ax1 = plt.subplots(figsize=(20, 10))
        tidy = df.melt(id_vars='Factor').rename(columns=str.title)
        sns.barplot(x='Factor', y='Value', hue='Variable', data=tidy, ax=ax1)
        ax1.set_xlabel('Experiment Result', fontsize=30)
        ax1.set_ylabel('Probability', fontsize=30)
        plt.legend(fontsize=25, loc='upper left')
        sns.despine(fig)
        ax1.set_title('ideal results, $\\theta=1/3$                                                  ', fontsize=25)
        plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])
        set_ticks_size(ax1, 22, 25)
        plt.savefig('images\\compare_ideal.jpg')
        plt.show()


    plot_histogram_wrapped(1,4, 40, 1 / 3,circular=False)
    # plot_histogram_wrapped(3,4, 40, 1 / 3)
    # plot_histogram_wrapped(5,4, 40, 1 / 3)
    plot_histogram_wrapped_alg(4, 40, 1 / 3)


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
    logical_MEANs = np.load(os.path.join(folder, 'logical_MEANs.npy'))
    traditional_MEANs = np.load(os.path.join(folder, 'traditional_MEANs.npy'))
    ideal_MEANs = np.load(os.path.join(folder, 'ideal_MEANs.npy'))
    f_worst_1q_IPEA = np.load(os.path.join(folder, 'f_worst_1q.npy'))
    f_worst_2q_IPEA = np.load(os.path.join(folder, 'f_worst_2q.npy'))
    T_list_for_IPEA = np.load(os.path.join(folder, 'T_list.npy'))

    return ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI,\
           logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA,\
           logical_MEANs,traditional_MEANs,ideal_MEANs

if plot_data_new:
    ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA,logical_MEANs,traditional_MEANs,ideal_MEANs = load_IPEA_new(
        rotation_type, noise_type)

#########################################################################################
########### load data from folder for data generated in 2021  ###########################
#########################################################################################

def load_IPEA(rotation_type,noise_type):

    file = os.path.join('data_2021\\IPEA_Fisher', 'IPEASTDs' + noise_type + '_' + rotation_type + '_all')

    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data_2021\\IPEA_Fisher\\' + rotation_type+'\\'+noise_type+'\\' + str(angle))
    else:
        folder = os.path.join(path, 'data_2021\\IPEA_Fisher\\' + rotation_type+'\\'+noise_type+'\\' + str(angle))

    logical_MEANs = np.load(os.path.join(folder, 'logical_MEANs.npy'))
    traditional_MEANs = np.load(os.path.join(folder, 'traditional_MEANs.npy'))
    ideal_MEANs = np.load(os.path.join(folder, 'ideal_MEANs.npy'))

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

    return ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA, logical_MEANs,traditional_MEANs,ideal_MEANs

def load_IPEA_old_python(rotation_type,noise_type):
    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data_2021\\IPEA_Fisher\\' +rotation_type+'\\'+noise_type+'\\' + str(angle))
    else:
        folder = os.path.join(path, 'data_2021\\IPEA_Fisher\\' +rotation_type+'\\'+noise_type+'\\' + str(angle))

    logical_STDs = np.load(os.path.join(folder, 'logical_STDs.npy'))
    logical_STDs_noLI = np.load(os.path.join(folder, 'logical_STDs_noLI.npy'))
    traditional_STDs = np.load(os.path.join(folder, 'traditional_STDs.npy'))
    traditional_STDs_noLI = np.load(os.path.join(folder, 'traditional_STDs_noLI.npy'))
    ideal_STDs = np.load(os.path.join(folder, 'ideal_STDs.npy'))
    ideal_STDs_noLI = np.load(os.path.join(folder, 'ideal_STDs_noLI.npy'))
    logical_MEANs = np.load(os.path.join(folder, 'logical_MEANs.npy'))
    traditional_MEANs = np.load(os.path.join(folder, 'traditional_MEANs.npy'))
    ideal_MEANs = np.load(os.path.join(folder, 'ideal_MEANs.npy'))
    f_worst_1q_IPEA = np.load(os.path.join(folder, 'f_worst_1q.npy'))
    f_worst_2q_IPEA = np.load(os.path.join(folder, 'f_worst_2q.npy'))
    T_list_for_IPEA = np.load(os.path.join(folder, 'T_list.npy'))

    return ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA, logical_MEANs,traditional_MEANs,ideal_MEANs


def load_WCGF():

    f_worst_1q = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_1_T2.npy')
    f_worst_2q = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_2_T2.npy')
    f_worst_1q_T1 = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_1_T1.npy')
    f_worst_2q_T1 = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_2_T1.npy')
    f_worst_1q_CNOTs = np.load('data_2021\\WCGF\\N_1_decohere_mode_2_for_CNOTs_explore.npy')
    f_worst_2q_CNOTs = np.load('data_2021\\WCGF\\N_2_decohere_mode_2_for_CNOTs_explore.npy')
    return f_worst_1q,f_worst_2q,f_worst_1q_T1,f_worst_2q_T1,f_worst_1q_CNOTs,f_worst_2q_CNOTs

if plot_data_from_2021:

    ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA,logical_MEANs,traditional_MEANs,ideal_MEANs = load_IPEA_old_python(
        rotation_type, noise_type)

    # f_worst_1q,f_worst_2q,f_worst_1q_T1,f_worst_2q_T1,_,_ = load_WCGF()

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


def plot_IPEA_single_fig(precisions, withLI=False, x_points='single', save=True,xstart=None,yend=None):
    fig, axes = plt.subplots()
    if withLI:
        ideal = ideal_STDs
        traditional = traditional_STDs
        logical = logical_STDs
    else:
        ideal = ideal_STDs_noLI
        traditional = traditional_STDs_noLI
        logical = logical_STDs_noLI
        xstart = None
        yend = None

    if x_points == 'single':
        x = f_worst_1q_IPEA
        plt.xlabel('Worst-Case Single Gate Fidelity', fontsize=15)
    elif x_points == 'two':
        x = f_worst_2q_IPEA
        plt.xlabel('Worst-Case Entangling Gate Fidelity', fontsize=15)
    else:
        x = T_list_for_IPEA
        plt.xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
        plt.xscale('log')

    print(x.shape)

    color = iter(['red','blue','green'])
    for i in range(len(precisions)):
        precision = precisions[i]
        c = next(color)

        j = precisions[i] - 1
        trad_data = traditional[j, :]
        ideal_data = ideal[j, :]
        logical_data = logical[j, :]

        if withLI:
            # fill between
            logic = np.abs(logical_data - ideal_data)
            interp_l = interp1d(x, logic)
            trad = np.abs(trad_data - ideal_data)
            interp_t = interp1d(x, trad)
            new_x = np.linspace(0.997, 0.99999, 10000)
            new_logic = interp_l(new_x)
            new_trad = interp_t(new_x)

            intersect_index = np.argmin(np.abs(new_logic - new_trad))
            plt.scatter(new_x[intersect_index], new_logic[intersect_index], color=c)
            axes.plot(np.NaN, np.NaN, label='$m=$' + str(precision), color=c)
        else:
            # fill between
            logic = np.abs(logical_data - ideal_data)
            interp_l = interp1d(x, logic)
            trad = np.abs(trad_data - ideal_data)
            interp_t = interp1d(x, trad)
            new_x = np.linspace(0.98, 0.99999, 10000)
            new_logic = interp_l(new_x)
            new_trad = interp_t(new_x)

            intersect_index = np.argmin(np.abs(new_logic - new_trad))
            plt.scatter(new_x[intersect_index], new_logic[intersect_index], color=c)
            axes.plot(np.NaN, np.NaN, label='$2^{-m}=$' + str(2**(-precision)) + ", $\sigma_{ideal}=$" + str(round_sig(ideal_data[0],sig=3)), color=c)


        axes.plot(new_x, new_trad, '-.', color=c)
        axes.plot(new_x, new_logic, '-', color=c)

        # if i == 0:
        #     axes.scatter(0.999926, 0.0016, color=c)
        # elif i == 1:
        #     axes.scatter(0.9979292, 0.073019, color=c)
        # elif i == 2:
        #     axes.scatter(0.9989193, 0.04115, color=c)

        # if i == 0:
        #     axes.scatter(0.999926,0.0016,color=c)
        # elif i == 1:
        #     axes.scatter(0.99724, 0.07337, color=c)
        # elif i == 2:
        #     axes.scatter(0.99935,0.026,color=c)
    if withLI:
        axes.set_ylabel('$\sqrt{N}\left(\\frac{\sigma}{\sqrt{1-l_i}} - \sigma_{ideal}\\right)$', fontsize = 13)
    else:
        axes.set_ylabel('$\sigma - \sigma_{ideal}$', fontsize=13)
    axes.plot(np.NaN, np.NaN, '-', color='black', label='logical ancilla LPS')
    axes.plot(np.NaN, np.NaN, '-.', color='black', label='physical ancilla')
    for label in (axes.get_xticklabels()):
        label.set_fontsize(13)
    if xstart is not None:
        axes.set_xlim((xstart, 1))
    if yend is not None:
        axes.set_ylim((0, yend))
    # axes.set_yscale('log')
    axes.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        if withLI:
            plt.savefig('images\\IPEA_STD_withLI')
        else:
            plt.savefig('images\\IPEA_STD_noLI')
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


def plot_IPEA_MEAN_single_fig(precisions, x_points='single', save=True, xstart=None,yend=None,ystart=0,xend=1):
    fig, axes = plt.subplots()
    ideal = ideal_MEANs
    traditional = traditional_MEANs
    logical = logical_MEANs
    if x_points == 'single':
        x = f_worst_1q_IPEA
        plt.xlabel('Worst-Case Single Gate Fidelity', fontsize=15)
        # xstart = 0.9975
        # xstart = 0.9965
        # xend = 1
    elif x_points == 'two':
        x = f_worst_2q_IPEA
        plt.xlabel('Worst-Case Entangling Gate Fidelity', fontsize=15)
        # xstart = 0.995
        # xend = 1
    else:
        x = T_list_for_IPEA
        plt.xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
        plt.xscale('log')
        # xstart = 10
        # xend = 2000

    print(x.shape)

    color = iter(['red','blue','green'])
    for i in range(len(precisions)):
        precision = precisions[i]
        c = next(color)

        axes.plot(np.NaN, np.NaN, '--', label='$2^{-m}=$' + str(2**(-precision)), color=c)

        j = precisions[i] - 1
        trad_data = traditional[j, :]
        ideal_data = ideal[j, :]
        logical_data = logical[j, :]

        # fill between
        logic = np.abs(logical_data - ideal_data)
        interp_l = interp1d(x, logic,kind='cubic')
        trad = np.abs(trad_data - ideal_data)
        interp_t = interp1d(x, trad,kind='cubic')
        precs = np.ones_like(x) * 2 ** (-precision)
        interp_precs = interp1d(x, precs)
        new_x = np.linspace(0.9825, 0.9999, 10000)
        new_logic = interp_l(new_x)
        new_trad = interp_t(new_x)
        new_precs = interp_precs(new_x)

        axes.plot(new_x, np.abs(new_trad), '-.', color=c)
        axes.plot(new_x, np.abs(new_logic), '-', color=c)
        axes.plot(x,np.ones_like(x)*2**(-precision),'--',color=c)

        intersect_index = np.argmin(np.abs(new_logic-new_trad))
        plt.scatter(new_x[intersect_index],new_logic[intersect_index], color=c)

        logic_fill_temp = new_logic[new_logic<=new_trad]
        trad_fill_temp = new_trad[new_logic<=new_trad]
        x_fill_temp = new_x[new_logic<=new_trad]
        precs_fill_temp = new_precs[new_logic<=new_trad]

        logic_fill = logic_fill_temp[trad_fill_temp>precs_fill_temp]
        trad_fill = trad_fill_temp[trad_fill_temp>precs_fill_temp]
        x_fill = x_fill_temp[trad_fill_temp>precs_fill_temp]
        plt.fill_between(x_fill, logic_fill, trad_fill, facecolor=c, edgecolor="none", alpha=.3)

    axes.set_ylabel('$|\\bar{\\theta} - \\bar{\\theta}_{ideal}|$', fontsize = 13)
    axes.plot(np.NaN, np.NaN, '-', color='black', label='logical ancilla LPS')
    axes.plot(np.NaN, np.NaN, '-.', color='black', label='physical ancilla')

    for label in (axes.get_xticklabels()):
        label.set_fontsize(13)
    if xstart is not None:
        axes.set_xlim((xstart, xend))
    if yend is not None:
        axes.set_ylim((ystart, yend))
        # axes.set_yscale('log')
    # axes.set_ylim((0, 0.1))
    # axes.set_yscale('log')
    # axes.set_xscale('log')
    axes.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig('images\\IPEA_MEAN_linear.jpg')
    plt.show()


def plot_estimated_number_of_trials(precisions, x_points='single', save=True):
    fig, axes = plt.subplots()
    ideal = ideal_STDs
    traditional = traditional_STDs
    logical = logical_STDs
    if x_points == 'single':
        x = f_worst_1q_IPEA
        plt.xlabel('Worst-Case Single Gate Fidelity', fontsize=15)
        # xstart = 0.9975
        xstart = 0.999965
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

    color = iter(cm.rainbow(np.linspace(0, 1, len(precisions))))
    for i in range(len(precisions)):
        precision = precisions[i]
        c = next(color)

        N_l = np.real(2**(2*precision) * (np.array(logical)[precision-1,:])**2)
        N_t = np.real(2**(2*precision) * (np.array(traditional)[precision-1,:])**2)
        N_i = np.real(2**(2*precision) * (np.array(ideal)[precision-1,:])**2)
        logic_difference = N_l.astype('int') - N_i.astype('int')
        trad_difference = N_t.astype('int') - N_i.astype('int')
        axes.plot(x, logic_difference,'-', c=c)
        axes.plot(x, trad_difference,'-.', c=c)
        axes.plot(np.NaN, np.NaN, label='$2^{-m}=$' + str(2 ** (-precision)) + ", $N_{ideal}=$" + str(int(np.ceil(N_i[0]))),
                  color=c)
        # fill between
        if precision>5:
            interp_l = interp1d(x, logic_difference,kind='cubic')
            interp_t = interp1d(x, trad_difference,kind='cubic')
            new_x = np.linspace(0.997, 0.9999, 10000)
            new_logic = interp_l(new_x)
            new_trad = interp_t(new_x)
            intersect_index = np.argmin(np.abs(new_logic - new_trad))
            plt.scatter(new_x[intersect_index], new_logic[intersect_index], color=c)
        # axes.plot(x, N_l,'--', c=c)
        # axes.plot(x, N_t,'-.', c=c)
        # axes.plot(x, N_i,'-', c=c)
    axes.plot(np.NaN, np.NaN, '-', color='black', label='logical ancilla LPS')
    axes.plot(np.NaN, np.NaN, '-.', color='black', label='physical ancilla')

    # axes.plot(np.NaN, np.NaN, '-', color='black', label='ideal (no decoherence)')
    # axes.set_ylabel('Estimated Minimal Number of Trials', fontsize=13)
    axes.set_ylabel('$N-N_{ideal}$', fontsize=13)
    for label in (axes.get_xticklabels()):
        label.set_fontsize(13)
    axes.set_xlim((0.995, 1))
    axes.set_ylim((1e0, 1e6))
    axes.set_yscale('log')
    axes.set_title('$N>2^{2m}\\frac{\sigma^2}{1-l_i}$', fontsize = 13)
    axes.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig('images\\IPEA_minimal_trials.jpg')
    plt.show()

if (plot_data_new or plot_data_from_2021):
    # plot_IPEA_single_fig([1,2,3],withLI=True,x_points='single',save=False)
    # plot_IPEA([5,6,9],withLI=True,x_points='single',save=False)
    # plot_IPEA_infidelity([5,6,9],withLI=True,x_points='single',save=False)
    plot_IPEA_single_fig([3,6,9],withLI=True,x_points='single',save=True,xstart=0.997,yend=0.15)
    plot_IPEA_single_fig([3,5,9],withLI=False,x_points='single',save=True)
    plot_IPEA_MEAN_single_fig([5,6,9],x_points='single',save=True,xstart=0.9825,yend=0.05,ystart=0,xend=1)
    plot_estimated_number_of_trials([3,5,6,7,8,9],x_points='single',save=True)
    # plot_IPEA_single_fig_infidelity([5,6,9],withLI=True,x_points='single',save=False)


if plot_histogram_STD_for_ideal_simulation:
    U= 'Rz'
    angles = [1/np.sqrt(2),1/np.sqrt(3),1/np.sqrt(5),1/np.sqrt(7),1/np.sqrt(11)]
    precision = 10
    for angle in angles:
        plot_histogram_STD_VS_num_digits(angle,precision,'Rz','T1','data\\IPEA_test\\'+U+'\\' + str(angle),show=False,create_data=True)
    plt.show()

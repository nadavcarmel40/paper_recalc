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
import pickle
import imageio



print('all packages imported')


#########################################################################################
########################## constants to control this file ###############################
#########################################################################################

# plot related variables
rotation_type = 'Rx'
noise_type = 'T2'
perfect_syndrome_extraction = False
name = 'make_videos'

# data creation related variables - default from paper is precision = 9, T_list = None, angle = 1/np.sqrt(3), e = '0'
precision = 4
T_list = None
# T_list = [10,100,1000,10000]
angle = 1/3
e = '0'

generate_data = False

plot_data_new = False

show_small_histograms = False

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

def getSTD(d,li,numeric=False):
    sum_x = 0
    sum_y = 0
    for key in d.keys():
        if numeric:
            theta = key
        else:
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

def getMEAN_circular_iterative(loaded_distribution,epsilon = 1 / 10000,numeric=True):
    curr_avg = 1 / 2
    prev_avg = 0
    num_iterations = 0
    while np.abs(prev_avg - curr_avg) > epsilon:
        num_iterations += 1
        prev_avg = curr_avg
        curr_avg = 0
        for key in loaded_distribution.keys():
            if not numeric:
                theta = bin2num(key)/2**len(key)
            else:
                theta = key
            if np.abs(theta - prev_avg) < 0.5:
                curr_avg += loaded_distribution[key] * theta
            else:
                if theta < prev_avg:
                    curr_avg += loaded_distribution[key] * (theta + 1)
                else:
                    curr_avg += loaded_distribution[key] * (theta - 1)
    return curr_avg

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

def save_all_precisions(angle, precision,U='Rz', decay='T2', T_list=None,new_data=True, N=None,load=True):
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
                                                         decay=decay,T_list=T_list,new_data=new_data,load=load)

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

def createHistogram_N(N,T2,angle,precision,algorithm,U='Rz',decay='T2',new_data=True,folder_of_data=None,folder_to_save=None):
    """
    creates the histogram from previously saved data, by doing a sort of monta carlo simulation with num_trials
    """
    path = os.getcwd()

    if folder_of_data is None:
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
        folder = os.path.join(path, folder_of_data)

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

    li = np.sum(list(d_li.values()))/len(list(d.values()))

    # save data
    path = os.getcwd()
    if folder_to_save is None:
        if new_data:
            data = 'data'
            if perfect_syndrome_extraction:
                folder_to_save = os.path.join(path,
                                              data + '\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\' + U + '\\' + decay + '\\' + str(
                                                  angle) + '\\' + str(N))
            else:
                folder_to_save = os.path.join(path,
                                      data + '\\IPEA\\NoisySyndromeExtraction\\' + name + '\\' + U + '\\' + decay + '\\' + str(
                                          angle) + '\\' + str(N))
        else:
            data = 'data_2021'
            if perfect_syndrome_extraction:
                folder_to_save = os.path.join(path, data + '\\IPEA_Fisher\\' + U + '\\' + decay + '\\' + str(angle))
            else:
                folder_to_save = os.path.join(path, data + '\\IPEA_Fisher\\' + U + '\\' + decay + '\\' + str(angle))
    else:
        folder_to_save = os.path.join(path, folder_to_save)

    distribution_with_vals = {}
    for k in d.keys():
        distribution_with_vals[bin2num(k) / 2 ** len(k)] = d[k]
    dict_to_save = {'distribution': distribution_with_vals, 'lost_info': li}

    try:
        os.makedirs(folder_to_save)
    except:
        pass

    with open(os.path.join(folder_to_save,algorithm+'_'+str(precision)+'_'+str(T2)+'.pkl'), 'wb') as f:
        pickle.dump(dict_to_save, f)

    return d, li

def loadHistogram_N(N,T2,angle,precision,algorithm,U='Rz',decay='T2',new_data=True,folder_to_save=None):
    # load data
    path = os.getcwd()
    if folder_to_save is None:
        if new_data:
            data = 'data'
            if perfect_syndrome_extraction:
                folder_to_save = os.path.join(path,
                                              data + '\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\' + U + '\\' + decay + '\\' + str(
                                                  angle) + '\\' + str(N))
            else:
                folder_to_save = os.path.join(path,
                                      data + '\\IPEA\\NoisySyndromeExtraction\\' + name + '\\' + U + '\\' + decay + '\\' + str(
                                          angle) + '\\' + str(N))
        else:
            data = 'data_2021'
            if perfect_syndrome_extraction:
                folder_to_save = os.path.join(path, data + '\\IPEA_Fisher\\' + U + '\\' + decay + '\\' + str(angle))
            else:
                folder_to_save = os.path.join(path, data + '\\IPEA_Fisher\\' + U + '\\' + decay + '\\' + str(angle))
    else:
        folder_to_save = os.path.join(path, folder_to_save)

    with open(os.path.join(folder_to_save,algorithm+'_'+str(precision)+'_'+str(T2)+'.pkl'), 'rb') as f:
        loaded_dict = pickle.load(f)

    loaded_distribution = loaded_dict['distribution']
    loaded_lost_info = loaded_dict['lost_info']

    return loaded_distribution, loaded_lost_info

def calc_STD_MEANs_N(N,angle, precision,U='Rz', decay='T2', T_list=None, noisy_too=True,folder_of_data=None,new_data=True,load=True):
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
        if not load:
            d_i,li_i = createHistogram_N(N,T2,angle,precision,'ideal',U=U, decay=decay,folder_of_data=folder_of_data,new_data=new_data)
        else:
            d_i, li_i = loadHistogram_N(N, T2, angle, precision, 'ideal', U=U, decay=decay,
                                          new_data=new_data)
        stds = getSTD(d_i,li_i,numeric=load)
        STD_i.append(stds[0])
        STD_i_noLostInfo.append(stds[1])
        MEAN_i.append(getMEAN_circular_iterative(d_i,numeric=load))
        P_i.append(np.sum(heapq.nlargest(2, d_i.values())))

        if noisy_too:
            if not load:
                d_l,li_l = createHistogram_N(N,T2,angle,precision,'logical',U=U, decay=decay,new_data=new_data)
                d_t,li_t = createHistogram_N(N,T2,angle,precision,'traditional',U=U, decay=decay,new_data=new_data)
            else:
                d_l, li_l = loadHistogram_N(N, T2, angle, precision, 'logical', U=U, decay=decay, new_data=new_data)
                d_t, li_t = loadHistogram_N(N, T2, angle, precision, 'traditional', U=U, decay=decay,
                                              new_data=new_data)

            stds = getSTD(d_l,li_l,numeric=load)
            STD_l.append(stds[0])
            STD_l_noLostInfo.append(stds[1])
            stds = getSTD(d_t, li_t, numeric=load)
            STD_t.append(stds[0])
            STD_t_noLostInfo.append(stds[1])
            MEAN_l.append(getMEAN_circular_iterative(d_l,numeric=load))
            MEAN_t.append(getMEAN_circular_iterative(d_t,numeric=load))
            P_t.append(np.sum(heapq.nlargest(2, d_t.values())))
            P_l.append(np.sum(heapq.nlargest(2, d_l.values())))

        print('created STDs for T2/Tgate = ' + str(T2) + ' in ' + str(time.time()-s)+' seconds')
    print(' ----------------     created all data        --------------')

    return STD_i,STD_t,STD_l,STD_i_noLostInfo,STD_t_noLostInfo,STD_l_noLostInfo, MEAN_i, MEAN_t, MEAN_l, P_i, P_t, P_l

def set_ticks_size(ax, x, y):
    # Set tick font size
    for label in (ax.get_xticklabels()):
        label.set_fontsize(x)
    # Set tick font size
    for label in (ax.get_yticklabels()):
        label.set_fontsize(y)

def create_data(precision, type, T, angle):
    create_data_func = create_data_chooser(rotation_type, noise_type)
    print('starting to create data for algorithm')
    start = time.time()
    create_data_func(T, angle, precision, type)
    print('ended in ' + str(time.time() - start) + ' seconds')

def plot_histogram_wrapped_noise_single_algorithm(d_i,d_n,T,algorithm='traditional'):
    keys_t = list(d_n.keys())
    vals_t = [d_n[k] for k in keys_t]
    keys_i = list(d_i.keys())
    vals_i = [d_i[k] for k in keys_i]
    df = pandas.DataFrame({
        'Factor': keys_i,
        '\n $T_2=\infty$ \n $\\bar{\\theta}=$' + str(round(getMEAN_circular_iterative(d_i), 3)) +
        ',\n $\sigma=$' + str(round(getSTD(d_i, 0, numeric=True)[0], 2)): vals_i,
        '\n $T_2='+str(round_sig(T,sig=3))+' T_{gate}$ \n $\\bar{\\theta}=$' + str(round(getMEAN_circular_iterative(d_n), 3)) +
        ',\n $\sigma=$' + str(round(getSTD(d_n, 0, numeric=True)[0], 3)): vals_t,
    })
    # get values in the same order as keys, and parse percentage values

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
    path = os.getcwd()
    filename = os.path.join(path,'images\\for_video\\'+algorithm+'_'+str(T)+'.jpg')
    plt.savefig(filename)
    plt.show()

def plot_histogram_wrapped_noise_both_algorithms(d_i,d_t,d_l,T):
    keys_l = list(d_l.keys())
    vals_l = [d_l[k] for k in keys_l]
    keys_t = list(d_t.keys())
    vals_t = [d_t[k] for k in keys_t]
    keys_i = list(d_i.keys())
    vals_i = [d_i[k] for k in keys_i]
    df = pandas.DataFrame({
        'Factor': keys_i,
        '\n physical ancilla, \n $T_2=\infty$ \n $\\bar{\\theta}=$' + str(round(getMEAN_circular_iterative(d_i), 3)) +
        ',\n $\sigma=$' + str(round(getSTD(d_i, 0, numeric=True)[0], 2)): vals_i,
        '\n logical ancilla, \n $T_2='+str(round_sig(T,sig=3))+' T_{gate}$ \n $\\bar{\\theta}=$' + str(round(getMEAN_circular_iterative(d_l), 3)) +
        ',\n $\sigma=$' + str(round(getSTD(d_l, 0, numeric=True)[0], 3)): vals_l,
        '\n physical ancilla, \n $T_2='+str(round_sig(T,sig=3))+' T_{gate}$ \n $\\bar{\\theta}=$' + str(round(getMEAN_circular_iterative(d_t), 3)) +
        ',\n $\sigma=$' + str(round(getSTD(d_t, 0, numeric=True)[0], 3)): vals_t,
    })
    # get values in the same order as keys, and parse percentage values

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
    path = os.getcwd()
    filename = os.path.join(path,'images\\for_video\\both_'+str(T)+'.jpg')
    plt.savefig(filename)
    plt.show()

def load_IPEA_new(rotation_type,noise_type,N=None):
    path = os.getcwd()
    if perfect_syndrome_extraction:
        if N is None:
            folder = os.path.join(path, 'data\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\'+rotation_type+'\\'+noise_type+'\\' + str(angle))
        else:
            folder = os.path.join(path,
                                  'data\\IPEA\\PerfectSyndromeExtraction\\' + name + '\\' + rotation_type + '\\' + noise_type + '\\' + str(
                                      angle)+'\\'+str(N))
    else:
        if N is None:
            folder = os.path.join(path, 'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\'+rotation_type+'\\'+noise_type+'\\' + str(angle))
        else:
            folder = os.path.join(path,
                                  'data\\IPEA\\NoisySyndromeExtraction\\' + name + '\\' + rotation_type + '\\' + noise_type + '\\' + str(
                                      angle)+'\\'+str(N))

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

def plot_IPEA_MEAN_single_fig(precisions, x_points='single', save=True, xstart=None,yend=None,ystart=0,xend=1,x_p=None):
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

    color = {5:'red',7:'blue',9:'green'}
    for i in range(len(precisions)):
        precision = precisions[i]
        c = color[precision]

        j = precisions[i] - 1
        trad_data = traditional[j, :]
        ideal_data = ideal[j, :]
        logical_data = logical[j, :]

        # lost info
        li = 1-(logical_STDs_noLI[j,:]/logical_STDs[j,:])**2

        # fill between
        logic = np.abs(logical_data - ideal_data)
        interp_l = interp1d(x, logic,kind='cubic')
        trad = np.abs(trad_data - ideal_data)
        interp_t = interp1d(x, trad,kind='cubic')
        precs = np.ones_like(x) * 2 ** (-precision)
        interp_precs = interp1d(x, precs)
        interp_li = interp1d(x,li)
        new_x = np.linspace(0.9825, 0.9999, 10000)
        new_logic = interp_l(new_x)
        new_trad = interp_t(new_x)
        new_precs = interp_precs(new_x)
        new_li = interp_li(new_x)

        axes.plot(new_x, np.abs(new_trad), '-.', color=c)
        axes.plot(new_x, np.abs(new_logic), '-', color=c)
        axes.plot(x,np.ones_like(x)*2**(-precision),'--',color=c)

        intersect_index = np.argmin(np.abs(new_logic-new_trad))
        plt.scatter(new_x[intersect_index],new_logic[intersect_index], color=c,label='$l_i=$'+ str(round_sig(np.real(new_li[intersect_index])*100,sig=4)) + '%')

        logic_fill_temp = new_logic[new_logic<=new_trad]
        trad_fill_temp = new_trad[new_logic<=new_trad]
        x_fill_temp = new_x[new_logic<=new_trad]
        precs_fill_temp = new_precs[new_logic<=new_trad]

        logic_fill = logic_fill_temp[trad_fill_temp>precs_fill_temp]
        trad_fill = trad_fill_temp[trad_fill_temp>precs_fill_temp]
        x_fill = x_fill_temp[trad_fill_temp>precs_fill_temp]
        plt.fill_between(x_fill, logic_fill, trad_fill, facecolor=c, edgecolor="none", alpha=.3)

        lbl = '$2^{-m}=$' + str(2 ** (-precision))
        axes.plot(np.NaN, np.NaN, '--', label=lbl, color=c)

    axes.set_ylabel('$|\\bar{\\theta} - \\bar{\\theta}_{ideal}|$', fontsize = 13)
    axes.plot(np.NaN, np.NaN, '-', color='black', label='logical ancilla LPS')
    axes.plot(np.NaN, np.NaN, '-.', color='black', label='physical ancilla')

    for label in (axes.get_xticklabels()):
        label.set_fontsize(13)
    if xstart is not None:
        axes.set_xlim((xstart, None))
    if yend is not None:
        axes.set_ylim((None, yend))
        # axes.set_yscale('log')
    # axes.set_ylim((0, 0.1))
    axes.set_xlim((0.9825,1.001))
    # axes.set_yscale('log')
    # axes.set_xscale('log')
    axes.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if x_p is not None:
        axes.axvline(x=x_p,color='black')
    else:
        x_p = ''
    if save:
        path = os.getcwd()
        filename = os.path.join(path, 'images\\for_video\\IPEA_MEAN_linear_' + str(x_p) + '.jpg')
        filename = os.path.join(path, 'images\\IPEA_MEAN_linear_' + str(precisions) + '.jpg')
        plt.savefig(filename)
    plt.show()
    plt.close()

def plot_IPEA_single_fig(precisions, withLI=False, x_points='single', save=True,xstart=None,yend=None,x_p=None):
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
    elif x_points == 'two':
        x = f_worst_2q_IPEA
        plt.xlabel('Worst-Case Entangling Gate Fidelity', fontsize=15)
    else:
        x = T_list_for_IPEA
        plt.xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
        plt.xscale('log')

    print(x.shape)

    color = iter(['red', 'blue', 'green'])
    for i in range(len(precisions)):
        precision = precisions[i]
        c = next(color)

        j = precisions[i] - 1
        trad_data = traditional[j, :]
        ideal_data = ideal[j, :]
        logical_data = logical[j, :]

        # lost info
        li = 1 - (logical_STDs_noLI[j, :] / logical_STDs[j, :]) ** 2

        if withLI:
            # fill between
            logic = np.abs(logical_data - ideal_data)
            interp_l = interp1d(x, logic)
            trad = np.abs(trad_data - ideal_data)
            interp_t = interp1d(x, trad)
            new_x = np.linspace(0.997, 0.99999, 10000)
            new_logic = interp_l(new_x)
            new_trad = interp_t(new_x)
            interp_li = interp1d(x, li)
            new_li = interp_li(new_x)

            intersect_index = np.argmin(np.abs(new_logic - new_trad))
            lbl = '$m=$' + str(precision)

            plt.scatter(new_x[intersect_index], new_logic[intersect_index], color=c, label='$l_i=$' + str(
                round(np.real(new_li[intersect_index]) * 100, 2)) + '%')
            axes.plot(np.NaN, np.NaN, label=lbl, color=c)
        else:
            # fill between
            logic = np.abs(logical_data - ideal_data)
            interp_l = interp1d(x, logic)
            trad = np.abs(trad_data - ideal_data)
            interp_t = interp1d(x, trad)
            new_x = np.linspace(0.98, 0.99999, 10000)
            new_logic = interp_l(new_x)
            new_trad = interp_t(new_x)
            interp_li = interp1d(x, li)
            new_li = interp_li(new_x)

            intersect_index = np.argmin(np.abs(new_logic - new_trad))
            lbl = '$m=$' + str(precision)
            plt.scatter(new_x[intersect_index], new_logic[intersect_index], color=c, label='$l_i=$' + str(
                round(np.real(new_li[intersect_index]) * 100, 2)) + '%')
            axes.plot(np.NaN, np.NaN, label=lbl, color=c)

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
        axes.set_ylabel('$\\frac{\sigma}{\sqrt{1-l_i}} - \sigma_{ideal}$', fontsize=13)
    else:
        axes.set_ylabel('$\sigma - \sigma_{ideal}$', fontsize=13)
    axes.plot(np.NaN, np.NaN, '-', color='black', label='logical ancilla LPS')
    axes.plot(np.NaN, np.NaN, '-.', color='black', label='physical ancilla')
    for label in (axes.get_xticklabels()):
        if withLI:
            label.set_fontsize(13)
        else:
            label.set_fontsize(11)
    if xstart is not None:
        axes.set_xlim((xstart, None))
    if yend is not None:
        axes.set_ylim((-0.01, yend))
    # axes.set_yscale('log')
    axes.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if x_p is not None:
        axes.axvline(x=x_p,color='black')
    else:
        x_p = ''
    if save:
        path = os.getcwd()
        if withLI:
            filename = os.path.join(path, 'images\\for_video\\IPEA_STD_withLI_' + str(x_p) + '.jpg')
        else:
            filename = os.path.join(path, 'images\\for_video\\IPEA_STD_noLI_' + str(x_p) + '.jpg')
        plt.savefig(filename)
    # plt.show()
    plt.close()

def plot_estimated_number_of_trials_old(precisions, x_points='single', save=True,x_p=None):
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
        j = precision-1
        li = 1 - (logical_STDs_noLI[j, :] / logical_STDs[j, :]) ** 2
        # fill between
        if precision>5:
            interp_l = interp1d(x, logic_difference,kind='cubic')
            interp_t = interp1d(x, trad_difference,kind='cubic')
            new_x = np.linspace(0.997, 0.9999, 10000)
            new_logic = interp_l(new_x)
            new_trad = interp_t(new_x)
            interp_li = interp1d(x, li)
            new_li = interp_li(new_x)
            intersect_index = np.argmin(np.abs(new_logic - new_trad))
            plt.scatter(new_x[intersect_index], new_logic[intersect_index], color=c, label='$m=$' + str(precision)
                                                    + ", $N_{ideal}=$" + str(int(np.ceil(N_i[0])))+', $l_i=$'
                                                    + str(round(np.real(new_li[intersect_index]) * 100, 2)) + '%')
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
    axes.legend(loc='upper left')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if x_p is not None:
        axes.axvline(x=x_p,color='black')
    else:
        x_p = ''
    if save:
        path = os.getcwd()
        filename = os.path.join(path, 'images\\for_video\\estimated_number_of_trials_' + str(x_p) + '.jpg')
        plt.savefig(filename)
    # plt.show()
    plt.close()

def save_pictures_for_video(angle,precision,U='Rz',decay='T2',new_data=True,folder_to_load=None,create_data_quantum=False, create_histo=False,
                           save_pictures_histograms=False, save_pictures_means=False):

    T_list_histo = list(np.geomspace(1, 1e3 + 100, 100, endpoint=True))
    f_worst_1q_histo = map_decohere_to_worst_gate_fidelity(T_list_histo, 1, Tgate=1, decohere="2", save=False)
    interp = interp1d(f_worst_1q_histo, T_list_histo)
    # x = np.linspace(0.83,0.99976,100)
    x = np.linspace(0.976, 0.99976, 100)
    T_list_histo = interp(x)
    # T_list_histo = np.linspace(2,150,149,endpoint=True)

    xs = np.linspace(0.976,0.99976,100)

    T_list_histo = map_decohere_to_worst_gate_fidelity(T_list_histo, 1, Tgate=1, decohere="2", save=False)

    j = 0
    for T2 in T_list_histo:
        j += 1
        print('T2=',T2)
        print(j)
        print()

        if create_data_quantum:
            create_data(4, 'traditional', T2, 1 / 3)
            create_data(4, 'logical', T2, 1 / 3)
            create_data(4, 'ideal', T2, 1 / 3)

        if create_histo:
            d_l, li_l = createHistogram_N(1, T2, angle, precision, 'logical', U=U, decay=decay, new_data=new_data)
            distribution_with_vals = {}
            for k in d_l.keys():
                distribution_with_vals[bin2num(k) / 2 ** len(k)] = d_l[k]
            d_l = distribution_with_vals
            d_t, li_t = createHistogram_N(1, T2, angle, precision, 'traditional', U=U, decay=decay, new_data=new_data)
            distribution_with_vals = {}
            for k in d_t.keys():
                distribution_with_vals[bin2num(k) / 2 ** len(k)] = d_t[k]
            d_t = distribution_with_vals
            d_i, li_i = createHistogram_N(1, T2, angle, precision, 'ideal', U=U, decay=decay,new_data=new_data)
            distribution_with_vals = {}
            for k in d_i.keys():
                distribution_with_vals[bin2num(k) / 2 ** len(k)] = d_i[k]
            d_i = distribution_with_vals
        else:
            d_t, li_t = loadHistogram_N(1,T2,angle,precision,'traditional',U=U,decay=decay,new_data=new_data,folder_to_save=folder_to_load)
            d_l, li_l = loadHistogram_N(1,T2,angle,precision,'logical',U=U,decay=decay,new_data=new_data,folder_to_save=folder_to_load)
            d_i, li_i = loadHistogram_N(1, T2, angle, precision, 'ideal', U=U, decay=decay,new_data=new_data)

        if save_pictures_histograms:
            # save pictures histogram
            plot_histogram_wrapped_noise_single_algorithm(d_i, d_t, T2, algorithm='traditional')
            plot_histogram_wrapped_noise_single_algorithm(d_i, d_l, T2, algorithm='logical')
            plot_histogram_wrapped_noise_both_algorithms(d_i, d_t, d_l, T2)

    if save_pictures_means:
        for x in xs:
            # save pictures
            plot_IPEA_single_fig([3, 6, 9], withLI=True, x_points='single', save=True, xstart=0.997, yend=0.15,x_p=x)
            plot_IPEA_single_fig([3, 5, 9], withLI=False, x_points='single', save=True,x_p=x)
            plot_IPEA_MEAN_single_fig([5, 6, 9], x_points='single', save=True, xstart=0.9825, yend=0.05, ystart=0, xend=1,x_p=x)
            plot_estimated_number_of_trials([3, 5, 6, 7, 8, 9], x_points='single', save=True,x_p=x)

def make_videos_histograms(T_list=None,name='',fps=10,rev=-1):
    if T_list is None:
        T_list_histo = list(np.geomspace(1, 1e3 + 100, 100, endpoint=True))
        f_worst_1q_histo = map_decohere_to_worst_gate_fidelity(T_list_histo, 1, Tgate=1, decohere="2", save=False)
        interp = interp1d(f_worst_1q_histo, T_list_histo)
        x = np.linspace(0.83,0.99976,100)
        T_list_histo = interp(x)
        T_list_histo = list(np.linspace(2, 150, 149, endpoint=True))+list(interp(x))
        T_list_histo.sort()
        T_list = T_list_histo

    frames_traditional = []
    frames_logical = []
    frames_both = []
    for T in T_list[::rev]:
        image = imageio.imread(f'./images/for_video/traditional_{T}.jpg')
        frames_traditional.append(image)
        image = imageio.imread(f'./images/for_video/logical_{T}.jpg')
        frames_logical.append(image)
        image = imageio.imread(f'./images/for_video/both_{T}.jpg')
        frames_both.append(image)

    if rev == -1:
        imageio.mimsave('./images/videos/traditional'+name+'.gif',  # output gif
                        frames_traditional,  # array of input frames
                        fps=fps)  # optional: frames per second
        imageio.mimsave('./images/videos/logical'+name+'.gif',  # output gif
                        frames_logical,  # array of input frames
                        fps=fps)  # optional: frames per second
        imageio.mimsave('./images/videos/both'+name+'.gif',  # output gif
                        frames_both,  # array of input frames
                        fps=fps)  # optional: frames per second
    else:
        imageio.mimsave('./images/videos/traditional' + name + 'reversed.gif',  # output gif
                        frames_traditional,  # array of input frames
                        fps=fps)  # optional: frames per second
        imageio.mimsave('./images/videos/logical' + name + 'reversed.gif',  # output gif
                        frames_logical,  # array of input frames
                        fps=fps)  # optional: frames per second
        imageio.mimsave('./images/videos/both' + name + 'reversed.gif',  # output gif
                        frames_both,  # array of input frames
                        fps=fps)  # optional: frames per second


def make_video_plots(save_pictures = False, T_list = None,fps=10,name='',rev=-1):
    if T_list is None:
        T_list_histo = list(np.geomspace(1, 1e3 + 100, 100, endpoint=True))
        f_worst_1q_histo = map_decohere_to_worst_gate_fidelity(T_list_histo, 1, Tgate=1, decohere="2", save=False)
        interp = interp1d(f_worst_1q_histo, T_list_histo)
        x = np.linspace(0.83,0.99976,100)
        T_list_histo = interp(x)
        T_list_histo = list(np.linspace(2, 150, 149, endpoint=True))+list(interp(x))
        T_list_histo.sort()
        T_list = T_list_histo
    xs = map_decohere_to_worst_gate_fidelity(T_list, 1, Tgate=1, decohere="2", save=False)
    if save_pictures:
        for x in xs:
            # save pictures
            # plot_IPEA_single_fig([5, 6, 9], withLI=True, x_points='single', save=True, xstart=0.997, yend=0.15,x_p=x)
            # plot_IPEA_single_fig([5, 6, 9], withLI=False, x_points='single', save=True,x_p=x, xstart=0.980)
            # plot_IPEA_MEAN_single_fig([5, 6, 9], x_points='single', save=True, xstart=0.9825, yend=0.05, ystart=0, xend=1,x_p=x)
            plot_estimated_number_of_trials([3, 5, 6, 7, 8, 9], x_points='single', save=True,x_p=x)
    frames_number = []
    frames_STD_withLI = []
    frames_STD_noLI = []
    frames_MEAN = []
    for x_p in xs[::rev]:
        image = imageio.imread(f'./images/for_video/estimated_number_of_trials_' + str(x_p) + '.jpg')
        frames_number.append(image)
        image = imageio.imread(f'./images/for_video/IPEA_STD_noLI_' + str(x_p) + '.jpg')
        frames_STD_noLI.append(image)
        image = imageio.imread(f'./images/for_video/IPEA_STD_withLI_' + str(x_p) + '.jpg')
        frames_STD_withLI.append(image)
        image = imageio.imread(f'./images/for_video/IPEA_MEAN_linear_' + str(x_p) + '.jpg')
        frames_MEAN.append(image)

    if rev == -1:
        imageio.mimsave('./images/videos/estimated_number_of_trials'+name+'.gif',  # output gif
                        frames_number,  # array of input frames
                        fps=fps)  # optional: frames per second
        imageio.mimsave('./images/videos/IPEA_STD_noLI'+name+'.gif',  # output gif
                        frames_STD_noLI,  # array of input frames
                        fps=fps)  # optional: frames per second
        imageio.mimsave('./images/videos/IPEA_STD_withLI'+name+'.gif',  # output gif
                        frames_STD_withLI,  # array of input frames
                        fps=fps)  # optional: frames per second
        imageio.mimsave('./images/videos/IPEA_MEAN_linear'+name+'.gif',  # output gif
                        frames_MEAN,  # array of input frames
                        fps=fps)  # optional: frames per second
    else:
        imageio.mimsave('./images/videos/estimated_number_of_trials' + name + 'reversed.gif',  # output gif
                        frames_number,  # array of input frames
                        fps=fps)  # optional: frames per second
        imageio.mimsave('./images/videos/IPEA_STD_noLI' + name + 'reversed.gif',  # output gif
                        frames_STD_noLI,  # array of input frames
                        fps=fps)  # optional: frames per second
        imageio.mimsave('./images/videos/IPEA_STD_withLI' + name + 'reversed.gif',  # output gif
                        frames_STD_withLI,  # array of input frames
                        fps=fps)  # optional: frames per second
        imageio.mimsave('./images/videos/IPEA_MEAN_linear' + name + 'reversed.gif',  # output gif
                        frames_MEAN,  # array of input frames
                        fps=fps)  # optional: frames per second

def plot_estimated_number_of_trials(precisions, x_points='single', save=True,x_p=None):
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
    # axes.plot(np.NaN, np.NaN, '-', color='black', label='logical ancilla LPS')
    # axes.plot(np.NaN, np.NaN, '-.', color='black', label='physical ancilla')
    axes.plot(x, np.ones_like(x), '--', color='black')
    axes.text(0.9980, 5, '$N>2^{2m}\\frac{\sigma^2}{1-l_i}$', style='italic', fontsize=15)
    color = iter(cm.rainbow(np.linspace(0, 1, len(precisions))))
    for i in range(len(precisions)):
        precision = precisions[i]
        c = next(color)

        N_l = np.real(2**(2*precision) * (np.array(logical)[precision-1,:])**2)
        N_t = np.real(2**(2*precision) * (np.array(traditional)[precision-1,:])**2)
        N_i = np.real(2**(2*precision) * (np.array(ideal)[precision-1,:])**2)
        logic_difference = N_t/N_l
        axes.plot(x, logic_difference,'-', c=c, label='m='+str(precision))
        j = precision-1
        li = 1 - (logical_STDs_noLI[j, :] / logical_STDs[j, :]) ** 2

        # axes.plot(x, N_l,'--', c=c)
        # axes.plot(x, N_t,'-.', c=c)
        # axes.plot(x, N_i,'-', c=c)

    # axes.plot(np.NaN, np.NaN, '-', color='black', label='ideal (no decoherence)')
    # axes.set_ylabel('Estimated Minimal Number of Trials', fontsize=13)
    axes.set_ylabel('$N_{physical}/N_{logical}$', fontsize=13)
    for label in (axes.get_xticklabels()):
        label.set_fontsize(13)
    axes.set_xlim((0.9968, 1))
    axes.set_ylim((-0.5, 7))
    # axes.set_yscale('log')
    # axes.set_title('$N>2^{2m}\\frac{\sigma^2}{1-l_i}$', fontsize = 13)
    axes.legend(loc='upper left')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if x_p is not None:
        axes.axvline(x=x_p,color='black')
    else:
        x_p = ''
    if save:
        path = os.getcwd()
        filename = os.path.join(path, 'images\\for_video\\estimated_number_of_trials_' + str(x_p) + '.jpg')
        plt.savefig(filename)
    # plt.show()
    plt.close()



########### load data for plots #############

rotation_type = 'Rx'
noise_type = 'T2'
perfect_syndrome_extraction = False
name = 'recalc_paper'
precision = 9
T_list = None
angle = 1 / np.sqrt(3)

ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA, logical_MEANs, traditional_MEANs, ideal_MEANs = load_IPEA_new(
    rotation_type, noise_type, N=None)

########### create interpretation function #############

T_list_histo = list(np.geomspace(1, 1e3 + 100, 100, endpoint=True))
f_worst_1q_histo = map_decohere_to_worst_gate_fidelity(T_list_histo, 1, Tgate=1, decohere="2", save=False)
interp = interp1d(f_worst_1q_histo, T_list_histo)

########### define relevant time lists #############

x = np.linspace(0.976, 0.99976, 100) #fidelity
T_list_1 = interp(x)

x = np.linspace(0.997, 0.99976, 100)
T_list_4 = interp(x)

x = np.linspace(0.83,0.99976,100)
T_list_3 = interp(x)

T_list_2 = np.linspace(2,135,134,endpoint=True)

########### define relevant time lists for video - uncomment to change #############

# Ts = list(T_list_3) + list(T_list_2)
# Ts.sort()
# name = '_good_histo'
#
# Ts = T_list_1
# name = '_good_noLI'
#
# Ts = T_list_4
# name = '_good_withLI'

# Ts = list(T_list_3) + list(T_list_2) + list(T_list_1) + list(T_list_4)
# Ts.sort()
# name = '_good_all'

########### create videos #############

# make_videos_histograms(T_list=Ts,name=name,fps=10,rev=1)
# make_video_plots(save_pictures = True, T_list = Ts,fps=10,name=name)

# create video of ideal distribution

# check plots
plot_IPEA_MEAN_single_fig([5,7,9],x_points='single',save=True,xstart=0.98,yend=0.05,ystart=0,xend=1)
# plot_estimated_number_of_trials([3,5,6,7,8,9],x_points='single',save=True)
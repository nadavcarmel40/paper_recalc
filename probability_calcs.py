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
from scipy.signal import convolve2d
from matplotlib.colors import DivergingNorm
import scipy.interpolate as interpolate
import matplotlib.colors as colors

print('all packages imported')


#########################################################################################
########################## constants to control this file ###############################
#########################################################################################

# plot related variables
rotation_type = 'Rz'
noise_type = 'T1'
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

def getSTD(d,li):
    sum_x = 0
    sum_y = 0
    for key in d.keys():
        theta = bin2num(key)/2**len(key)
        p = d[key]
        sum_x += p*np.cos(theta)
        sum_y -= p*np.sin(theta)
    theta_avg = np.arctan(sum_y/sum_x)
    R = np.sqrt(sum_x ** 2 + sum_y ** 2)
    std = np.sqrt(-2*np.log(R))
    return std/np.sqrt(1-li), std

def getMEAN_circular(d):
    sum_x = 0
    sum_y = 0
    for key in d.keys():
        theta = 2*np.pi*bin2num(key)/2**len(key)
        p = d[key]
        sum_x += p*np.cos(theta)
        sum_y -= p*np.sin(theta)
    theta_avg = np.arctan(sum_y/sum_x)
    return theta_avg/np.pi

def getMEAN(d):
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
    Ns = [1,3,5,7,9,11,15,21,29,39,51,65,79,99]
    for N in Ns:
        save_all_precisions(angle, precision, U=rotation_type, decay=noise_type, T_list=T_list, N=N)


if generate_MEANs_for_2021:
    save_all_precisions(angle,precision,U=rotation_type,decay=noise_type,T_list=None,new_data=True)

if show_small_histograms:
    print("The binary representation of angle to 4 binary digits is:", num2bin(1 / np.sqrt(3), 4))
    print("The binary representation of angle to 4 binary digits is:", num2bin(1 / np.sqrt(3) + 2 ** (-4), 4))

    create_data_func = create_data_chooser(rotation_type, noise_type)


    def set_ticks_size(ax, x, y, sparse_x=False,sparse_y=False):
        # Set tick font size
        for ind,label in enumerate(ax.get_xticklabels()):
            label.set_fontsize(x)
            if sparse_x:
                if ind%2==0:
                    label.set_visible(False)
        # Set tick font size
        for ind,label in enumerate(ax.get_yticklabels()):
            label.set_fontsize(y)
            if sparse_y:
                if ind % 2 == 0:
                    label.set_visible(False)



    def create_data(precision, type, T, angle):
        print('starting to create data for algorithm')
        start = time.time()
        create_data_func(T, angle, precision, type)
        print('ended in ' + str(time.time() - start) + ' seconds')
    # create_data(4,'traditional',1,1/3)
    # create_data(4,'traditional',5,1/3)
    # create_data(4,'ideal',1,1/3)

    def plot_histogram_wrapped(N,precision, T, angle):
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
        df = pandas.DataFrame({
            'Factor': keys_i,
            'Ideal \n $\\bar{\\theta}=$' + str(round(getMEAN(d_i), 3)) +
            '\n $\sigma=$' + str(round(getSTD(d_i, 0)[0], 2)): vals_i,
            'Logical \n $\\bar{\\theta}=$' + str(round(getMEAN(d_l), 3)) +
            ',\n $\sigma=$' + str(round(getSTD(d_l, 0)[0], 3)): vals_l,
            'Physical \n $\\bar{\\theta}=$' + str(round(getMEAN(d_t), 3)) +
            '\n $\sigma=$' + str(round(getSTD(d_t, 0)[0], 3)): vals_t,
        })
        # get values in the same order as keys, and parse percentage values

        fig, ax1 = plt.subplots(figsize=(20, 10))
        tidy = df.melt(id_vars='Factor').rename(columns=str.title)
        sns.barplot(x='Factor', y='Value', hue='Variable', data=tidy, ax=ax1)
        ax1.set_xlabel('Experiment Result', fontsize=40)
        ax1.set_ylabel('Probability', fontsize=40)
        plt.legend(fontsize=40, loc='upper left')
        sns.despine(fig)
        ax1.set_title('$T_2=40 T_{gate}$,   $\\theta=1/3$                                         ', fontsize=40)
        plt.tight_layout(rect=[0.05, 0.03, 1.1, 0.95])
        set_ticks_size(ax1, 35, 35,sparse_x=True)
        plt.savefig('images\\latest_histogram_'+str(N)+'.jpg')
        plt.show()

    def plot_histogram_wrapped_ideal(precision, T, angle):
        print('starting to create histogram for logical algorithm')
        d_1, li = createHistogram_N(1,T, angle, precision, 'ideal', U=rotation_type, decay=noise_type)
        d_3, li = createHistogram_N(3,T, angle, precision, 'ideal', U=rotation_type, decay=noise_type)
        d_5, li = createHistogram_N(5,T, angle, precision, 'ideal', U=rotation_type, decay=noise_type)

        keys_1 = np.array(list(d_1.keys()))
        vals_1 = [d_1[k] for k in keys_1]
        keys_3 = np.array(list(d_3.keys()))
        vals_3 = [d_3[k] for k in keys_3]
        keys_5 = np.array(list(d_5.keys()))
        vals_5 = [d_5[k] for k in keys_5]
        keys_1 = np.array([bin2num(k) / 2 ** len(k) for k in d_1.keys()])
        print('lost information is: ' + str(li))
        # keys_1[::2] = 0
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
        ax1.set_xlabel('Experiment Result', fontsize=40)
        ax1.set_ylabel('Probability', fontsize=40)
        plt.legend(fontsize=35, loc='upper left')
        sns.despine(fig)
        ax1.set_title('ideal results, $\\theta=1/3$                                            ', fontsize=40)
        plt.tight_layout(rect=[0.05, 0.03, 1.1, 0.95])
        set_ticks_size(ax1, 35, 35,sparse_x=True)
        plt.savefig('images\\compare_ideal.jpg')
        plt.show()

    def plot_histogram_wrapped_noise(N,precision,angle):
        print('starting to create histogram for logical algorithm')
        d_t, li = createHistogram_N(N,1, angle, precision, 'traditional', U=rotation_type, decay=noise_type)
        d_l, li = createHistogram_N(N,5, angle, precision, 'traditional', U=rotation_type, decay=noise_type)
        d_i, li = createHistogram_N(N,1, angle, precision, 'ideal', U=rotation_type, decay=noise_type)

        keys_l = list(d_l.keys())
        vals_l = [d_l[k] for k in keys_l]
        keys_t = list(d_t.keys())
        vals_t = [d_t[k] for k in keys_t]
        keys_i = list(d_i.keys())
        vals_i = [d_i[k] for k in keys_i]
        keys_i = [bin2num(k) / 2 ** len(k) for k in d_i.keys()]
        print('lost information is: ' + str(li))
        df = pandas.DataFrame({
            'Factor': keys_i,
            '\n $T_2=\infty$ \n $\\bar{\\theta}=$' + str(round(getMEAN(d_i), 3)) +
            ',\n $\sigma=$' + str(round(getSTD(d_i, 0)[0], 2)): vals_i,
            '\n $T_2=30 T_{gate}$ \n $\\bar{\\theta}=$' + str(round(getMEAN(d_l), 3)) +
            ',\n $\sigma=$' + str(round(getSTD(d_l, 0)[0], 3)): vals_l,
            '\n $T_2=10 T_{gate}$ \n $\\bar{\\theta}=$' + str(round(getMEAN(d_t), 3)) +
            ',\n $\sigma=$' + str(round(getSTD(d_t, 0)[0], 3)): vals_t,
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
        plt.savefig('images\\histogram_noise_low_'+str(N)+'.jpg')
        plt.show()

    def plot_series_ideal_precisions(max_precision,angle,create = False):
        if create:
            create_data(max_precision, 'ideal', 1, 1 / 3)

        def plotter(precision):
            d_i, li = createHistogram_N(1,1, angle, precision, 'ideal', U=rotation_type, decay=noise_type)

            keys_i = list(d_i.keys())
            vals_i = [d_i[k] for k in keys_i]
            keys_i = [bin2num(k) / 2 ** len(k) for k in d_i.keys()]
            print('lost information is: ' + str(li))
            df = pandas.DataFrame({
                'Factor': keys_i,
                '\n $T_2=\infty$ \n $\\bar{\\theta}=$' + str(round(getMEAN_circular_iterative(d_i,numeric=False), 3)) +
                ',\n $\sigma=$' + str(round(getSTD(d_i, 0)[0], 2)): vals_i,
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
            if precision > 4:
                ax1.axes.xaxis.set_ticklabels([])
            plt.savefig('images\\histogram_ideal_'+str(precision)+'.jpg')
            plt.show()

        for precision in range(1,max_precision+1):
            plotter(precision)


    # plot_series_ideal_precisions(13, 1/3, create=False)
    plot_histogram_wrapped(1,4, 40, 1 / 3)
    # plot_histogram_wrapped(3,4, 40, 1 / 3)
    # plot_histogram_wrapped(5,4, 40, 1 / 3)
    plot_histogram_wrapped_ideal(4, 40, 1 / 3)
    # plot_histogram_wrapped_noise(1, 4, 1 / 3)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


#########################################################################################
########### load data from folder for data generated now  ###############################
#########################################################################################

def load_IPEA_new(rotation_type,noise_type,N=1):
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

    # logical_STDs = np.load(os.path.join(folder, 'logical_STDs.npy'))
    logical_STDs = None
    # logical_STDs_noLI = np.load(os.path.join(folder, 'logical_STDs_noLI.npy'))
    logical_STDs_noLI = None
    # traditional_STDs = np.load(os.path.join(folder, 'traditional_STDs.npy'))
    traditional_STDs = None
    traditional_STDs_noLI = np.load(os.path.join(folder, 'traditional_STDs_noLI.npy'))
    ideal_STDs = np.load(os.path.join(folder, 'ideal_STDs.npy'))
    ideal_STDs_noLI = np.load(os.path.join(folder, 'ideal_STDs_noLI.npy'))
    logical_MEANs = np.load(os.path.join(folder, 'logical_MEANs.npy'))
    traditional_MEANs = np.load(os.path.join(folder, 'traditional_MEANs.npy'))
    ideal_MEANs = np.load(os.path.join(folder, 'ideal_MEANs.npy'))
    f_worst_1q_IPEA = np.load(os.path.join(folder, 'f_worst_1q.npy'))
    f_worst_2q_IPEA = np.load(os.path.join(folder, 'f_worst_2q.npy'))
    T_list_for_IPEA = np.load(os.path.join(folder, 'T_list.npy'))
    # P_i = np.load(os.path.join(folder, 'P_i.npy'))
    P_i = None
    # P_l = np.load(os.path.join(folder, 'P_l.npy'))
    P_l = None
    # P_t = np.load(os.path.join(folder, 'P_t.npy'))
    P_t = None

    return ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI,\
           logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA,\
           logical_MEANs,traditional_MEANs,ideal_MEANs,P_i,P_l,P_t

if plot_data_new:

    ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI,\
    f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA,logical_MEANs,traditional_MEANs,ideal_MEANs,P_i,P_l,P_t = load_IPEA_new(
        rotation_type, noise_type,N=None)


#########################################################################################
########### load data from folder for data generated in 2021  ###########################
#########################################################################################

def load_IPEA(rotation_type,noise_type,file=None):
    if file is None:
        file = os.path.join('data_2021\\IPEA_Fisher', 'IPEASTDs' + noise_type + '_' + rotation_type + '_all')

    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data_2021\\IPEA_Fisher\\' + rotation_type+'\\'+noise_type+'\\' + str(angle))
    else:
        folder = os.path.join(path, 'data_2021\\IPEA_Fisher\\' + rotation_type+'\\'+noise_type+'\\' + str(angle))

    # logical_MEANs = np.load(os.path.join(folder, 'logical_MEANs.npy'))
    logical_MEANs = None
    # traditional_MEANs = np.load(os.path.join(folder, 'traditional_MEANs.npy'))
    traditional_MEANs = None
    # ideal_MEANs = np.load(os.path.join(folder, 'ideal_MEANs.npy'))
    ideal_MEANs = None

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
    # logical_MEANs = np.load(os.path.join(folder, 'logical_MEANs.npy'))
    logical_MEANs = None
    # traditional_MEANs = np.load(os.path.join(folder, 'traditional_MEANs.npy'))
    traditional_MEANs = None
    # ideal_MEANs = np.load(os.path.join(folder, 'ideal_MEANs.npy'))
    ideal_MEANs = None
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

    file = 'data_2021\\IPEA_Fisher\\IPEASTDsT2_Rx.mat'

    ideal_STDs, traditional_STDs, logical_STDs, ideal_STDs_noLI, traditional_STDs_noLI, logical_STDs_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA,logical_MEANs,traditional_MEANs,ideal_MEANs = load_IPEA(
        rotation_type, noise_type,file=None)

    # f_worst_1q_IPEA,f_worst_2q,f_worst_1q_T1,f_worst_2q_T1,_,_ = load_WCGF()

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


def plot_IPEA_single_fig(precisions, withLI=False, x_points='single', save=True,xstart=None,yend=None,ax_return=None):

    if ax_return is not None:
        axes = ax_return
    else:
        fig, axes = plt.subplots()

    axes.locator_params(axis='x', nbins=6)
    axes.locator_params(axis='y', nbins=5)

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
        axes.set_xlabel('Single Gate Fidelity', fontsize=25)
    elif x_points == 'two':
        x = f_worst_2q_IPEA
        axes.set_xlabel('Worst-Case Entangling Gate Fidelity', fontsize=15)
    else:
        x = T_list_for_IPEA
        axes.set_xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
        axes.set_xscale('log')

    print(x.shape)

    for i in range(len(precisions)):

        precision = precisions[i]
        c = color(precision)
        axes.plot(np.NaN, np.NaN, label='m=' + str(precision), color=color(precision))

        j = precisions[i] - 1
        trad_data = traditional[j, :]
        ideal_data = ideal[j, :]
        logical_data = logical[j, :]

        # lost info
        li = 1 - (logical_STDs_noLI[j, :] / logical_STDs[j, :]) ** 2

        if True:
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

                # axes.scatter(new_x[intersect_index], new_logic[intersect_index], color=c,label='$l_i=$' + str(
                #     round(np.real(new_li[intersect_index]) * 100, 2)) + '%')
                axes.scatter(new_x[intersect_index], new_logic[intersect_index], color=c, s=200)
                # axes.plot(np.NaN, np.NaN, label=lbl, color=c)
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
                axes.scatter(new_x[intersect_index], new_logic[intersect_index], color=c,label='$l_i=$' + str(
                    round(np.real(new_li[intersect_index]) * 100, 2)) + '%', s=100)
                axes.plot(np.NaN, np.NaN, label=lbl, color=c)

        else:
            new_x=x
            new_logic = np.abs(logical_data - ideal_data)
            new_trad = np.abs(trad_data - ideal_data)

        axes.plot(new_x, new_trad, '-.', color=c,linewidth=3)
        axes.plot(new_x, new_logic, '-', color=c,linewidth=3)

    if withLI:
        axes.set_ylabel('$\\frac{\sigma}{\sqrt{1-l_i}} - \sigma_{ideal}$', fontsize = 25)
    else:
        axes.set_ylabel('$\sigma - \sigma_{ideal}$', fontsize=25)
    axes.plot(np.NaN, np.NaN, '-', color='black', label='logical ancilla LPS')
    axes.plot(np.NaN, np.NaN, '-.', color='black', label='physical ancilla')
    if xstart is not None:
        axes.set_xlim((xstart, None))
    if yend is not None:
        axes.set_ylim((-0.01, yend))
    # axes.set_yscale('log')
    # axes.legend(fontsize=18)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    set_ticks_size(axes,20,20)
    if save:
        if withLI:
            plt.savefig('images\\IPEA_STD_withLI'+rotation_type+noise_type)
        else:
            plt.savefig('images\\IPEA_STD_noLI'+rotation_type+noise_type)
    if ax_return is not None:
        return axes
    else:
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


def plot_IPEA_MEAN_single_fig(precisions, x_points='single', save=True, xstart=None,yend=None,ystart=0,xend=1,ax_return=None):

    if ax_return is not None:
        axes = ax_return
    else:
        fig, axes = plt.subplots()
    axes.locator_params(axis='x', nbins=5)
    axes.locator_params(axis='y', nbins=5)
    ideal = ideal_MEANs
    traditional = traditional_MEANs
    logical = logical_MEANs
    if x_points == 'single':
        x = f_worst_1q_IPEA
        axes.set_xlabel('Single Gate Fidelity', fontsize=25)
        # xstart = 0.9975
        # xstart = 0.9965
        # xend = 1
    elif x_points == 'two':
        x = f_worst_2q_IPEA
        axes.set_xlabel('Worst-Case Entangling Gate Fidelity', fontsize=15)
        # xstart = 0.995
        # xend = 1
    else:
        x = T_list_for_IPEA
        axes.set_xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
        axes.set_xscale('log')
        # xstart = 10
        # xend = 2000

    print(x.shape)

    for i in range(len(precisions)):
        precision = precisions[i]
        c = color(precision)

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

        axes.plot(new_x, np.abs(new_trad), '-.', color=c,linewidth=3)
        axes.plot(new_x, np.abs(new_logic), '-', color=c,linewidth=3)
        axes.plot(x,np.ones_like(x)*2**(-precision),'--',color=c,linewidth=3)

        intersect_index = np.argmin(np.abs(new_logic-new_trad))
        # axes.scatter(new_x[intersect_index],new_logic[intersect_index], color=c,label='$l_i=$'+ str(round_sig(np.real(new_li[intersect_index])*100,sig=4)) + '%')
        axes.scatter(new_x[intersect_index],new_logic[intersect_index], color=c,s=100)

        logic_fill_temp = new_logic[new_logic<=new_trad]
        trad_fill_temp = new_trad[new_logic<=new_trad]
        x_fill_temp = new_x[new_logic<=new_trad]
        precs_fill_temp = new_precs[new_logic<=new_trad]

        logic_fill = logic_fill_temp[trad_fill_temp>precs_fill_temp]
        trad_fill = trad_fill_temp[trad_fill_temp>precs_fill_temp]
        x_fill = x_fill_temp[trad_fill_temp>precs_fill_temp]
        axes.fill_between(x_fill, logic_fill, trad_fill, facecolor=c, edgecolor="none", alpha=.3)

        # lbl = '$2^{-m}=$' + str(2 ** (-precision))
        # axes.plot(np.NaN, np.NaN, '--', label=lbl, color=c)

    axes.set_ylabel('$|\\bar{\\theta} - \\bar{\\theta}_{ideal}|$', fontsize = 25)
    # axes.plot(np.NaN, np.NaN, '-', color='black', label='logical ancilla LPS')
    # axes.plot(np.NaN, np.NaN, '-.', color='black', label='physical ancilla')

    set_ticks_size(axes,20,20)
    if xstart is not None:
        axes.set_xlim((xstart, None))
    if yend is not None:
        axes.set_ylim((None, yend))
        # axes.set_yscale('log')
    axes.set_ylim((1e-3, 1e-1))
    axes.set_yscale('log')
    # axes.set_xscale('log')
    # axes.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig('images\\IPEA_MEAN_log_new.jpg')
    if ax_return is not None:
        return axes
    else:
        plt.show()

def color(precision):
    colors = list(iter(cm.brg(np.linspace(0, 1, 1000))))
    colors.reverse()

    return colors[int(precision/9*999)]

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

def plot_estimated_number_of_trials_new(precisions, x_points='single', save=True, ax_return = None):
    if ax_return is not None:
        axes = ax_return
    else:
        fig, axes = plt.subplots()
    ideal = ideal_STDs
    traditional = traditional_STDs
    logical = logical_STDs

    axes.locator_params(axis='x', nbins=3)
    axes.locator_params(axis='y', nbins=4)


    if x_points == 'single':
        x = f_worst_1q_IPEA
        axes.set_xlabel('Single Gate Fidelity', fontsize=15)
        # xstart = 0.9975
        xstart = 0.999965
        xend = 1
    elif x_points == 'two':
        x = f_worst_2q_IPEA
        axes.set_xlabel('Worst-Case Entangling Gate Fidelity', fontsize=15)
        xstart = 0.995
        xend = 1
    else:
        x = T_list_for_IPEA
        axes.set_xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
        axes.set_xscale('log')
        xstart = 10
        xend = 2000
    axes.plot(x, np.ones_like(x), '--', color='black')
    for i in range(len(precisions)):
        precision = precisions[i]
        c = color(precision)

        N_l = np.real(2**(2*precision) * (np.array(logical)[precision-1,:])**2)
        N_t = np.real(2**(2*precision) * (np.array(traditional)[precision-1,:])**2)
        logic_difference = 1/(N_l / N_t)
        # logic_difference = savitzky_golay(logic_difference,1,1)
        axes.plot(x, logic_difference,'-', c=c)
        axes.plot(np.NaN, np.NaN, label='m=' + str(precision),color=color(precision))

        # if precision>5:
        #     interp_l = interp1d(x, logic_difference,kind='cubic')
        #     new_x = np.linspace(0.997, 0.9999, 10000)
        #     new_logic = interp_l(new_x)
        #     intersect_index = np.argmin(np.abs(new_logic - 1))
        #     axes.scatter(new_x[intersect_index], new_logic[intersect_index], color=c,s=300)


    axes.set_ylabel('$N-N_{ideal}$', fontsize=15)
    axes.set_ylabel('$N_{physical}/N_{logical}$', fontsize=15)
    set_ticks_size(axes,15,15)
    axes.set_xlim((0.9968, 1))
    # axes.set_ylim((-0.5, 7))
    # axes.set_yscale('log')

    # axes.text(0.9971, 4, '$N>2^{2m}\\frac{\sigma^2}{1-l_i}$', style='italic', fontsize=33)
    # axes.legend(loc='upper left')
    # plt.tight_layout(rect=[0.0, 0.03, 0.95, 0.95])
    if save:
        plt.savefig('images\\IPEA_minimal_trials.jpg')
    if ax_return is not None:
        return axes
    else:
        plt.show()


def interpolate_2d_grid(X,Y,Z):
    x = []
    y = []
    z = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x.append(X[i, j])
            y.append(Y[i, j])
            z.append(Z[i, j])

    x_new = np.linspace(np.min(x), np.max(x), 1000)
    y_new = np.linspace(np.min(y), np.max(y), 1000)

    Ynew, Xnew = np.meshgrid(y_new, x_new)

    interp = interpolate.LinearNDInterpolator(list(zip(x, y)), z)
    znew = interp(Xnew, Ynew)

    window = 1
    kernel = np.ones((2*window+1,2*window+1))/(2*window+1)**2
    znew = convolve2d(znew,kernel,boundary='symm',mode='same')
    return Xnew,Ynew,znew

def plot_color(X,Y,Z,xlabel,ylabel,title):
    fig,ax = plt.subplots()
    norm = DivergingNorm(vmin=Z.min(), vcenter=0, vmax=Z.max())
    # norm = colors.LogNorm(vmin=0, vmax=0.05)
    lims = dict(cmap='RdBu')
    plt.pcolormesh(X, Y, Z, shading='flat', norm=norm, **lims)
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel(ylabel,fontsize=15)
    ax.set_title(title,fontsize=14)
    set_ticks_size(ax, 13, 12)
    ax.set_yscale('log')
    ax.set_ylim(1,100)
    plt.colorbar(label="Physical Control             Logical Control")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

def plot_error_probability_heatmap(Ns = (1,3,5,7,9,11,15,21,29,39,51,65,79,99),m=9,x_points='single',interpolate=True,save=False,successful_only=True):
    _, _, l_STD_withLI, _, _, l_STD_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA, _, _, _, P_i_const_N, P_l_const_N, P_t_const_N\
        = load_IPEA_new(rotation_type, noise_type, 1)

    if successful_only:
        saved_information_fraction = 1
    else:
        saved_information_fraction = (np.array(l_STD_noLI)[m - 1, :] / np.array(l_STD_withLI)[m - 1, :]) ** 2

    if x_points == 'two':
        fid = f_worst_2q_IPEA
        xlabel = 'Worst-Case Entangling Gate Fidelity'
    elif x_points == 'single':
        fid = f_worst_1q_IPEA
        xlabel = 'Worst-Case Single Gate Fidelity'

    # prepare data
    error_probability_difference = np.zeros((fid.shape[0],len(Ns)))
    Ns_logical_matrix = np.zeros((fid.shape[0],len(Ns)))
    Ns_traditional_matrix = np.zeros((fid.shape[0],len(Ns)))
    # build X and Y and Z_l and Z_t
    for ind,N in enumerate(Ns):
        _, _, l_STD_withLI, _, _, l_STD_noLI, _, _, _, _, _, _, P_i_const_N, P_l_const_N, P_t_const_N = load_IPEA_new(rotation_type, noise_type, N)

        data_logical = np.array(P_l_const_N)[m - 1, :]
        data_traditional = np.array(P_t_const_N)[m - 1, :]
        data_ideal = np.array(P_i_const_N)[m - 1, :]
        Ns_logical = N/saved_information_fraction
        Ns_logical_matrix[:,ind] = Ns_logical
        Ns_traditional_matrix[:,ind] = np.ones(fid.shape[0])*N

        if successful_only:
            error_probability_difference[:,ind] = (data_logical-data_ideal)-(data_traditional-data_ideal)

    if successful_only:
        Y,X = np.meshgrid(Ns,fid)
        Z = error_probability_difference
        Xnew,Ynew,Znew = interpolate_2d_grid(X,Y,Z)
    title = "$P_{success}^{logical}-P_{success}^{physical}$, $m=$"+str(m)
    ylabel = "$N_{ps}$"
    if interpolate:
        plot_color(Xnew, Ynew, Znew, xlabel,ylabel,title)
    else:
        plot_color(X, Y, Z, xlabel, ylabel, title)
    if save:
        plt.savefig(f'images\\success_probability_2d_{m}')
    plt.show()


def plot_error_probability(precisions,N=1,save=False,x_points='single',xlims=None, ylims=None):
    fig, axes = plt.subplots()
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', nbins=6)
    _, _, _, _, _, _, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA, _, _, _, P_i, P_l, P_t\
        = load_IPEA_new(rotation_type, noise_type, N)
    ideal = P_i
    traditional = P_t
    logical = P_l

    if x_points == 'single':
        x = f_worst_1q_IPEA
        plt.xlabel('Single Gate Fidelity', fontsize=20)
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
        _, _, _, _, _, _, _, _, _, _, _, _, P_i, P_l, P_t \
            = load_IPEA_new(rotation_type, noise_type, N)
        precision = precisions[i]
        c = next(color)

        j = precisions[i] - 1
        trad_data = traditional[j, :]
        ideal_data = ideal[j, :]
        logical_data = logical[j, :]

        # fill between
        logic = np.abs(logical_data - ideal_data)
        interp_l = interp1d(x, logic,kind='cubic')
        trad = np.abs(trad_data - ideal_data)
        interp_t = interp1d(x, trad,kind='cubic')
        new_x = np.linspace(0.977, 0.99999, 10000)
        new_logic = interp_l(new_x)
        new_trad = interp_t(new_x)

        # intersect_index = np.argmin(np.abs(new_logic - new_trad))
        # plt.scatter(new_x[intersect_index], new_logic[intersect_index], color=c)

        axes.plot(new_x, new_trad, '-.', color=c)
        axes.plot(new_x, new_logic, '-', color=c)
        axes.plot(np.NaN, np.NaN, label='$m=$' + str(precision), color=c)

    axes.set_ylabel('$\Delta P_s(n='+str(N)+')$', fontsize=20)
    axes.plot(np.NaN, np.NaN, '-', color='black', label='logical LPS')
    axes.plot(np.NaN, np.NaN, '-.', color='black', label='physical')
    for label in (axes.get_xticklabels()):
        label.set_fontsize(20)
    for label in (axes.get_yticklabels()):
        label.set_fontsize(20)
    # axes.set_yscale('log')
    axes.legend(fontsize=15)
    # plt.title('Post-Selected Trials for each Digit: '+str(N))
    axes.set_ylim((0,1))
    plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.95])
    if xlims is not None:
        axes.set_xlim(xlims)
    if ylims is not None:
        axes.set_ylim(ylims)
    if save:
        plt.savefig(f'images\\probability_N_{N}.jpg')
    plt.show()


def plot_num_trials_each_digit(precisions,Ns=(1,3,5,7,9,11,15,21,29,39,51,65,79,99),epsilon=0.05,save=False,x_points='single',xlims=None, ylims=None):
    fig, axes = plt.subplots()

    _, _, l_STD_withLI, _, _, l_STD_noLI, f_worst_1q_IPEA, f_worst_2q_IPEA, T_list_for_IPEA, _, _, _, _, _, _\
        = load_IPEA_new(rotation_type, noise_type, 1)

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

    for i in precisions:
        saved_information_fraction = (np.array(l_STD_noLI)[i - 1, :] / np.array(l_STD_withLI)[i - 1, :]) ** 2

        # build data
        Ns_logical_tot = np.zeros((x.shape[0],len(Ns)))
        Ns_trad_tot = np.zeros((x.shape[0],len(Ns)))
        P_l_tot = np.zeros((x.shape[0],len(Ns)))
        P_t_tot = np.zeros((x.shape[0],len(Ns)))
        Fid_tot = np.zeros((x.shape[0],len(Ns)))
        for ind,n in enumerate(Ns):
            Ns_logical_tot[:,ind] = list(np.real(n/saved_information_fraction))
            Ns_trad_tot[:,ind] = np.ones_like(list(np.real(saved_information_fraction)))*n
            _, _, _, _, _, _, _, _, _, _, _, _, P_i, P_l, P_t \
                = load_IPEA_new(rotation_type, noise_type, n)
            P_l_tot[:,ind] = list(P_l[i-1,:])
            P_t_tot[:,ind] = list(P_t[i-1,:])
            Fid_tot[:,ind] = list(x)

        Fnew_t, Nnew_t, Pnew_t = interpolate_2d_grid(Fid_tot, Ns_trad_tot, P_t_tot)
        Fnew_l, Nnew_l, Pnew_l = interpolate_2d_grid(Fid_tot, Ns_logical_tot, P_l_tot)
        Pnew_l = np.nan_to_num(Pnew_l,nan=10)
        #
        # Fnew_t, Nnew_t, Pnew_t = Fid_tot, Ns_trad_tot, P_t_tot
        # Fnew_l, Nnew_l, Pnew_l = Fid_tot, Ns_logical_tot, P_l_tot

        # find intersections with epsilon
        Ns_logical_intersect = []
        Fs_logical_intersect = []
        Ns_trad_intersect = []
        Fs_trad_intersect = []
        for ind,fidelity in enumerate(Fnew_t[:,0]):
            intersection_index_l = np.argmin(np.abs(Pnew_l[ind,:]-(1-epsilon)))
            intersection_index_t = np.argmin(np.abs(Pnew_t[ind,:]-(1-epsilon)))
            Fs_logical_intersect.append(Fnew_l[ind,intersection_index_l])
            Ns_logical_intersect.append(Nnew_l[ind,intersection_index_l])
            Fs_trad_intersect.append(Fnew_t[ind,intersection_index_t])
            Ns_trad_intersect.append(Nnew_t[ind,intersection_index_t])

        c = next(color)
        axes.plot(Fs_logical_intersect, Ns_logical_intersect, '-', color=c)
        axes.plot(Fs_trad_intersect, Ns_trad_intersect, '-.', color=c)
        axes.plot(np.NaN, np.NaN, label='$m=$' + str(i), color=c)

    axes.set_ylabel('Number of Trials per Digit', fontsize=13)
    axes.plot(np.NaN, np.NaN, '-', color='black', label='logical ancilla LPS')
    axes.plot(np.NaN, np.NaN, '-.', color='black', label='physical ancilla')
    for label in (axes.get_xticklabels()):
        label.set_fontsize(13)
    axes.set_yscale('log')
    axes.legend()
    plt.title('$\epsilon=$'+str(epsilon))
    # axes.set_ylim((0, 1))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if xlims is not None:
        axes.set_xlim(xlims)
    if ylims is not None:
        axes.set_ylim(ylims)
    if save:
        plt.savefig(f'images\\probability_epsilon_{epsilon}.jpg')
    plt.show()

save=True
# plot_error_probability_heatmap(Ns = (1,3,5,7,9,11,15,21,29,39,51,65,79,99),m=9,successful_only=True,interpolate=True,save=save)
# plot_error_probability_heatmap(Ns = (1,3,5,7,9,11,15,21,29,39,51,65,79,99),m=8,successful_only=True,interpolate=True,save=save)
# plot_error_probability_heatmap(Ns = (1,3,5,7,9,11,15,21,29,39,51,65,79,99),m=7,successful_only=True,interpolate=True,save=save)
# plot_error_probability_heatmap(Ns = (1,3,5,7,9,11,15,21,29,39,51,65,79,99),m=6,successful_only=True,interpolate=True,save=save)
# plot_error_probability_heatmap(Ns = (1,3,5,7,9,11,15,21,29,39,51,65,79,99),m=5,successful_only=True,interpolate=True,save=save)
# plot_error_probability_heatmap(Ns = (1,3,5,7,9,11,15,21,29,39,51,65,79,99),m=4,successful_only=True,interpolate=True,save=save)
# plot_error_probability_heatmap(Ns = (1,3,5,7,9,11,15,21,29,39,51,65,79,99),m=3,successful_only=True,interpolate=True,save=save)
# plot_error_probability_heatmap(Ns = (1,3,5,7,9,11,15,21,29,39,51,65,79,99),m=2,successful_only=True,interpolate=True,save=save)
# plot_error_probability_heatmap(Ns = (1,3,5,7,9,11,15,21,29,39,51,65,79,99),m=1,successful_only=True,interpolate=True,save=save)
# plot_error_probability([3,6,9],N=1,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=3,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=5,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=7,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=9,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=11,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=15,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=21,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=29,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=39,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=51,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=65,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=79,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_error_probability([3,6,9],N=99,save=save,x_points='single',xlims=None,ylims=(0,1))
# plot_num_trials_each_digit([3,5,9],Ns=(1,3,5,7,9,11,15,21,29,39,51,65,79,99),epsilon=0.05,save=False,x_points='single',xlims=None, ylims=None)
if (plot_data_new or plot_data_from_2021):
    # plot_IPEA_single_fig([1,2,3],withLI=True,x_points='single',save=False)
    # plot_IPEA([5,6,9],withLI=True,x_points='single',save=False)
    # plot_IPEA_infidelity([5,6,9],withLI=True,x_points='single',save=False)
    plot_IPEA_single_fig([5,6,9],withLI=True,x_points='single',save=False,xstart=0.997,yend=0.1)
    # plot_IPEA_single_fig([5,6,9],withLI=False,x_points='single',save=True)
    # plot_IPEA_MEAN_single_fig([5,6,9],x_points='single',save=True,xstart=0.9825,yend=0.05,ystart=0,xend=1)
    # plot_estimated_number_of_trials_new([2,5,6,7,8,9],x_points='single',save=False)
    # plot_IPEA_single_fig_infidelity([5,6,9],withLI=True,x_points='single',save=False)



"""
This file has functions to create the data of figure 3a in the letter titled
"Hybrid Logical-Physical Qubit Interaction for Quantum Metrology"
re-creation of code from year 2021 for paper GitHub
Written by Nadav Carmel 17/04/2023


"""

#########################################################################################
######################################### imports #######################################
#########################################################################################

from simulators.LogicalConstants import *
import math
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import os
import time
import qiskit
from scipy.optimize import curve_fit
from scipy.ndimage import convolve1d
from simulators.Utils import map_decohere_to_worst_gate_fidelity

print('all packages imported')


#########################################################################################
########################## constants to control this file ###############################
#########################################################################################

# generate data. default should be e = '0', perfect_syndrome_extraction = False, T2_list_main=None, num_angles=10

perfect_syndrome_extraction = False
e = '0'
T2_list_main = None
# T2_list_main = [10,12,14,16,18,20,30,40,50,60,70,80,90,100,1000,10000]
num_angles = 10
foldername = 'recreate_paper'

# plot-related constants
K = False
size = 10
Tgate=1
T2_list = list(np.linspace(Tgate * 1, Tgate * 20, 20, endpoint=True)) + list(
    np.linspace(Tgate * 20, Tgate * 120, 50, endpoint=False)) + list(
    np.geomspace(Tgate * 125, Tgate * 1000, 25, endpoint=False)) + list(
    np.geomspace(Tgate * 1000, Tgate * 10000, 25, endpoint=True))

generate_data = False

plot_data_new = False

plot_data_from_2021 = True


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
    # repair from |-> to |0>
    H_for_LogicalRegister(reg, noisy)
    reg.run([[('X',0,None,None),('X',1,None,None),('X',2,None,None),('X',3,None,None),('X',4,None,None)]])

def EC_for_LogicalRegister(reg, noisy, perfect=False, LPS=True):
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
    ## use sensor qubit as measurement qubit - set it in the |0> state assuming it is in |1>
    if e=='1':
        applyOperator(reg,noisy,reg.Sx[5]) # update mes qubit to be in state 0

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
    if e=='1':
        applyOperator(reg,noisy,reg.Sx[5]) # update mes qubit to be in state 1

    if (noisy and perfect):
        reg.setError(dephase=dephase,amplitude_damp=amp)

def logicalCNOT(reg):
    """
    assumes control is logical 5 qubit in indexes 0-4
    """
    reg.run([[('H', 5, None, None)]])
    # print(debugLogical(reg.state))
    reg.run([[('CZ', 5, 0, None)],[('CZ', 5, 1, None)],[('CZ', 5, 2, None)],[('CZ', 5, 3, None)],[('CZ', 5, 4, None)]])
    reg.run([[('H', 5, None, None)]])

def cU_for_LogicalRegister(reg,phi,type='logical'):
    """
    runs logical controlled operation for logical register with sensor qubit
    :param reg: quantum register object
    :param noisy: boolean - True for noisy register
    :param phi: such that 2*pi*phi is the eigenphase
    :return: None
    """
    # first sensor rotation
    reg.run([[('SingleQubitOperator',5,None,(-1j*phi/8*sigmaz()).expm())]])
    if type == 'logical':
        # controlled not
        reg.run([[('CNOT', 5, 0, None),('CNOT', 5, 1, None),('CNOT', 5, 2, None),('CNOT', 5, 3, None),('CNOT', 5, 4, None)]])
    # second sensor rotation
    reg.run([[('SingleQubitOperator',5,None,(1j*phi/4*sigmaz()).expm())]])
    if type == 'logical':
        # controlled not
        reg.run([[('CNOT', 5, 0, None),('CNOT', 5, 1, None),('CNOT', 5, 2, None),('CNOT', 5, 3, None),('CNOT', 5, 4, None)]])

    # third sensor rotation
    reg.run([[('SingleQubitOperator',5,None,(-1j*phi/8*sigmaz()).expm())]])

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
    reg.run([[('SingleQubitOperator', 2, None, (-1j*theta/2*sigmaz()).expm())]])
    reg.run([[('CNOT', 2, 0, None)]])
    reg.run([[('CNOT', 2, 4, None)]])
    reg.run([[('SingleQubitOperator',0,None,K.dag()),('Y',2,None,None),('SingleQubitOperator',4,None,K.dag())]])

def measure_for_LogicalRegister(reg,noisy):
    """
    does the action of measurement of the logical ancilla.
    :param reg: quantum register
    :param noisy: boolean - True for noisy register
    :return: (digit,p) for measurement result and probability
    """
    ## use sensor qubit as measurement qubit - set it in the |0> state assuming it is in |1>
    if e=='1':
        applyOperator(reg,noisy,reg.Sx[5]) # update mes qubit to be in state 0

    reg.run([[('H',5,None,None)]])
    reg.run([[('CZ',0,5,None)]])
    reg.run([[('CZ',1,5,None)]])
    reg.run([[('CZ',2,5,None)]])
    reg.run([[('CZ',3,5,None)]])
    reg.run([[('CZ',4,5,None)]])
    reg.run([[('H',5,None,None)]])

    # measurement and collapse
    mes_qubit_state = reg.state.ptrace([5])
    p0 = mes_qubit_state[0,0]
    p1 = mes_qubit_state[1,1]
    project = reg.qI
    if np.random.rand() < p0:
        return '0', [p0,p1]
    else:
        return '1', [p0,p1]

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
    return result[2:]

def Kitaev_for_simpleRegister(PS=False, noisy=False, reg_params = None, phi = 1/8, power = 0,dephase=True, amp_damp = False):
    """
    this function creates a noisy or perfect 2-qubit quantum register and runs IQPE on it.
     prints probabilities in each stage and the final measurements results.
    :param PS: boolean, if True: put Post Selection in the run
    :param noisy: boolean, if True: run with noisy register
    :param reg_params: list of (T1,T2,T1s,T2s,dt,Tgate) if noisy==True
    :param phi: such that 2*pi*phi is the eigenphase
    :param percision: number of digits
    :return: None
    """
    if noisy:
        (T1,T2,T1s,T2s,dt,Tgate) = reg_params
        register = InCoherentQuantumRegister(2,initial_state_for_NoisyRegister(e),T1,T2,T1s=T1s,T2s=T2s,dt=dt,Tgate=Tgate)
        registerK = InCoherentQuantumRegister(2,initial_state_for_NoisyRegister(e),T1,T2,T1s=T1s,T2s=T2s,dt=dt,Tgate=Tgate)
        # register.collect = True
        # register.collect_bloch = True
        register.setError(dephase=dephase,amplitude_damp=amp_damp)
        registerK.setError(dephase=dephase,amplitude_damp=amp_damp)
    else:
        register = EfficientQuantumRegister(2,initial_state_for_PerfectRegister(e))
        registerK = EfficientQuantumRegister(2,initial_state_for_PerfectRegister(e))

    command = [[('H',0,None,None)]]
    digit = register.run(command)
    digit = registerK.run(command)


    command = [[('Rz',0,None,np.pi/2)]]
    digit = registerK.run(command)


    command = [[('Rz',1,None,2**power*2*np.pi*phi/4)]]
    digit = register.run(command)
    digit = registerK.run(command)

    command = [[('CNOT',1,0,None)]]
    digit = register.run(command)
    digit = registerK.run(command)

    command = [[('Rz',1,None,-2**power*2*np.pi*phi/2)]]
    digit = register.run(command)
    digit = registerK.run(command)

    command = [[('CNOT',1,0,None)]]
    digit = register.run(command)
    digit = registerK.run(command)

    command = [[('Rz',1,None,2**power*2*np.pi*phi/4)]]
    digit = register.run(command)
    digit = registerK.run(command)


    # post selection
    if PS:
        command = [[('SingleQubitOperator',1,None,basis(2,int(e))*basis(2,int(e)).dag())]]
        digit = register.run(command)
        digit = registerK.run(command)

    command = [[('H',0,None,None)]]
    digit = register.run(command)
    digit = registerK.run(command)
    # print(registerK.state)

    return register.state, register.state.tr(), registerK.state, registerK.state.tr()

def Kitaev_for_LogicalRegister_1LPS(PS=False, noisy=False, reg_params = None, phi = 1/8, power = 0,dephase=True, amp_damp = False):
    """
    this function creates a noisy or perfect 6-qubit quantum register and runs IQPE on it.
     prints probabilities in each stage and the final measurements results.
    :param PS: boolean, if True: put Post Selection in the run
    :param noisy: boolean, if True: run with noisy register
    :param reg_params: list of (T1,T2,T1s,T2s,dt,Tgate) if noisy==True
    :param phi: such that 2*pi*phi is the eigenphase
    :param percision: number of digits
    :return: [LPSstete, LPSstate_norm, LPSstateK, LPSstateK_norm, ECstate, ECstate_norm, ECstateK, ECstateK_norm]
    """
    if noisy:
        (T1,T2,T1s,T2s,dt,Tgate) = reg_params
        register = InCoherentQuantumRegister(6,initial_state_for_NoisyLogicalRegister(e),T1,T2,T1s=T1s,T2s=T2s,dt=dt,Tgate=Tgate)
        registerK = InCoherentQuantumRegister(6,initial_state_for_NoisyLogicalRegister(e),T1,T2,T1s=T1s,T2s=T2s,dt=dt,Tgate=Tgate)
        # register.collect = True
        # register.collect_bloch = True
        register.setError(dephase=dephase,amplitude_damp=amp_damp)
        registerK.setError(dephase=dephase,amplitude_damp=amp_damp)
    else:
        register = EfficientQuantumRegister(6,initial_state_for_PerfectLogicalRegister(e))
        registerK = EfficientQuantumRegister(6,initial_state_for_PerfectLogicalRegister(e))


    prepare_for_LogicalRegister(register,noisy)

    H_for_LogicalRegister(register, noisy)

    registerK.update(register.state)
    rotation_for_LogicalRegister(registerK,np.pi/2)


    cU_for_LogicalRegister(register, 2**power*2*np.pi*phi)
    cU_for_LogicalRegister(registerK, 2**power*2*np.pi*phi)

    # post selection on |1><1|
    if PS:
        register.run([[('SingleQubitOperator',5,None,basis(2,int(e))*basis(2,int(e)).dag())]])
        registerK.run([[('SingleQubitOperator',5,None,basis(2,int(e))*basis(2,int(e)).dag())]])
        # register.state = register.state.unit()

    H_for_LogicalRegister(register, noisy)
    H_for_LogicalRegister(registerK, noisy)


    EC_for_LogicalRegister(register, noisy, LPS=True)
    EC_for_LogicalRegister(registerK, noisy, LPS=True)
    # print(debugLogical(registerK.state))

    return debugLogical(register.state), register.state.tr(), debugLogical(registerK.state), registerK.state.tr()

def Kitaev_for_LogicalRegister_1EC(PS=False, noisy=False, reg_params = None, phi = 1/8, power = 0,dephase=True, amp_damp = False):
    """
    this function creates a noisy or perfect 6-qubit quantum register and runs IQPE on it.
     prints probabilities in each stage and the final measurements results.
    :param PS: boolean, if True: put Post Selection in the run
    :param noisy: boolean, if True: run with noisy register
    :param reg_params: list of (T1,T2,T1s,T2s,dt,Tgate) if noisy==True
    :param phi: such that 2*pi*phi is the eigenphase
    :param percision: number of digits
    :return: None
    """
    if noisy:
        (T1,T2,T1s,T2s,dt,Tgate) = reg_params
        register = InCoherentQuantumRegister(6,initial_state_for_NoisyLogicalRegister(e),T1,T2,T1s=T1s,T2s=T2s,dt=dt,Tgate=Tgate)
        registerK = InCoherentQuantumRegister(6,initial_state_for_NoisyLogicalRegister(e),T1,T2,T1s=T1s,T2s=T2s,dt=dt,Tgate=Tgate)
        # register.collect = True
        # register.collect_bloch = True
        register.setError(dephase=dephase,amplitude_damp=amp_damp)
        registerK.setError(dephase=dephase,amplitude_damp=amp_damp)
    else:
        register = EfficientQuantumRegister(6,initial_state_for_PerfectLogicalRegister(e))
        registerK = EfficientQuantumRegister(6,initial_state_for_PerfectLogicalRegister(e))


    prepare_for_LogicalRegister(register,noisy)

    H_for_LogicalRegister(register, noisy)

    registerK.update(register.state)
    rotation_for_LogicalRegister(registerK,np.pi/2)


    cU_for_LogicalRegister(register, 2**power*2*np.pi*phi)
    cU_for_LogicalRegister(registerK, 2**power*2*np.pi*phi)

    # post selection on |1><1|
    if PS:
        register.run([[('SingleQubitOperator',5,None,basis(2,int(e))*basis(2,int(e)).dag())]])
        registerK.run([[('SingleQubitOperator',5,None,basis(2,int(e))*basis(2,int(e)).dag())]])
        # register.state = register.state.unit()

    H_for_LogicalRegister(register, noisy)
    H_for_LogicalRegister(registerK, noisy)


    EC_for_LogicalRegister(register, noisy, LPS=False, perfect=perfect_syndrome_extraction)
    EC_for_LogicalRegister(registerK, noisy, LPS=False, perfect=perfect_syndrome_extraction)
    # print(debugLogical(registerK.state))

    return debugLogical(register.state), register.state.tr(), debugLogical(registerK.state), registerK.state.tr()


def main(alg, T2_list_main=None, num_angles=10, name='test'):
    """
    runs algorithm measuring Rz of angle,
    :param num_points: number of different T2 or for the ancilla qubit
    :param num_angles: number of angles to average over
    :param startp: lower T2
    :param endp: higher T2
    :param name: '_LT','_VLT','_VST','','test'
    :param alg:     'traditional' - for regular 2 qubit Kitaev QPE
                    'traditionalSPS' - the above but with one SPS
                    'logical1EC' - ancilla is logical qubit, creating the data for logical post selection (i had a confusion when first creating the data)
                    'logical1LPS' - the above, but creating the data for error correction
    :return: None
    """
    print("-------------------------------------------------------")
    print("starting algorithm named ",alg)
    print("-------------------------------------------------------")
    path = os.getcwd()
    Tgate = 1
    dt = Tgate/20
    reg_params = (1,1,[1,1],[1,1],dt,Tgate)
    if T2_list_main is None:
        T2_list_main = list(set(T2_list))
        T2_list_main.sort()
    T1 = 1
    power = 0
    phi_list = np.linspace(0,1,num_angles,endpoint=False)
    if perfect_syndrome_extraction:
        folder = os.path.join(path, 'data\\Kitaev\\PerfectSyndromeExtraction\\'+name)
    else:
        folder = os.path.join(path, 'data\\Kitaev\\NoisySyndromeExtraction\\' + name)
    try:
        os.makedirs(folder)
    except:
        pass
    try:
        for phi in phi_list:
            ideal, trash, idealK, trash2 = Kitaev_for_simpleRegister(PS=False, noisy=True, reg_params = reg_params, phi = phi, power = power, dephase=False, amp_damp=False)
            filename = folder+'\\idealState_angle'+str(phi)
            np.save(filename,ideal)
            filename = folder+'\\idealStateK_angle'+str(phi)
            np.save(filename,idealK)
    except:
        pass

    fidelities = []
    lostInfo = []
    fidelitiesK = []
    lostInfoK = []
    distance = []
    distanceK = []

    start = time.time()

    for j,T2 in enumerate(T2_list_main[:]):
        stop = time.time()
        print()
        print('started iteration '+str(j)+' in '+ str(stop-start)+' seconds')

        # traditional
        fidelities_angles = []
        lostInfo_angles = []
        fidelitiesK_angles = []
        lostInfoK_angles = []
        distance_angles = []
        distanceK_angles = []

        for i,phi in enumerate(phi_list[:]):
            if i%3 == 0:
                stop = time.time()
                print('started angle iteration number ' + str(i) + ' in ' + str(stop-start) + ' seconds')
            ideal, trash, idealK, trash2 = Kitaev_for_simpleRegister(PS=False, noisy=True, reg_params = reg_params, phi = phi, power = power, dephase=False, amp_damp=False)
            # now for the explored registers
            # traditional
            if alg == 'traditional':
                state, state_norm, stateK,  stateK_norm = Kitaev_for_simpleRegister(PS=False, noisy=True, reg_params = [1,1,[T1,T1],[T2,T2],dt,Tgate], phi = phi, power = power, dephase=True, amp_damp=False)
            # traditional with SPS
            elif alg == 'traditionalSPS':
                state, state_norm, stateK,  stateK_norm = Kitaev_for_simpleRegister(PS=True, noisy=True, reg_params = [1,1,[T1,T1],[T2,T2],dt,Tgate], phi = phi, power = power, dephase=True, amp_damp=False)
            # logical
            elif alg == 'logical1EC':
                state, state_norm, stateK,  stateK_norm = Kitaev_for_LogicalRegister_1EC(PS=True, noisy=True, reg_params = [1,1,[T1,T1,T1,T1,T1,T1],[T2,T2,T2,T2,T2,T2],dt,Tgate], phi = phi, power = power, dephase=True, amp_damp=False)
            # logical 1 LPS
            elif alg == 'logical1LPS':
                state, state_norm, stateK,  stateK_norm = Kitaev_for_LogicalRegister_1LPS(PS=True, noisy=True, reg_params = [1,1,[T1,T1,T1,T1,T1,T1],[T2,T2,T2,T2,T2,T2],dt,Tgate], phi = phi, power = power, dephase=True, amp_damp=False)

            # save states
            try:
                # save states
                filename = os.path.join(folder, alg+'State_T2'+str(T2)+'_phi'+str(phi))
                np.save(filename, state)
                filename = os.path.join(folder, alg+'KState_T2'+str(T2)+'_phi'+str(phi))
                np.save(filename, stateK)

            except:
                pass

            # now update fidelities and lost information
            lostInfo_angles.append(1-state_norm)
            lostInfoK_angles.append(1-stateK_norm)

            # normalize
            state = state/state.tr()
            stateK = stateK/stateK.tr()


            distance_angles.append(np.sqrt((ideal.ptrace(0)[0,0]-state.ptrace(0)[0,0])**2+(ideal.ptrace(0)[1,1]-state.ptrace(0)[1,1])**2))
            distanceK_angles.append(np.sqrt((idealK.ptrace(0)[0,0]-stateK.ptrace(0)[0,0])**2+(idealK.ptrace(0)[1,1]-stateK.ptrace(0)[1,1])**2))

            #calculate fidelity
            fidelities_angles.append(fidelity(ideal,state))
            fidelitiesK_angles.append(fidelity(idealK,stateK))

        # save data for each kind of register, all angles for certain T2
        try:
            # save data for each kind of register, all angles for certain T2
            filename = os.path.join(folder, alg+'_T2_'+str(T2)+'_angle_fidelities')
            np.save(filename,fidelities_angles)
            filename = os.path.join(folder, alg+'K_T2_'+str(T2)+'_angle_fidelities')
            np.save(filename,fidelitiesK_angles)

            # save data for each kind of register, all angles for certain T2
            filename = os.path.join(folder, alg+'_T2_'+str(T2)+'_angle_lost_information')
            np.save(filename,lostInfo_angles)
            filename = os.path.join(folder, alg+'K_T2_'+str(T2)+'_angle_lost_information')
            np.save(filename,lostInfoK_angles)

            # save data for each kind of register, all angles for certain T2
            filename = os.path.join(folder, alg+'_T2_'+str(T2)+'_angle_distance')
            np.save(filename,distance_angles)
            filename = os.path.join(folder, alg+'K_T2_'+str(T2)+'_angle_distance')
            np.save(filename,distanceK_angles)

        except:
            pass

        # do the averaging
        fidelities.append(np.sum(np.array(fidelities_angles))/num_angles)
        fidelitiesK.append(np.sum(np.array(fidelitiesK_angles))/num_angles)
        lostInfo.append(np.sum(np.array(lostInfo_angles))/num_angles)
        lostInfoK.append(np.sum(np.array(lostInfoK_angles))/num_angles)
        distance.append(np.sum(np.array(distance_angles))/num_angles)
        distanceK.append(np.sum(np.array(distanceK_angles))/num_angles)

    # save lists of fidelity and lost info vs T2
    try:
        # save lists of fidelity vs T2
        np.save(os.path.join(folder, alg+'_lost_information_average'),lostInfo)
        np.save(os.path.join(folder, alg+'_fidelity_average'),fidelities)
        np.save(os.path.join(folder, alg+'_distance_average'),distance)
        np.save(os.path.join(folder, alg+'K_lost_information_average'),lostInfoK)
        np.save(os.path.join(folder, alg+'K_fidelity_average'),fidelitiesK)
        np.save(os.path.join(folder, alg+'K_distance_average'),distanceK)

    except:
        pass


if generate_data:
    main('traditionalSPS',T2_list_main=T2_list_main,num_angles=num_angles,name=foldername)
    main('traditional',T2_list_main=T2_list_main,num_angles=num_angles,name=foldername)
    main('logical1EC',T2_list_main=T2_list_main,num_angles=num_angles,name=foldername)
    main('logical1LPS',T2_list_main=T2_list_main,num_angles=num_angles,name=foldername)

#########################################################################################
########### load data from folder for data generated now  ###############################
#########################################################################################

def load_Kitaev_data_new(K,name):

    if perfect_syndrome_extraction:
        name = 'PerfectSyndromeExtraction\\' + name
    else:
        name = 'NoisySyndromeExtraction\\' + name

    tradfidelities = np.load(
        'data\\Kitaev\\'+name+'\\traditional_fidelity_average.npy')
    tradfidelitiesK = np.load(
        'data\\Kitaev\\'+name+'\\traditionalK_fidelity_average.npy')
    traditionalLostinfo = np.load(
        'data\\Kitaev\\'+name+'\\traditional_lost_information_average.npy')
    traditionalLostinfoK = np.load(
        'data\\Kitaev\\'+name+'\\traditionalK_lost_information_average.npy')
    traditionalDistance = np.load(
        'data\\Kitaev\\'+name+'\\traditional_distance_average.npy')
    traditionalDistanceK = np.load(
        'data\\Kitaev\\'+name+'\\traditionalK_distance_average.npy')

    tradSPSfidelities = np.load(
        'data\\Kitaev\\'+name+'\\traditionalSPS_fidelity_average.npy')
    tradSPSfidelitiesK = np.load(
        'data\\Kitaev\\'+name+'\\traditionalSPSK_fidelity_average.npy')
    tradSPSitionalLostinfo = np.load(
        'data\\Kitaev\\'+name+'\\traditionalSPS_lost_information_average.npy')
    tradSPSitionalLostinfoK = np.load(
        'data\\Kitaev\\'+name+'\\traditionalSPSK_lost_information_average.npy')
    tradSPSitionalDistance = np.load(
        'data\\Kitaev\\'+name+'\\traditionalSPS_distance_average.npy')
    tradSPSitionalDistanceK = np.load(
        'data\\Kitaev\\'+name+'\\traditionalSPSK_distance_average.npy')

    if K:
        f_t = list(tradfidelitiesK)
        f_tSPS = list(tradSPSfidelitiesK)
        li_t = list(traditionalLostinfoK)
        li_tSPS =list(tradSPSitionalLostinfoK)
        D_t = list(traditionalDistanceK)
        D_tSPS = list(tradSPSitionalDistanceK)
    else:
        f_t = list(tradfidelities)
        f_tSPS = list(tradSPSfidelities)
        li_t = list(traditionalLostinfo)
        li_tSPS =list(tradSPSitionalLostinfo)
        D_t = list(traditionalDistance)
        D_tSPS = list(tradSPSitionalDistance)

    f_lLPS1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1LPS_fidelity_average.npy')
    f_lLPSK1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1LPSK_fidelity_average.npy')
    li_lLPS1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1LPS_lost_information_average.npy')
    li_lLPSK1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1LPSK_lost_information_average.npy')
    d_lLPS1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1LPS_distance_average.npy')
    d_lLPSK1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1LPSK_distance_average.npy')

    f_lEC1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1EC_fidelity_average.npy')
    f_lECK1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1ECK_fidelity_average.npy')
    li_lEC1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1EC_lost_information_average.npy')
    li_lECK1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1ECK_lost_information_average.npy')
    d_lEC1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1EC_distance_average.npy')
    d_lECK1 = np.load(
        'data\\Kitaev\\'+name+'\\logical1ECK_distance_average.npy')

    if K:
        f_lEC = list(f_lECK1)
        f_lLPS = list(f_lLPSK1)
        li_lEC = list(li_lECK1)
        li_lLPS = list(li_lLPSK1)
        D_lEC = list(d_lECK1)
        D_lLPS = list(d_lLPSK1)
    else:
        f_lEC = list(f_lEC1)
        f_lLPS = list(f_lLPS1)
        li_lEC = list(li_lEC1)
        li_lLPS = list(li_lLPS1)
        D_lEC = list(d_lEC1)
        D_lLPS = list(d_lLPS1)

    return f_t, f_tSPS, li_t, li_tSPS, D_t, D_tSPS, f_lEC, f_lLPS, li_lEC, li_lLPS, D_lEC, D_lLPS

if plot_data_new:
    f_t, f_tSPS, li_t, li_tSPS, D_t, D_tSPS, f_lEC, f_lLPS, li_lEC, li_lLPS, D_lEC, D_lLPS = load_Kitaev_data_new(K,foldername)
    f_worst_1q = map_decohere_to_worst_gate_fidelity(T2_list_main,1,Tgate=1,decohere="2",save=False)
    f_worst_2q = map_decohere_to_worst_gate_fidelity(T2_list_main,2,Tgate=1,decohere="2",save=False)


#########################################################################################
########### load data from folder for data generated in 2021  ###########################
#########################################################################################

def load_Kitaev_data(K):

    tradfidelities = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditional_fidelity_average.npy')
    tradfidelitiesK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditionalK_fidelity_average.npy')
    traditionalLostinfo = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditional_lost_information_average.npy')
    traditionalLostinfoK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditionalK_lost_information_average.npy')
    traditionalDistance = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditional_distance_average.npy')
    traditionalDistanceK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditionalK_distance_average.npy')

    tradSPSfidelities = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditionalSPS_fidelity_average.npy')
    tradSPSfidelitiesK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditionalSPSK_fidelity_average.npy')
    tradSPSitionalLostinfo = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditionalSPS_lost_information_average.npy')
    tradSPSitionalLostinfoK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditionalSPSK_lost_information_average.npy')
    tradSPSitionalDistance = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditionalSPS_distance_average.npy')
    tradSPSitionalDistanceK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\traditionalSPSK_distance_average.npy')

    ltradfidelities = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditional_fidelity_average.npy')
    ltradfidelitiesK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditionalK_fidelity_average.npy')
    ltraditionalLostinfo = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditional_lost_information_average.npy')
    ltraditionalLostinfoK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditionalK_lost_information_average.npy')
    ltraditionalDistance = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditional_distance_average.npy')
    ltraditionalDistanceK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditionalK_distance_average.npy')

    ltradSPSfidelities = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditionalSPS_fidelity_average.npy')
    ltradSPSfidelitiesK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditionalSPSK_fidelity_average.npy')
    ltradSPSitionalLostinfo = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditionalSPS_lost_information_average.npy')
    ltradSPSitionalLostinfoK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditionalSPSK_lost_information_average.npy')
    ltradSPSitionalDistance = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditionalSPS_distance_average.npy')
    ltradSPSitionalDistanceK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\traditionalSPSK_distance_average.npy')

    vltradfidelities = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditional_fidelity_average.npy')
    vltradfidelitiesK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditionalK_fidelity_average.npy')
    vltraditionalLostinfo = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditional_lost_information_average.npy')
    vltraditionalLostinfoK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditionalK_lost_information_average.npy')
    vltraditionalDistance = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditional_distance_average.npy')
    vltraditionalDistanceK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditionalK_distance_average.npy')

    vltradSPSfidelities = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditionalSPS_fidelity_average.npy')
    vltradSPSfidelitiesK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditionalSPSK_fidelity_average.npy')
    vltradSPSitionalLostinfo = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditionalSPS_lost_information_average.npy')
    vltradSPSitionalLostinfoK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditionalSPSK_lost_information_average.npy')
    vltradSPSitionalDistance = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditionalSPS_distance_average.npy')
    vltradSPSitionalDistanceK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\traditionalSPSK_distance_average.npy')

    vstradfidelities = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditional_fidelity_average.npy')
    vstradfidelitiesK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditionalK_fidelity_average.npy')
    vstraditionalLostinfo = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditional_lost_information_average.npy')
    vstraditionalLostinfoK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditionalK_lost_information_average.npy')
    vstraditionalDistance = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditional_distance_average.npy')
    vstraditionalDistanceK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditionalK_distance_average.npy')

    vstradSPSfidelities = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditionalSPS_fidelity_average.npy')
    vstradSPSfidelitiesK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditionalSPSK_fidelity_average.npy')
    vstradSPSitionalLostinfo = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditionalSPS_lost_information_average.npy')
    vstradSPSitionalLostinfoK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditionalSPSK_lost_information_average.npy')
    vstradSPSitionalDistance = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditionalSPS_distance_average.npy')
    vstradSPSitionalDistanceK = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\traditionalSPSK_distance_average.npy')

    if K:
        f_t = list(vstradfidelitiesK) + list(tradfidelitiesK) + list(ltradfidelitiesK) + list(vltradfidelitiesK)
        f_tSPS = list(vstradSPSfidelitiesK) + list(tradSPSfidelitiesK) + list(ltradSPSfidelitiesK) + list(
            vltradSPSfidelitiesK)
        li_t = list(vstraditionalLostinfoK) + list(traditionalLostinfoK) + list(ltraditionalLostinfoK) + list(
            vltraditionalLostinfoK)
        li_tSPS = list(vstradSPSitionalLostinfoK) + list(tradSPSitionalLostinfoK) + list(
            ltradSPSitionalLostinfoK) + list(vltradSPSitionalLostinfoK)
        D_t = list(vstraditionalDistanceK) + list(traditionalDistanceK) + list(ltraditionalDistanceK) + list(
            vltraditionalDistanceK)
        D_tSPS = list(vstradSPSitionalDistanceK) + list(tradSPSitionalDistanceK) + list(
            ltradSPSitionalDistanceK) + list(vltradSPSitionalDistanceK)
    else:
        f_t = list(vstradfidelities) + list(tradfidelities) + list(ltradfidelities) + list(vltradfidelities)
        f_tSPS = list(vstradSPSfidelities) + list(tradSPSfidelities) + list(ltradSPSfidelities) + list(
            vltradSPSfidelities)
        li_t = list(vstraditionalLostinfo) + list(traditionalLostinfo) + list(ltraditionalLostinfo) + list(
            vltraditionalLostinfo)
        li_tSPS = list(vstradSPSitionalLostinfo) + list(tradSPSitionalLostinfo) + list(ltradSPSitionalLostinfo) + list(
            vltradSPSitionalLostinfo)
        D_t = list(vstraditionalDistance) + list(traditionalDistance) + list(ltraditionalDistance) + list(
            vltraditionalDistance)
        D_tSPS = list(vstradSPSitionalDistance) + list(tradSPSitionalDistance) + list(ltradSPSitionalDistance) + list(
            vltradSPSitionalDistance)

    f_lEC1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1LPS_fidelity_average.npy')
    f_lECK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1LPSK_fidelity_average.npy')
    li_lEC1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1LPS_lost_information_average.npy')
    li_lECK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1LPSK_lost_information_average.npy')
    d_lEC1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1LPS_distance_average.npy')
    d_lECK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1LPSK_distance_average.npy')

    f_lLPS1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1EC_fidelity_average.npy')
    f_lLPSK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1ECK_fidelity_average.npy')
    li_lLPS1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1EC_lost_information_average.npy')
    li_lLPSK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1ECK_lost_information_average.npy')
    d_lLPS1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1EC_distance_average.npy')
    d_lLPSK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logical1ECK_distance_average.npy')

    f_lEC2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1LPS_fidelity_average.npy')
    f_lECK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1LPSK_fidelity_average.npy')
    li_lEC2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1LPS_lost_information_average.npy')
    li_lECK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1LPSK_lost_information_average.npy')
    d_lEC2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1LPS_distance_average.npy')
    d_lECK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1LPSK_distance_average.npy')

    f_lLPS2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1EC_fidelity_average.npy')
    f_lLPSK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1ECK_fidelity_average.npy')
    li_lLPS2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1EC_lost_information_average.npy')
    li_lLPSK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1ECK_lost_information_average.npy')
    d_lLPS2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1EC_distance_average.npy')
    d_lLPSK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logical1ECK_distance_average.npy')

    f_lEC3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1LPS_fidelity_average.npy')
    f_lECK3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1LPSK_fidelity_average.npy')
    li_lEC3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1LPS_lost_information_average.npy')
    li_lECK3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1LPSK_lost_information_average.npy')
    d_lEC3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1LPS_distance_average.npy')
    d_lECK3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1LPSK_distance_average.npy')

    f_lLPS3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1EC_fidelity_average.npy')
    f_lLPSK3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1ECK_fidelity_average.npy')
    li_lLPS3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1EC_lost_information_average.npy')
    li_lLPSK3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1ECK_lost_information_average.npy')
    d_lLPS3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1EC_distance_average.npy')
    d_lLPSK3 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VLT\\logical1ECK_distance_average.npy')

    f_lEC4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1LPS_fidelity_average.npy')
    f_lECK4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1LPSK_fidelity_average.npy')
    li_lEC4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1LPS_lost_information_average.npy')
    li_lECK4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1LPSK_lost_information_average.npy')
    d_lEC4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1LPS_distance_average.npy')
    d_lECK4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1LPSK_distance_average.npy')

    f_lLPS4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1EC_fidelity_average.npy')
    f_lLPSK4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1ECK_fidelity_average.npy')
    li_lLPS4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1EC_lost_information_average.npy')
    li_lLPSK4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1ECK_lost_information_average.npy')
    d_lLPS4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1EC_distance_average.npy')
    d_lLPSK4 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_VST\\logical1ECK_distance_average.npy')

    if K:
        f_lEC = list(f_lECK4) + list(f_lECK1) + list(f_lECK2) + list(f_lECK3)
        f_lLPS = list(f_lLPSK4) + list(f_lLPSK1) + list(f_lLPSK2) + list(f_lLPSK3)
        li_lEC = list(li_lECK4) + list(li_lECK1) + list(li_lECK2) + list(li_lECK3)
        li_lLPS = list(li_lLPSK4) + list(li_lLPSK1) + list(li_lLPSK2) + list(li_lLPSK3)
        D_lEC = list(d_lECK4) + list(d_lECK1) + list(d_lECK2) + list(d_lECK3)
        D_lLPS = list(d_lLPSK4) + list(d_lLPSK1) + list(d_lLPSK2) + list(d_lLPSK3)
    else:
        f_lEC = list(f_lEC4) + list(f_lEC1) + list(f_lEC2) + list(f_lEC3)
        f_lLPS = list(f_lLPS4) + list(f_lLPS1) + list(f_lLPS2) + list(f_lLPS3)
        li_lEC = list(li_lEC4) + list(li_lEC1) + list(li_lEC2) + list(li_lEC3)
        li_lLPS = list(li_lLPS4) + list(li_lLPS1) + list(li_lLPS2) + list(li_lLPS3)
        D_lEC = list(d_lEC4) + list(d_lEC1) + list(d_lEC2) + list(d_lEC3)
        D_lLPS = list(d_lLPS4) + list(d_lLPS1) + list(d_lLPS2) + list(d_lLPS3)

    f_lFTEC1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1EC_fidelity_average.npy')
    f_lFTECK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1ECK_fidelity_average.npy')
    li_lFTEC1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1EC_lost_information_average.npy')
    li_lFTECK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1ECK_lost_information_average.npy')
    d_lFTEC1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1EC_distance_average.npy')
    d_lFTECK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1ECK_distance_average.npy')

    f_lFTLPS1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1LPS_fidelity_average.npy')
    f_lFTLPSK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1LPSK_fidelity_average.npy')
    li_lFTLPS1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1LPS_lost_information_average.npy')
    li_lFTLPSK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1LPSK_lost_information_average.npy')
    d_lFTLPS1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1LPS_distance_average.npy')
    d_lFTLPSK1 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS\\logicalFT1LPSK_distance_average.npy')

    f_lFTEC2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1EC_fidelity_average.npy')
    f_lFTECK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1ECK_fidelity_average.npy')
    li_lFTEC2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1EC_lost_information_average.npy')
    li_lFTECK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1ECK_lost_information_average.npy')
    d_lFTEC2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1EC_distance_average.npy')
    d_lFTECK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1ECK_distance_average.npy')

    f_lFTLPS2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1LPS_fidelity_average.npy')
    f_lFTLPSK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1LPSK_fidelity_average.npy')
    li_lFTLPS2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1LPS_lost_information_average.npy')
    li_lFTLPSK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1LPSK_lost_information_average.npy')
    d_lFTLPS2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1LPS_distance_average.npy')
    d_lFTLPSK2 = np.load(
        'data_2021\\averageing_fidelity\\Kitaev\\logical_FT_improved_trad_SPS_LT\\logicalFT1LPSK_distance_average.npy')

    if K:
        f_lFTEC = list(f_lFTECK1) + list(f_lFTECK2)
        f_lFTLPS = list(f_lFTLPSK1) + list(f_lFTLPSK2)
        li_lFTEC = list(li_lFTECK1) + list(li_lFTECK2)
        li_lFTLPS = list(li_lFTLPSK1) + list(li_lFTLPSK2)
        D_lFTEC = list(d_lFTECK1) + list(d_lFTECK2)
        D_lFTLPS = list(d_lFTLPSK1) + list(d_lFTLPSK2)
    else:
        f_lFTEC = list(f_lFTEC1) + list(f_lFTEC2)
        f_lFTLPS = list(f_lFTLPS1) + list(f_lFTLPS2)
        li_lFTEC = list(li_lFTEC1) + list(li_lFTEC2)
        li_lFTLPS = list(li_lFTLPS1) + list(li_lFTLPS2)
        D_lFTEC = list(d_lFTEC1) + list(d_lFTEC2)
        D_lFTLPS = list(d_lFTLPS1) + list(d_lFTLPS2)

    return f_t,f_tSPS,li_t,li_tSPS,D_t,D_tSPS,f_lEC,f_lLPS,li_lEC,li_lLPS,D_lEC,D_lLPS,f_lFTEC,f_lFTLPS,li_lFTEC,li_lFTLPS,D_lFTEC,D_lFTLPS

def load_WCGF():

    f_worst_1q = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_1_T2.npy')
    f_worst_2q = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_2_T2.npy')
    f_worst_1q_T1 = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_1_T1.npy')
    f_worst_2q_T1 = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_2_T1.npy')
    f_worst_1q_CNOTs = np.load('data_2021\\WCGF\\N_1_decohere_mode_2_for_CNOTs_explore.npy')
    f_worst_2q_CNOTs = np.load('data_2021\\WCGF\\N_2_decohere_mode_2_for_CNOTs_explore.npy')
    return f_worst_1q,f_worst_2q,f_worst_1q_T1,f_worst_2q_T1,f_worst_1q_CNOTs,f_worst_2q_CNOTs

if plot_data_from_2021:
    f_t, f_tSPS, li_t, li_tSPS, D_t, D_tSPS, f_lEC, f_lLPS, li_lEC, li_lLPS, D_lEC, D_lLPS, f_lFTEC, f_lFTLPS, li_lFTEC, li_lFTLPS, D_lFTEC, D_lFTLPS = load_Kitaev_data(K)
    f_worst_1q,f_worst_2q,f_worst_1q_T1,f_worst_2q_T1,_,_ = load_WCGF()

#########################################################################################
############################## plot #####################################################
#########################################################################################

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


def calc_error(smoothed, original, window):
    smoothed = np.array(smoothed)
    original = np.array(original)
    errors = np.abs(smoothed - original)
    weights = np.ones(2 * window + 1) / (2 * window + 1)
    yerr = convolve1d(errors, weights, mode='nearest')
    return yerr


def draw_error_band(ax, x, y, err, **kwargs):
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * err
    yp = y + ny * err
    xn = x - nx * err
    yn = y - ny * err

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[len(xp)] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))


def set_ticks_size(ax, x, y):
    # Set tick font size
    for label in (ax.get_xticklabels()):
        label.set_fontsize(x)
    # Set tick font size
    for label in (ax.get_yticklabels()):
        label.set_fontsize(y)


def plot_Kitaev(x_points='two',window=10,save=False):
    start = 1
    end = 120

    if plot_data_from_2021:
        if x_points == 'single':
            start = 10
            fig, ax = plt.subplots()
            x = list(f_worst_1q[start:19]) + list(f_worst_1q[20:end])
            f_lEC_averaged = list(f_lEC[start:19])+[0.5*(f_lEC[19]+f_lEC[20])]+list(f_lEC[21:end])
            yhat = savitzky_golay(f_lEC_averaged, 41, 3)  # window size 51, polynomial order 3
            yerr = calc_error(yhat, f_lEC_averaged, window)
            plt.fill_between(x, yhat - yerr, yhat + yerr, facecolor="green", edgecolor="none", alpha=.3)
            ax.plot(x, yhat, label='logical ancilla SPS+EC', color='green')
            x=f_worst_1q[start:end]
            ax.plot(x, np.array(f_lLPS[start:end]), label='logical ancilla SPS+LPS', color='blue')
            ax.plot(x, np.array(f_tSPS[start:end]), label='physical ancilla SPS', color='red')
            ax.set_xlabel('Worst-Case Single Gate Fidelity', fontsize=15)
            ax.set_ylabel('Whole Circuit Fidelity', fontsize=18)
            # ax.plot(x, np.array(f_lEC[start:end]), label='logical 1EC')
            ax.legend(fontsize=size)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            for label in (ax.get_xticklabels()):
                label.set_fontsize(13)
            # Set tick font size
            for label in (ax.get_yticklabels()):
                label.set_fontsize(13)

        elif x_points == 'two':
            start = 14
            x = f_worst_2q[start:end]
            fig, ax = plt.subplots()
            ax.set_xlabel('Worst-Case Entangling Gate Fidelity', fontsize=15)
            ax.set_ylabel('Fidelity', fontsize=18)
            ax.plot(x, np.array(f_tSPS[start:end]), label='traditional SPS', color='red')
            # ax.plot(x, np.array(f_lEC[start:end]), label='logical 1EC')
            yhat = savitzky_golay(f_lEC[start:end], 41, 3)  # window size 51, polynomial order 3
            yerr = calc_error(yhat, f_lEC[start:end], window)
            plt.fill_between(x, yhat - yerr, yhat + yerr, facecolor="green", edgecolor="none", alpha=.3)
            ax.plot(x, yhat, label='logical 1EC', color='green')
            ax.plot(x, np.array(f_lLPS[start:end]), label='logical 1LPS', color='blue')
            for label in (ax.get_xticklabels()):
                label.set_fontsize(13)
            # Set tick font size
            for label in (ax.get_yticklabels()):
                label.set_fontsize(13)
            ax.legend(fontsize=size)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        else:
            x = T2_list[start:end]
            fig, ax = plt.subplots()
            plt.xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
            ax.set_ylabel('Fidelity', fontsize=18)
            ax.plot(x, np.array(f_tSPS[start:end]), label='traditional SPS', color='red')
            # ax.plot(x, np.array(f_lEC[start:end]), label='logical 1EC')
            yhat = savitzky_golay(f_lEC[start:end], 41, 3)  # window size 51, polynomial order 3
            yerr = calc_error(yhat,f_lEC[start:end],window)
            plt.fill_between(x, yhat - yerr, yhat + yerr,facecolor="green", edgecolor="none", alpha=.3)
            ax.plot(x,yhat,label='logical 1EC', color='green')
            ax.plot(x, np.array(f_lLPS[start:end]), label='logical 1LPS', color='blue')
            ax.legend(fontsize=size)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # Set tick font size
            for label in (ax.get_xticklabels()):
                label.set_fontsize(16)
            # Set tick font size
            for label in (ax.get_yticklabels()):
                label.set_fontsize(13)
            ax.set_xscale('log')

        if save:
            plt.savefig('images\\Kitaev_paper')
        plt.show()

    else:
        fig, ax = plt.subplots()
        if x_points == 'single':
            x = f_worst_1q
            ax.set_xlabel('Worst-Case Single Gate Fidelity', fontsize=15)
        elif x_points == 'two':
            x = f_worst_2q
            ax.set_xlabel('Worst-Case Entangling Gate Fidelity', fontsize=15)
        else:
            x = T2_list_main
            plt.xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
            ax.set_xscale('log')
        x = f_worst_1q
        ax.plot(x, np.array(f_lLPS), label='logical ancilla SPS+LPS', color='blue')
        ax.plot(x, np.array(f_tSPS), label='physical ancilla SPS', color='red')
        ax.set_ylabel('Whole Circuit Fidelity', fontsize=18)
        ax.scatter(x, np.array(f_lEC), label='logical ancilla SPS+EC', color='green')
        ax.legend(fontsize=size)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        for label in (ax.get_xticklabels()):
            label.set_fontsize(13)
        # Set tick font size
        for label in (ax.get_yticklabels()):
            label.set_fontsize(13)

        plt.show()


def plot_Kitaev_infidelity(x_points='two',window=10,save=False):
    start = 1
    end = 120

    if x_points == 'single':
        if plot_data_from_2021:
            start = 10
            end = 120
        else:
            start = 0
            end = len(T2_list_main)
        fig, ax = plt.subplots()
        x = 1-np.array(f_worst_1q[start:end])
        ax.set_xlabel('1 - Fidelity(single qubit gate)', fontsize=15)
        ax.set_ylabel('1 - Fidelity(Kitaev QPE circuit)', fontsize=15)
        ax.plot(x, 1-np.array(f_tSPS[start:end]), label='traditional SPS', color='red')
        ax.plot(x, 1-np.array(f_lLPS[start:end]), label='logical 1LPS', color='blue')
        # yhat = savitzky_golay(f_lEC[start:end], 41, 3)
        # ax.scatter(x, 1-yhat, label='logical 1EC', color='green')
        ax.plot(fontsize=size)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        for label in (ax.get_xticklabels()):
            label.set_fontsize(13)
        # Set tick font size
        for label in (ax.get_yticklabels()):
            label.set_fontsize(13)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()

    elif x_points == 'two':
        print("not implemented yet)")

    else:
        print("not implemented yet)")

    if save:
        plt.savefig('images\\Kitaev_2_paper')
    plt.show()


def plot_x_dependence(save=False):

    if plot_data_from_2021:
        start = 10
        end = 120
    else:
        start = 0
        end = len(T2_list_main)
    P_l = 1 - (np.array(f_lLPS[start:end])) ** 2  # error probability for the whole logical circuit
    P_t = 1 - (np.array(f_tSPS[start:end])) ** 2  # error probability for the whole algorithm

    def func2(x, a, b):
        return a * x ** 2 + b
    def func3(x, a, b):
        return a * x ** 3 + b
    def func4(x, a, b):
        return a * x ** 4 + b

    popt2, pcov2 = curve_fit(func2, P_t[start:end], P_l[start:end])
    popt3, pcov3 = curve_fit(func3, P_t[start:end], P_l[start:end])
    popt4, pcov4 = curve_fit(func4, P_t[start:end], P_l[start:end])


    fig,axes = plt.subplots(1)


    axes.set_xlabel('$P_{err}$ - physical', fontsize=18)
    axes.plot(np.array(P_t)[start:end], func2(P_t, *popt2)[start:end],
             label='fit $y=ax^2$, $a=' + str(int(round(popt2[0]))) + '$', color='black')
    axes.plot(np.array(P_t)[start:end], func3(P_t, *popt3)[start:end],
                 label='fit $y=ax^3$, $a=' + str(int(round(popt3[0]))) + '$', color='blue')
    axes.plot(np.array(P_t)[start:end], func4(P_t, *popt4)[start:end],
                 label='fit $y=ax^4$, $a=' + str(int(round(popt4[0]))) + '$', color='black')
    axes.scatter(np.array(P_t)[start:end], np.array(P_l)[start:end], label='data')
    axes.plot(np.array(P_t)[start:end], np.array(P_t)[start:end], color='r', label='x=y')
    axes.set_ylabel('$P_{err}$ - logical', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    set_ticks_size(axes,12,13)
    plt.legend()

    if save:
        plt.savefig('images\\scalingError.png')

    plt.show()


def plot_Kitaev_Resource(x_points='two',save=False):

    c = ((2 - np.sqrt(2)) / 4) ** 2
    if plot_data_from_2021:
        start = 0
        end = 120
    else:
        start = 0
        end = len(T2_list_main)
    plt.ylabel('Minimum Number of Trials', fontsize=18)
    if K:
        ind = 6
    else:
        ind = 10

    if x_points == 'two':
        x = f_worst_2q
        plt.xlabel('Worst-Case Entangling Gate Fidelity', fontsize=18)
    else:
        x = T2_list
        plt.xlabel('$T_{2}$  $[T_{gate}]$', fontsize=18)
        plt.xscale('log')
    ax = plt.gca()
    # plt.title('With LI')
    for epsilon in [1e-1, 1e-2, 1e-3]:
        color = next(ax._get_lines.prop_cycler)['color']
        N_min_t = np.log(2 / epsilon) / (
                    2 * (1 - np.array(li_tSPS[start:end])) * (c - ((np.array(D_tSPS[start:end])) ** 2) / 2))
        N_min_t[N_min_t < 0] = 0
        N_min_t = [math.ceil(N_min_t[i]) for i in range(len(N_min_t))]
        start_t = len(N_min_t) - N_min_t[::-1].index(0)

        N_min_l = np.log(2 / epsilon) / (
                    2 * (1 - np.array(li_lLPS[start:end])) * (c - ((np.array(D_lLPS[start:end])) ** 2) / 2))
        N_min_l[N_min_l < 0] = 0
        N_min_l = [math.ceil(N_min_l[i]) for i in range(len(N_min_l))]
        start_l = len(N_min_l) - N_min_l[::-1].index(0)

        # N_min_lEC = np.log(2 / epsilon) / (
        #             2 * (1 - np.array(li_lEC[start:end])) * (c - ((np.array(D_lEC[start:end])) ** 2) / 2))
        # N_min_lEC[N_min_lEC < 0] = 0
        # N_min_lEC = [math.ceil(N_min_lEC[i]) for i in range(len(N_min_lEC))]
        # start_lEC = len(N_min_lEC) - N_min_lEC[::-1].index(0)

        plt.plot(x[start_t:end], N_min_t[start_t:end], label='$\epsilon = ' + str(epsilon) + '$', color=color)
        plt.plot(x[start_l:end], N_min_l[start_l:end], '--', color=color)
        # plt.scatter(x[start_lEC:end], N_min_lEC[start_lEC:end], color=color)

    plt.plot(np.NaN, np.NaN, '--', color='black', label='logical ancilla SPS+LPS')
    # plt.plot(np.NaN, np.NaN, '.', color='black', label='logical ancilla SPS+EC')
    plt.plot(np.NaN, np.NaN, color='black', label='physical ancilla SPS')
    plt.yscale('log')
    set_ticks_size(ax,12,13)
    # plt.xlim((50,1000))
    # plt.ylim((50,1000))
    plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        plt.savefig('images\\KitaevNonAccelerated.png')

    plt.show()


if (plot_data_new or plot_data_from_2021):
    plot_Kitaev(x_points='single',save=True)
    # plot_Kitaev_infidelity(x_points='single',save=False)
    # plot_x_dependence(save=False)
    # plot_Kitaev_Resource(x_points='single',save=False)

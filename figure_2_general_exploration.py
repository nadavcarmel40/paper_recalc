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
import matplotlib.pyplot as plt
import os
import time
from matplotlib.colors import DivergingNorm
import scipy.interpolate as interpolate
from scipy.signal import convolve2d
from simulators.Utils import map_decohere_to_worst_gate_fidelity

print('all packages imported')


#########################################################################################
########################## constants to control this file ###############################
#########################################################################################

# generate data. default should be ng = 200, nT2 = 50, perfect_syndrome_extraction = False
ng = 200
nT2 = 50
perfect_syndrome_extraction = False
name = 'recreate_paper'

T2_list = list(set([int(list(np.geomspace(10,10000,nT2, endpoint=True))[i]) for i in range(nT2)]))
T2_list.sort()

generate_data = True

plot_data_new = True

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

def traditionalCNOT(reg,control=0,target=1):
    """
    runs regular cnot with control and target qubits
    """
    # print(reg.state)
    reg.run([[('H', 1, None, None)]])
    reg.run([[('CZ', control,target,None)]])
    reg.run([[('H', 1, None, None)]])

def logicalCNOT(reg):
    """
    assumes control is logical 5 qubit in indexes 0-4
    """
    reg.run([[('H', 5, None, None)]])
    # print(debugLogical(reg.state))
    reg.run([[('CZ', 5, 0, None)],[('CZ', 5, 1, None)],[('CZ', 5, 2, None)],[('CZ', 5, 3, None)],[('CZ', 5, 4, None)]])
    reg.run([[('H', 5, None, None)]])

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
    reg.run([[('H',5,None,None)]])

    # extract syndrome
    syndrome = ''

    Ps = []

    def messureANDcollapse(LPS):
        if LPS:
            project = reg.qI

            ########## do logical post selection by forcing each generator to measure trivially, ##########
            ##########         and dont normalize the state to gather lost information           ##########

            project *= (reg.qI+reg.Sz[5])/2 #IIIII|0><0|
            applyOperator(reg,noisy,project) #    sensor
            Ps.append(reg.state.tr())
            reg.state = reg.state/reg.state.tr()
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
    reg.run([[('H',5,None,None)]])

    if (noisy and perfect):
        reg.setError(dephase=dephase,amplitude_damp=amp)

    return Ps

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

def CNOTExplore_generate_data(ng=200, nT2=50, perfect_syndrome_extraction = False):
    """
    Creates and saves all states after applying every CNOT for each algorithm.
    :param ng: maximum number of consecutive CNOT gates
    :param nT2: number of points for T2 in the range of 10 and 10000 Tgate, with numpy's geomspace.
    :param perfect_syndrome_extraction: True for perfect (not noisy) syndrome extraction. else - False. default is False.
    :return: ((f_t,d_t,None), (f_l, d_l, li_l)) for fidelity, distance and lost information of
     traditional, logical algorithms, assuming syndrome extraction is according to var perfect_syndrome_extraction.
    """

    Tgate = 1
    num_gates = list(set([int(list(np.linspace(1,ng,ng, endpoint=True))[i]) for i in range(ng)]))
    num_gates.sort()
    T2_list = list(set([int(list(np.geomspace(10,10000,nT2, endpoint=True))[i]) for i in range(nT2)]))
    T2_list.sort()
    revive_run_from_index = 0
    T2_list = T2_list[revive_run_from_index:]
    T1 = 1
    dt = Tgate/20
    start = time.time()

    for w,T2 in enumerate(T2_list):
        end = time.time()
        print('started T2 ='+str(T2)+' in time '+str(end-start))
        path = os.getcwd()
        if perfect_syndrome_extraction:
            folder = os.path.join(path, 'data\\CNOTExplore\\PerfectSyndromeExtraction\\'+name+'\\'+str(T2))
        else:
            folder = os.path.join(path, 'data\\CNOTExplore\\NoisySyndromeExtraction\\'+name+'\\' + str(T2))
        try:
            os.makedirs(folder)
        except:
            pass
        tradCNOTreg = InCoherentQuantumRegister(2, tensor([((basis(2,0)+basis(2,1))/np.sqrt(2))*((basis(2,0)+basis(2,1))/np.sqrt(2)).dag(),((basis(2,0)+basis(2,1))/np.sqrt(2))*((basis(2,0)+basis(2,1))/np.sqrt(2)).dag()]),T1,T2,dt=dt,Tgate=Tgate)
        tradCNOTreg.setError(dephase=True, amplitude_damp=False)
        idealCNOTreg = InCoherentQuantumRegister(2, tensor([((basis(2,0)+basis(2,1))/np.sqrt(2))*((basis(2,0)+basis(2,1))/np.sqrt(2)).dag(),((basis(2,0)+basis(2,1))/np.sqrt(2))*((basis(2,0)+basis(2,1))/np.sqrt(2)).dag()]),T1,T2,dt=dt,Tgate=Tgate)
        idealCNOTreg.setError(dephase=False, amplitude_damp=False)
        logicalCNOTreg = InCoherentQuantumRegister(6, tensor([logic_plus_dm,((basis(2,0)+basis(2,1))/np.sqrt(2))*((basis(2,0)+basis(2,1))/np.sqrt(2)).dag()]),T1,T2,dt=dt,Tgate=Tgate)
        logicalCNOTreg.setError(dephase=True, amplitude_damp=False)

        Ps_ft = []
        Ps_l = []
        k_prev = 0
        for q,k in enumerate(num_gates):
            end = time.time()
            print('started num_gates='+str(k)+' in time '+str(end-start))
            # apply CNOTs
            for i in range(k-k_prev):
                traditionalCNOT(tradCNOTreg)
                traditionalCNOT(idealCNOTreg)
                logicalCNOT(logicalCNOTreg)

                np.save(folder+'\\l_state_K_'+str(k_prev+i), logicalCNOTreg.state)
                np.save(folder+'\\t_state_K_'+str(k_prev+i), tradCNOTreg.state)
                np.save(folder+'\\i_state_K_'+str(k_prev+i), idealCNOTreg.state)

            logicalstate = logicalCNOTreg.state

            Ps_l += EC_for_LogicalRegister(logicalCNOTreg,True,perfect=perfect_syndrome_extraction)
            np.save(folder+'\\Ps_ft'+str(k), Ps_ft)
            np.save(folder+'\\Ps_l'+str(k), Ps_l)

            k_prev = k

            logicalCNOTreg.update(logicalstate)

    return None

def getData(ng=200,nT2=50,perfect_syndrome_extraction=False):
    """
    applies perfect syndrome extraction on the previously saved states of after CNOTs. saves the final Fidelities,
     Distances and Lost Information inside tensors in the folder 'PerfectSyndromeExtraction'.
    :param ng: maximum number of gates
    :param nT2: number of points for T2 in the range of 10 and 10000 Tgate, with numpy's geomspace.
    :param perfect_syndrome_extraction: True for perfect (not noisy) syndrome extraction. else - False. default is False.
    :return: ((f_t,d_t,None), (f_l, d_l, li_l)) for fidelity, distance and lost information of
     traditional, logical, assuming syndrome extraction is perfect.
    """

    Tgate = 1
    num_gates = list(set([int(list(np.linspace(1,ng,ng, endpoint=True))[i]) for i in range(ng)]))
    num_gates.sort()
    T2_list = list(set([int(list(np.geomspace(10,10000,nT2, endpoint=True))[i]) for i in range(nT2)]))
    T2_list.sort()
    T2_list = T2_list[:]
    T1 = 1
    dt = Tgate/20
    start = time.time()
    f_t = np.zeros((int(len(T2_list)),int(len(num_gates))))
    f_l = np.zeros((int(len(T2_list)),int(len(num_gates))))
    li_l = np.zeros((int(len(T2_list)),int(len(num_gates))))
    d_t = np.zeros((int(len(T2_list)),int(len(num_gates))))
    d_l = np.zeros((int(len(T2_list)),int(len(num_gates))))

    path = os.getcwd()
    if perfect_syndrome_extraction:
        folder1 = os.path.join(path, 'data\\CNOTExplore\\PerfectSyndromeExtraction\\'+name)
    else:
        folder1 = os.path.join(path, 'data\\CNOTExplore\\NoisySyndromeExtraction\\'+name)

    for w,T2 in enumerate(T2_list):
        end = time.time()
        print('started T2 ='+str(T2)+' in time '+str(end-start))
        path = os.getcwd()
        if perfect_syndrome_extraction:
            folder = os.path.join(path, 'data\\CNOTExplore\\PerfectSyndromeExtraction\\'+name+'\\' + str(T2))
        else:
            folder = os.path.join(path, 'data\\CNOTExplore\\NoisySyndromeExtraction\\'+name+'\\' + str(T2))

        k_prev = 0

        tradCNOTreg = InCoherentQuantumRegister(2, tensor([fock_dm(2,0) for i in range(2)]),T1,T2,dt=dt,Tgate=Tgate)
        tradCNOTreg.setError(dephase=False, amplitude_damp=False)
        idealCNOTreg = InCoherentQuantumRegister(2, tensor([fock_dm(2,0) for i in range(2)]),T1,T2,dt=dt,Tgate=Tgate)
        idealCNOTreg.setError(dephase=False, amplitude_damp=False)
        logicalCNOTreg = InCoherentQuantumRegister(6, tensor([fock_dm(2,0) for i in range(6)]),T1,T2,dt=dt,Tgate=Tgate)
        logicalCNOTreg.setError(dephase=False, amplitude_damp=False)

        for q,k in enumerate(num_gates):
            end = time.time()
            if q % 49 == 0:
                print('started num_gates='+str(k)+' in time '+str(end-start))
            # load states
            for i in range(k-k_prev):

                l_s = Qobj(np.load(folder+'\\l_state_K_'+str(k_prev+i)+'.npy'), dims = [[2 for i in range(6)],[2 for i in range(6)]])
                t_s = Qobj(np.load(folder+'\\t_state_K_'+str(k_prev+i)+'.npy'), dims = [[2 for i in range(2)],[2 for i in range(2)]])
                i_s = Qobj(np.load(folder+'\\i_state_K_'+str(k_prev+i)+'.npy'), dims = [[2 for i in range(2)],[2 for i in range(2)]])

            Ps_l = list(np.load(folder+'\\Ps_l'+str(k)+'.npy'))

            tradCNOTreg.update(t_s)
            idealCNOTreg.update(i_s)
            logicalCNOTreg.update(l_s)

            Ps_l += EC_for_LogicalRegister(logicalCNOTreg,True)

            k_prev = k

            li_l_k = 0
            for i in range(len(Ps_l)):
                li_l_iteration = 1
                for k in range(i):
                    li_l_iteration *= Ps_l[k]
                li_l_iteration *= (1-Ps_l[i])
                li_l_k += li_l_iteration
            li_l[w,q] = li_l_k

            l_state = debugLogical(logicalCNOTreg.state/logicalCNOTreg.state.tr())

            f_t[w,q] = fidelity(i_s,t_s/t_s.tr())
            f_l[w,q] = fidelity(i_s,l_state)

            ideal = i_s
            trad = t_s

            d_t[w,q] = np.real(np.sqrt((ideal[0,0]-trad[0,0])**2+(ideal[1,1]-trad[1,1])**2+(ideal[2,2]-trad[2,2])**2+(ideal[3,3]-trad[3,3])**2))
            d_l[w,q] = np.real(np.sqrt((ideal[0,0]-l_state[0,0])**2+(ideal[1,1]-l_state[1,1])**2+(ideal[2,2]-l_state[2,2])**2+(ideal[3,3]-l_state[3,3])**2))

            np.save(folder1+'\\f_t',f_t)
            np.save(folder1+'\\d_t',d_t)
            np.save(folder1+'\\f_l',f_l)
            np.save(folder1+'\\d_l',d_l)
            np.save(folder1+'\\li_l',li_l)

    return ((f_t,d_t,None), (f_l, d_l, li_l))

if generate_data:
    CNOTExplore_generate_data(ng=ng, nT2=nT2, perfect_syndrome_extraction=perfect_syndrome_extraction)
    t, l = getData(ng,nT2,perfect_syndrome_extraction)


#########################################################################################
########### load data from folder for data generated now  ###############################
#########################################################################################

def load_CNOTExplore_new_data(perfectSyndromeExtraction,ng=200,nT2=50):
    path = os.getcwd()
    if perfectSyndromeExtraction:
        folder = os.path.join(path, 'data\\CNOTExplore\\PerfectSyndromeExtraction')
    else:
        folder = os.path.join(path, 'data\\CNOTExplore\\NoisySyndromeExtraction')
    t = np.load(folder+'\\f_t.npy'), np.load(folder+'\\d_t.npy'), None
    l = np.load(folder+'\\f_l.npy'), np.load(folder+'\\d_l.npy'), np.load(folder+'\\li_l.npy')
    num_gates = list(set([int(list(np.linspace(1, ng, ng, endpoint=True))[i]) for i in range(ng)]))
    num_gates.sort()
    T2_list = list(set([int(list(np.geomspace(10, 10000, nT2, endpoint=True))[i]) for i in range(nT2)]))
    T2_list.sort()
    return t,l,num_gates,T2_list

if plot_data_new:
    t,l,num_gates,T2_list = load_CNOTExplore_new_data(perfect_syndrome_extraction,ng=ng,nT2=nT2)
    f_worst_1q_CNOTs = map_decohere_to_worst_gate_fidelity(T2_list,1,Tgate=1,decohere="2",save=False)
    f_worst_2q_CNOTs = map_decohere_to_worst_gate_fidelity(T2_list,2,Tgate=1,decohere="2",save=False)


#########################################################################################
########### load data from folder for data generated in 2021  ###########################
#########################################################################################

def load_CNOTExplore(perfectSyndromeExtraction):

    if perfectSyndromeExtraction:
        t = np.load('data_2021\\CNOTExplore\\PerfectSyndromeExtraction\\f_t.npy'), np.load('data_2021\\CNOTExplore\\PerfectSyndromeExtraction\\d_t.npy'), None
        l = np.load('data_2021\\CNOTExplore\\PerfectSyndromeExtraction\\f_l.npy'), np.load('data_2021\\CNOTExplore\\PerfectSyndromeExtraction\\d_l.npy'), np.load('data_2021\\CNOTExplore\\PerfectSyndromeExtraction\\li_l.npy')
        ft = np.load('data_2021\\CNOTExplore\\PerfectSyndromeExtraction\\f_ft.npy'), np.load('data_2021\\CNOTExplore\\PerfectSyndromeExtraction\\d_ft.npy'), np.load('data_2021\\CNOTExplore\\PerfectSyndromeExtraction\\li_ft.npy')
        start = 0
        end = 50
    else:
        t = np.concatenate((np.load('data_2021\\CNOTExplore\\f_t.npy'),np.load('data_2021\\CNOTExplore\\f_t2.npy')[:21,:])),np.concatenate((np.load('data_2021\\CNOTExplore\\d_t.npy'),np.load('data_2021\\CNOTExplore\\d_t2.npy')[:21,:])),None
        l = np.concatenate((np.load('data_2021\\CNOTExplore\\f_l.npy'),np.load('data_2021\\CNOTExplore\\f_l2.npy')[:21,:])),np.concatenate((np.load('data_2021\\CNOTExplore\\d_l.npy'),np.load('data_2021\\CNOTExplore\\d_l2.npy')[:21,:])), np.concatenate((np.load('data_2021\\CNOTExplore\\li_l.npy'),np.load('data_2021\\CNOTExplore\\li_l2.npy')[:21,:]))
        ft = np.concatenate((np.load('data_2021\\CNOTExplore\\f_ft.npy'),np.load('data_2021\\CNOTExplore\\f_ft2.npy')[:21,:])),np.concatenate((np.load('data_2021\\CNOTExplore\\d_ft.npy'),np.load('data_2021\\CNOTExplore\\d_ft2.npy')[:21,:])), np.concatenate((np.load('data_2021\\CNOTExplore\\li_ft.npy'),np.load('data_2021\\CNOTExplore\\li_ft2.npy')[:21,:]))
        start = 0
        end = 46

    ng = 200
    num_gates = list(set([int(list(np.linspace(1, 200, ng, endpoint=True))[i]) for i in range(ng)]))
    num_gates.sort()

    return t,l,ft,start,end,num_gates

def load_WCGF():

    f_worst_1q = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_1_T2.npy')
    f_worst_2q = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_2_T2.npy')
    f_worst_1q_T1 = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_1_T1.npy')
    f_worst_2q_T1 = np.load('data_2021\\WCGF\\fidelities_for_errors_1d_N_2_T1.npy')
    f_worst_1q_CNOTs = np.load('data_2021\\WCGF\\N_1_decohere_mode_2_for_CNOTs_explore.npy')
    f_worst_2q_CNOTs = np.load('data_2021\\WCGF\\N_2_decohere_mode_2_for_CNOTs_explore.npy')
    return f_worst_1q,f_worst_2q,f_worst_1q_T1,f_worst_2q_T1,f_worst_1q_CNOTs,f_worst_2q_CNOTs

if plot_data_from_2021:
    t,l,ft,start_cnot,end_cnot,num_gates = load_CNOTExplore(perfect_syndrome_extraction)
    _,_,_,_,f_worst_1q_CNOTs,f_worst_2q_CNOTs = load_WCGF()

#########################################################################################
############################## plot 2 2d heatmap ########################################
#########################################################################################

def set_ticks_size(ax,x,y):
    # Set tick font size
    for label in (ax.get_xticklabels()):
        label.set_fontsize(x)
    # Set tick font size
    for label in (ax.get_yticklabels()):
        label.set_fontsize(y)

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

    window = 30
    kernel = np.ones((2*window+1,2*window+1))/(2*window+1)**2
    znew = convolve2d(znew,kernel,boundary='symm',mode='same')
    return Xnew,Ynew,znew

def plot_color(X,Y,Z,xlabel,ylabel,title):
    fig,ax = plt.subplots()
    norm = DivergingNorm(vmin=Z.min(), vcenter=0, vmax=Z.max())
    lims = dict(cmap='RdBu')
    plt.pcolormesh(X, Y, Z, shading='flat', norm=norm, **lims)
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel(ylabel,fontsize=15)
    ax.set_title(title,fontsize=14)
    set_ticks_size(ax, 13, 12)
    plt.colorbar(label="Physical Control             Logical Control")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

def plot_CNOTExplore_heatmap(x_points='two',a=0,interpolate=True,save=False):

    if x_points == 'two':
        fid = f_worst_2q_CNOTs
        ylabel = 'Worst-Case Entangling Gate Fidelity'
    elif x_points == 'single':
        fid = f_worst_1q_CNOTs
        ylabel = 'Worst-Case Single Gate Fidelity'
    if plot_data_from_2021:
        if perfect_syndrome_extraction:
            end_cnot = 50
        else:
            end_cnot = 46
        start_ind = end_cnot - 27
    else:
        start_ind = 0
        end_cnot = len(fid)

    X, Y = np.meshgrid(num_gates,fid[start_ind:end_cnot])
    Z = l[a][start_ind:end_cnot,:]-t[a][start_ind:end_cnot,:]
    Xnew,Ynew,Znew = interpolate_2d_grid(X,Y,Z)
    title = "Fidelity: Logical Control minus Physical Control"
    xlabel = "Circuit Depth"
    if interpolate:
        plot_color(Xnew, Ynew, Znew, xlabel,ylabel,title)
    else:
        plot_color(X, Y, Z, xlabel, ylabel, title)
    if save:
        plt.savefig('images\\CNOTS_paper')
    plt.show()

if (plot_data_new or plot_data_from_2021):
    plot_CNOTExplore_heatmap(x_points='single',a=0,interpolate=True,save=False)
    plot_CNOTExplore_heatmap(x_points='single',a=0,interpolate=False)

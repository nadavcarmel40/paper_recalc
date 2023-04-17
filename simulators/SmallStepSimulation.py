from simulators.Constants import *
from simulators.Utils import *
from qutip import *
import scipy as sp
import concurrent

print('hi')


#################################################################################
##############                  register classes                   ##############
#################################################################################

class NoisyQuantumRegister():

    def __init__(self, N, rho_0, T1, T2, w01s=None, T1s=None, T2s=None, pureT2=False, dt=None, Tgate=None):
        """
        starting_states, T1s, T2s can be lists in length num_qubits for the qubits in the registers
        or one number for them all. must all be same type: list or number
        N is the number of qubits in the register
        w01s is a list of qubit energys
        Pdephase and Pdecay are for whole gate, not for small time step
        """
        self.N = N
        self.qI = tensor([qeye(2) for i in range(N)])
        if w01s == None:
            w01s = [freq for i in range(N)]
        # here we define properties useful for the quantum register
        self.times = []
        self.dt = dt
        if dt is None:
            self.dt = T1 / 10000
        self.Tgate = Tgate
        if Tgate is None:
            self.Tgate = 20 * self.dt

        self.state = rho_0

        # deal with the possibility of qubits with different T1,T2
        self.T1 = T1s
        self.T2 = T2s
        self.pureT2 = [0 for i in range(N)]
        if (T1s == None and T2s == None):
            self.T1 = [T1 for i in range(N)]
            self.T2 = [T2 for i in range(N)]
        if pureT2 is False:
            for qubitIndex in range(N):
                self.pureT2[qubitIndex] = 2 * self.T1[qubitIndex] * self.T2[qubitIndex] / (
                        self.T2[qubitIndex] - 2 * self.T1[qubitIndex])
        else:
            for qubitIndex in range(N):
                self.pureT2[qubitIndex] = self.T2[qubitIndex]
                self.T2[qubitIndex] = 1 / (1 / 2 / self.T1[qubitIndex] + 1 / self.T2[qubitIndex])

        self.Pdecay = [0 for i in range(N)]
        self.Pdephase = [0 for i in range(N)]
        for qubitIndex in range(N):
            self.Pdecay[qubitIndex] = 1 - np.exp(-self.dt / self.T1[qubitIndex])
            self.Pdephase[qubitIndex] = 1 - np.exp(-self.dt / self.T2[qubitIndex])

        self.Sx = []
        self.Sy = []
        self.Sz = []

        for qubitIndex in range(N):
            # here we create sigmax, sigmay, sigmaz,Operators for N qubit register
            listSx = [sigmax() if i == qubitIndex else qeye(2) for i in range(N)]
            self.Sx.append(tensor(listSx))
            listSy = [sigmay() if i == qubitIndex else qeye(2) for i in range(N)]
            self.Sy.append(tensor(listSy))
            listSz = [sigmaz() if i == qubitIndex else qeye(2) for i in range(N)]
            self.Sz.append(tensor(listSz))

        # here we create multi-qubit H0 operator
        self.H0 = tensor([hbar * w01s[qubitIndex] / 2 * sigmaz() for qubitIndex in range(N)])
        self.U0 = Qobj(-1j * self.H0.data / hbar * self.dt).expm()

        # decoherence flags
        self.dephase = True
        self.amplitude_damp = True

        # self data
        self.collect_bloch = False
        self.blochs = [Bloch() for qubit in range(self.N)]
        self.bloch3ds = [Bloch3d() for qubit in range(self.N)]

        self.collect = False
        self.history = [[[], [], []] for qubit in range(
            self.N)]  # history of points on bloch sphere. [],[],[] are lists where (xs[i],ys[i],zs[i]) is a point
        self.purities = [[] for qubit in range(self.N + 1)]  # list of purities for each qubit, last is for total purity
        self.__collect()

        self.projected = False

    def __collect(self):
        if self.collect:
            # update history
            for qubit in range(self.N):
                state = self.state.ptrace(qubit)
                x, y, z = get_bloch_from_rho(state.data)
                self.history[qubit][0].append(np.real(x))
                self.history[qubit][1].append(np.real(y))
                self.history[qubit][2].append(np.real(z))
                self.purities[qubit].append(np.trace(state * state))
            self.purities[-1].append(np.trace(self.state * self.state))
            if self.times == []:
                self.times = [0]
            else:
                self.times.append(self.times[-1] + self.dt)

    def __update_bloch(self, start_ind, end_ind):
        # update bloch after gate operation
        if self.collect_bloch:
            for qubit in range(self.N):
                self.blochs[qubit].add_points(
                    [self.history[qubit][0][start_ind:end_ind], self.history[qubit][1][start_ind:end_ind],
                     self.history[qubit][2][start_ind:end_ind]])

                # x = self.history[qubit][0][-1]
                # y = self.history[qubit][1][-1]
                # z = self.history[qubit][2][-1]
                # state = get_rho_from_bloch(x,y,z)
                # self.blochs[qubit].add_annotation(state, str(int(self.times[-1]/1e-9)) + ' ns')

                for i in range(end_ind - start_ind):
                    self.bloch3ds[qubit].add_points(
                        [self.history[qubit][0][start_ind + i], self.history[qubit][1][start_ind + i],
                         self.history[qubit][2][start_ind + i]])
                self.bloch3ds[qubit].point_color = 'b'

    def setError(self, dephase=True, amplitude_damp=True):
        self.dephase = dephase
        self.amplitude_damp = amplitude_damp

    def setCollectData(self, data=False, bloch=False):
        """
        set data collection options
        :param data: bool for collecting history of each qubit
        :param bloch:  bool for collection bloch data for each qubit
        :return: None
        """
        self.collect = data
        self.collect_bloch = bloch

    #################################################################################
    ###############          define actions in one time step          ###############
    #################################################################################

    def __decoherenceTS(self, qubitIndex):
        """
        qubitIndex - qubit to act on
        function does the action of decoherence acting for a small time dt << Ti
        updates the density matrix as qutip Qobj
        returns None
        """
        if ((not self.dephase) and (not self.amplitude_damp)):
            return None
        elif not self.dephase:  # only amplitude damping
            M1 = np.sqrt(1 - self.Pdecay[qubitIndex]) / 2 * (self.qI - self.Sz[qubitIndex]) + 1 / 2 * (
                    self.qI + self.Sz[qubitIndex])
            M2 = np.sqrt(self.Pdecay[qubitIndex]) / 2 * (self.Sx[qubitIndex] + 1j * self.Sy[qubitIndex])
            self.state = M1 * self.state * M1.dag() + M2 * self.state * M2.dag()
            return None
        elif not self.amplitude_damp:  # only dephasing
            self.state = (1 - self.Pdephase[qubitIndex] / 2) * self.state + self.Pdephase[qubitIndex] / 2 * self.Sz[
                qubitIndex] * self.state * \
                         self.Sz[qubitIndex]
            return None
        else:  # amplitude damping and dephasing
            M1 = np.sqrt(1 - self.Pdecay[qubitIndex]) / 2 * (self.qI - self.Sz[qubitIndex]) + 1 / 2 * (
                    self.qI + self.Sz[qubitIndex])
            M2 = np.sqrt(self.Pdecay[qubitIndex]) / 2 * (self.Sx[qubitIndex] + 1j * self.Sy[qubitIndex])
            temp_state = M1 * self.state * M1.dag() + M2 * self.state * M2.dag()
            self.state = (1 - self.Pdephase[qubitIndex] / 2) * temp_state + self.Pdephase[qubitIndex] / 2 * self.Sz[
                qubitIndex] * temp_state * \
                         self.Sz[qubitIndex]
            return None

    #################################################################################
    ###############          define 1-qubit gates                     ###############
    #################################################################################

    # remember to turn lines like "self.state = K1*Qobj(self.state.data)*K1.dag()"
    # to lines like "self.state.data = K1.data*self.state.data*K1.dag().data" to save RAM

    def __applyU(self, U):
        self.state = U * self.state * U.dag()

    def X(self, qubitIndex, phase=np.pi):
        """
        create H matrix for X gate
        """
        return -1j * self.Sx[qubitIndex] * phase / 2

    def Y(self, qubitIndex):
        """
        create H matrix for Y gate
        """
        phase = np.pi
        return -1j * self.Sy[qubitIndex] * phase / 2

    def Z(self, qubitIndex):
        """
        create H matrix for Z gate
        """
        phase = np.pi
        return -1j * self.Sz[qubitIndex] * phase / 2

    def S(self, qubitIndex):
        """
        create H matrix for S gate
        """
        phase = np.pi
        return -1j * self.Sz[qubitIndex] * phase / 4

    def T(self, qubitIndex):
        """
        create H matrix for T gate
        """
        phase = np.pi
        return -1j * self.Sz[qubitIndex] * phase / 8

    def H(self, qubitIndex):
        """
        create H matrix for H gate
        """
        phase = np.pi
        return -1j * phase / 2 * (2 ** (self.N - 1)) * 2 * (self.Sx[qubitIndex] + self.Sz[qubitIndex]).unit()

    #################################################################################
    ###############          define full 2-qubit gates               ################
    #################################################################################

    def __addControlQubit(self, H, q):
        """
        adds control qubit q to gate defined by hamiltonian H
        """
        return (self.qI - self.Sz[q]) / 2 * H

    def iSWAP(self, q1, q2):
        """
        does iSWAP gate between the qubits, small time step
        TGate assumed to be max(Tx,Ty) in default
        ############################################################# GATE DOESNT WORK
        """
        phase = np.pi
        SxSx = self.Sx[q1] * self.Sx[q2]
        SySy = self.Sy[q1] * self.Sy[q2]
        return -1j / 4 * (SxSx + SySy) * phase

    #################################################################################
    ###############          specific for improved QPA               ################
    #################################################################################

    def Rkinv(self, k, q):
        """
        k is 2 or 3
        q is qubit to act on
        returns H for Rk, inverse
        """
        phase = np.pi
        return -1j * phase * 2 / 2 ** k * (self.qI - self.Sz[q]) / 2

    def measure(self, qubitIndices, update=False, showHistogram=False):
        """
        measure qubits qubitIndices, return 0 or 1 string and update self state
        update is true to update the register's state to the change
        """
        state = self.state.ptrace(qubitIndices)
        p = list(np.real(np.diag(state.data.toarray())))

        # produce measurment
        p1 = [0] + p
        F = np.cumsum(p1 / np.sum(p1))
        x = rand()
        i = 1
        while x > F[i]:
            i += 1
        n = math.log2(len(p))
        res = "{0:b}".format(i - 1)
        while len(res) < n:
            res = '0' + res

        # update self state by measurment
        if update:
            states = []
            for i in range(self.N):
                if i not in qubitIndices:
                    states.append(self.state.ptrace(i))
                else:
                    q = qubitIndices.index(i)
                    if res[q] == '0':
                        qstate = produce_start_state('g')
                    else:
                        qstate = produce_start_state('e')
                    states.append(qstate)
            self.state = tensor(states)

        return res, p

    def powU2(self, U, k):
        """
        return U**(2**k)
        """
        i = 0
        result = U
        while i != k:
            result += result
            i += 1
        return 1j * result

    def update(self, rho, deletehistory=True):
        """
        updates self.state to be rho
        """
        if self.state.data.shape == rho.data.shape:
            self.state = rho
        if deletehistory:
            self.blochs = [Bloch() for qubit in range(self.N)]
            self.bloch3ds = [Bloch3d() for qubit in range(self.N)]

            self.history = [[[], [], []] for qubit in range(
                self.N)]  # history of points on bloch sphere. [],[],[] are lists where (xs[i],ys[i],zs[i]) is a point
            self.purities = [[] for qubit in
                             range(self.N + 1)]  # list of purities for each qubit, last is for total
            self.times = []
            self.__collect()


        else:
            print("rho is not of the right dims")

    #################################################################################
    ##############          do a list of gates on the qubit          ################
    #################################################################################

    def run(self, commands, showHistogram=False):
        """
        do a list of gates on the qubit.
        commands - list of commands that is a list of lists of lists, each list is of the form
            [('c',q1,q2), ('c',q1,q2), ...] where 'c' is a command and q1,q2 are the qubits involved
            (q2 optional), and each list as in description represents one time cut by gates
            'c' is a command from the set {'i','x','y','z','s','t','h','CZ','CX', 'iSWAP', 'R2inv', 'R3inv', {special commands}}
            special commands - [('i', int)] = 'i' is identity operator, int is number of Tgates to wait
                                [('CU k', k, -1, U)] = controlld-U operation, U is generator of the measured Operator in
                                 QPE circuit, k is it's power
                                [('applyOperator', U)] - U is operator to apply directly to register, for example -
                                  projection operator.
        returns - list of measurments as [(qi,m)..] where qi is the measured qubit and m is the
        measurment result.
        """

        for timecut in commands:
            H = self.H0 * self.dt / self.Tgate
            num_gates = 1  # used for identity operator
            self.projected = False
            for commandrep in timecut:
                command = commandrep[0]
                q1 = commandrep[1]
                if len(commandrep) == 3:
                    q2 = commandrep[2]

                # create the GATE action for small time step
                if command == 'i':
                    # q1 is now int number of Tgates to wait
                    num_gates = q1
                elif command == 'x':
                    H += self.X(q1) * self.dt / self.Tgate
                elif command == 'y':
                    H += self.Y(q1) * self.dt / self.Tgate
                elif command == 'z':
                    H += self.Z(q1) * self.dt / self.Tgate
                elif command == 's':
                    H += self.S(q1) * self.dt / self.Tgate
                elif command == 't':
                    H += self.T(q1) * self.dt / self.Tgate
                elif command == 'h':
                    H += self.H(q1) * self.dt / self.Tgate
                elif command == 'CZ':
                    H += self.__addControlQubit(self.Z(q2) * self.dt / self.Tgate, q1)
                elif command == 'iSWAP':
                    H += self.iSWAP(q1, q2) * self.dt / self.Tgate
                elif command == 'CX':
                    H += self.__addControlQubit(self.X(q2) * self.dt / self.Tgate, q1)
                elif command == 'R2inv':
                    H += self.__addControlQubit(self.Rkinv(2, q2) * self.dt / self.Tgate, q1)
                elif command == 'R3inv':
                    H += self.__addControlQubit(self.Rkinv(3, q2) * self.dt / self.Tgate, q1)
                elif command == 'm':
                    measurments, p = self.measure(q1, showHistogram=showHistogram)
                elif command.split()[0] == 'CU':
                    u = commandrep[-1]
                    k = int(command.split()[1])
                    FFed = commandrep[2]
                    n = int(math.log2(u.data.shape[0]))
                    U = tensor([qeye(2) if self.N - n - i - 1 >= 0 else u for i in range(self.N - n + 1)])
                    if FFed:
                        H += self.__addControlQubit(self.powU2(U, k) * self.dt / self.Tgate, q1)
                    if not FFed:
                        H += self.__addControlQubit(self.powU2(U, 0) * self.dt / self.Tgate, q1)  # as if acts only once
                        num_gates = int(2 ** k)  # for 2**k times
                elif command == 'applyOperator':
                    self.__applyU(q1)  # q1 is now operator, this step is done entirely with no small steps
                    self.projected = True  # delete if operator is not projection - assumed to take 0 time

            U = Qobj(H).expm()

            start_ind = len(self.history[0][0])

            if not self.projected:

                # apply GATE action with decoherence on every Qubit
                for i in range(int(num_gates * self.Tgate / self.dt)):
                    self.__applyU(U)
                    for qubitIndex in range(self.N):
                        self.__decoherenceTS(qubitIndex)
                    self.__collect()  # collect data if flag is True

            end_ind = len(self.history[0][0])
            self.__update_bloch(start_ind, end_ind)

        return None


class InCoherentQuantumRegister():

    def __init__(self, N, rho_0, T1, T2, w01s=None, T1s=None, T2s=None, dt=None, Tgate=None):
        """
        starting_states, T1s, T2s can be lists in length num_qubits for the qubits in the registers
        or one number for them all. must all be same type: list or number
        N is the number of qubits in the register
        w01s is a list of qubit energys
        Pdephase and Pdecay are for whole gate, not for small time step
        """
        self.N = N
        self.qI = tensor([qeye(2) for i in range(N)])
        if w01s == None:
            w01s = [freq for i in range(N)]
        # here we define properties useful for the quantum register
        self.times = []
        self.dt = dt
        if dt is None:
            self.dt = T1 / 10000
        self.Tgate = Tgate
        if Tgate is None:
            self.Tgate = 20 * self.dt

        self.state = rho_0

        # deal with the possibility of qubits with different T1,T2
        self.T1 = T1s
        self.T2 = T2s
        self.pureT2 = [0 for i in range(N)]
        if (T1s == None and T2s == None):
            self.T1 = [T1 for i in range(N)]
            self.T2 = [T2 for i in range(N)]

        self.Pdecay = [0 for i in range(N)]
        self.Pdephase = [0 for i in range(N)]
        for qubitIndex in range(N):
            self.Pdecay[qubitIndex] = 1 - np.exp(-self.dt / self.T1[qubitIndex])
            self.Pdephase[qubitIndex] = 1 - np.exp(-self.dt / self.T2[qubitIndex])

        self.Sx = []
        self.Sy = []
        self.Sz = []

        for qubitIndex in range(N):
            # here we create sigmax, sigmay, sigmaz,Operators for N qubit register
            listSx = [sigmax() if i == qubitIndex else qeye(2) for i in range(N)]
            self.Sx.append(tensor(listSx))
            listSy = [sigmay() if i == qubitIndex else qeye(2) for i in range(N)]
            self.Sy.append(tensor(listSy))
            listSz = [sigmaz() if i == qubitIndex else qeye(2) for i in range(N)]
            self.Sz.append(tensor(listSz))

        # here we create multi-qubit H0 operator
        self.H0 = tensor([hbar * w01s[qubitIndex] / 2 * sigmaz() for qubitIndex in range(N)])
        self.U0 = Qobj(-1j * self.H0.data / hbar * self.dt).expm()

        # decoherence flags
        self.dephase = True
        self.amplitude_damp = True

        # self data
        self.collect_bloch = False
        self.blochs = [Bloch() for qubit in range(self.N)]
        self.bloch3ds = [Bloch3d() for qubit in range(self.N)]

        self.collect = False
        self.history = [[[], [], []] for qubit in range(
            self.N)]  # history of points on bloch sphere. [],[],[] are lists where (xs[i],ys[i],zs[i]) is a point
        self.purities = [[] for qubit in range(self.N + 1)]  # list of purities for each qubit, last is for total purity
        self.__collect()

    def __collect(self):
        if self.collect:
            # update history
            for qubit in range(self.N):
                if self.N>1:
                    state = self.state.ptrace(qubit)
                else:
                    state = self.state
                x, y, z = get_bloch_from_rho(state.data)
                self.history[qubit][0].append(np.real(x))
                self.history[qubit][1].append(np.real(y))
                self.history[qubit][2].append(np.real(z))
                self.purities[qubit].append(np.trace(state * state))
            self.purities[-1].append(np.trace(self.state * self.state))
            if self.times == []:
                self.times = [0]
            else:
                self.times.append(self.times[-1] + self.dt)

    def __update_bloch(self, start_ind, end_ind):
        # update bloch after gate operation
        if self.collect_bloch:
            for qubit in range(self.N):
                self.blochs[qubit].add_points(
                    [self.history[qubit][0][start_ind:end_ind], self.history[qubit][1][start_ind:end_ind],
                     self.history[qubit][2][start_ind:end_ind]])

                # x = self.history[qubit][0][-1]
                # y = self.history[qubit][1][-1]
                # z = self.history[qubit][2][-1]
                # state = get_rho_from_bloch(x,y,z)
                # self.blochs[qubit].add_annotation(state, str(int(self.times[-1]/1e-9)) + ' ns')

                for i in range(end_ind - start_ind):
                    self.bloch3ds[qubit].add_points(
                        [self.history[qubit][0][start_ind + i], self.history[qubit][1][start_ind + i],
                         self.history[qubit][2][start_ind + i]])
                self.bloch3ds[qubit].point_color = 'b'

    def setError(self, dephase=True, amplitude_damp=True, T1s=None, T2s=None):
        self.dephase = dephase
        self.amplitude_damp = amplitude_damp
        if T1s != None:
            self.T1 = T1s
            for qubitIndex in range(self.N):
                self.Pdecay[qubitIndex] = 1 - np.exp(-self.dt / self.T1[qubitIndex])
        if T2s != None:
            self.T2 = T2s
            for qubitIndex in range(self.N):
                self.Pdephase[qubitIndex] = 1 - np.exp(-self.dt / self.T2[qubitIndex])

    def setCollectData(self, data=False, bloch=False):
        """
        set data collection options
        :param data: bool for collecting history of each qubit
        :param bloch:  bool for collection bloch data for each qubit
        :return: None
        """
        self.collect = data
        self.collect_bloch = bloch

    def __decoherenceTS(self, qubitIndex):
        """
        qubitIndex - qubit to act on
        function does the action of decoherence acting for a small time dt << Ti
        updates the density matrix as qutip Qobj
        returns None
        """
        if ((not self.dephase) and (not self.amplitude_damp)):
            return None
        elif not self.dephase:  # only amplitude damping
            M1 = np.sqrt(1 - self.Pdecay[qubitIndex]) / 2 * (self.qI - self.Sz[qubitIndex]) + 1 / 2 * (
                    self.qI + self.Sz[qubitIndex])
            M2 = np.sqrt(self.Pdecay[qubitIndex]) / 2 * (self.Sx[qubitIndex] + 1j * self.Sy[qubitIndex])
            self.state = M1 * self.state * M1.dag() + M2 * self.state * M2.dag()
            return None
        elif not self.amplitude_damp:  # only dephasing
            self.state = (1 - self.Pdephase[qubitIndex] / 2) * self.state + self.Pdephase[qubitIndex] / 2 * self.Sz[
                qubitIndex] * self.state * \
                         self.Sz[qubitIndex]
            return None
        else:  # amplitude damping and dephasing
            M1 = np.sqrt(1 - self.Pdecay[qubitIndex]) / 2 * (self.qI - self.Sz[qubitIndex]) + 1 / 2 * (
                    self.qI + self.Sz[qubitIndex])
            M2 = np.sqrt(self.Pdecay[qubitIndex]) / 2 * (self.Sx[qubitIndex] + 1j * self.Sy[qubitIndex])
            temp_state = M1 * self.state * M1.dag() + M2 * self.state * M2.dag()
            self.state = (1 - self.Pdephase[qubitIndex] / 2) * temp_state + \
                         self.Pdephase[qubitIndex] / 2 * self.Sz[qubitIndex] * temp_state * self.Sz[qubitIndex]
            return None

    def update(self, rho, deletehistory=True):
        """
        updates self.state to be rho
        """
        if self.state.data.shape == rho.data.shape:
            self.state = rho
        if deletehistory:
            self.blochs = [Bloch() for qubit in range(self.N)]
            self.bloch3ds = [Bloch3d() for qubit in range(self.N)]

            self.history = [[[], [], []] for qubit in range(
                self.N)]  # history of points on bloch sphere. [],[],[] are lists where (xs[i],ys[i],zs[i]) is a point
            self.purities = [[] for qubit in
                             range(self.N + 1)]  # list of purities for each qubit, last is for total
            self.times = []
            self.__collect()


        else:
            print("rho is not of the right dims")

    def measure(self, qubitIndices, update=False, showHistogram=False):
        """
        measure qubits qubitIndices, return 0 or 1 string and update self state
        update is true to update the register's state to the change
        """
        res = None
        p = None
        results = None

        if len(qubitIndices) == self.N:
            state = self.state
        else:
            state = self.state.ptrace(qubitIndices)
        p = list(np.real(np.diag(state.data.toarray())))

        # produce measurment
        p1 = [0] + p
        F = np.cumsum(p1 / np.sum(p1))
        x = rand()
        i = 1
        while x > F[i]:
            i += 1
        n = math.log2(len(p))
        res = "{0:b}".format(i - 1)
        while len(res) < n:
            res = '0' + res

        ######################################################speedup measurment process - not real #####################################
        res = '00'

        # update self state by measurment
        if update:
            # build operator to project state to measured state
            project = self.qI
            for index, qubitIndex in enumerate(qubitIndices):
                qubit_mes_res = res[index]
                if qubit_mes_res == '0':
                    project *= (self.qI + self.Sz[qubitIndex]) / 2  # I..I|0><0|I..I
                else:
                    project *= (self.qI - self.Sz[qubitIndex]) / 2  # I..I|1><1|I..I
            # apply projection
            self.state = project * self.state * project.dag()
            # self.state = self.state.unit()

        if showHistogram:
            results = {}
            for i, prob in enumerate(p):
                possible_res = "{0:b}".format(i)
                while len(possible_res) < len(qubitIndices):
                    possible_res = '0' + possible_res
                results[possible_res] = prob
            # plt.style.use('dark_background')
            # plot_histogram(results)
            # plt.show()

        return res, p, results

    def run(self, commands):
        """
        the function runs a list of physical commands on the physical qubits.
        :param commands: list of commands that is a list of lists of lists, each list is of the form
            [('c',q1,q2,operator), ('c',q1,q2, operator)] where 'c' is a command and q1,q2 are the qubits involved
            (q2 optional control qubit), and each list as in description represents one time cut by gates.
             'operator' - optioanl, for single qubit gates only or a number of close qubits.
             takes the form of: - None for defined gates (H,X,Y,Z,CNOT,CZ,S,T)
                                - angle for Rotations (Rx,Ry,Rz)
                                - number of gates for (i)
                                - 2x2 matrix for general single qubit operator
                                -2^dx2^d matrix for d-qubit operator - NOT SUPPORTED right now
        :return: None
        """

        for timecut in commands:
            H = self.H0 * self.dt / self.Tgate
            num_gates = 1  # used for identity operator

            for commandrep in timecut:
                name = commandrep[0]
                qubit = commandrep[1]
                control = commandrep[2]
                operator = commandrep[3]

                # create the GATE action for small time step
                dims = [[2] * self.N] * 2
                if name == 'i':
                    # operator is now int number of Tgates to wait
                    num_gates = operator
                elif name == 'X':
                    H += -1j * np.pi / 2 * self.Sx[qubit] * self.dt / self.Tgate
                elif name == 'Y':
                    H += -1j * np.pi / 2 * self.Sy[qubit] * self.dt / self.Tgate
                elif name == 'Z':
                    H += -1j * np.pi / 2 * self.Sz[qubit] * self.dt / self.Tgate
                elif name == 'S':
                    H += -1j * np.pi / 2 * ((self.qI + self.Sz[qubit]) / 2 + 1j * (
                            self.qI - self.Sz[qubit]) / 2) * self.dt / self.Tgate
                elif name == 'T':
                    H += -1j * np.pi / 2 * ((self.qI + self.Sz[qubit]) / 2 + (1 + 1j) / np.sqrt(2) * (
                            self.qI - self.Sz[qubit]) / 2) * self.dt / self.Tgate
                elif name == 'H':
                    H += -1j * np.pi / 2 * 1 / np.sqrt(2) * (self.Sx[qubit] + self.Sz[qubit]) * self.dt / self.Tgate
                elif name == 'CNOT':
                    H += Qobj(sp.linalg.logm(
                        1 / 2 * ((self.qI - self.Sz[control]) * self.Sx[qubit] + (self.qI + self.Sz[control]))),
                        dims=dims) * self.dt / self.Tgate
                elif name == 'CZ':
                    H += Qobj(sp.linalg.logm(
                        1 / 2 * ((self.qI - self.Sz[control]) * self.Sz[qubit] + (self.qI + self.Sz[control]))),
                        dims=dims) * self.dt / self.Tgate
                elif name == 'Rx':
                    theta = operator
                    H += (-1j * theta / 2 * self.Sx[qubit]) * self.dt / self.Tgate
                elif name == 'Ry':
                    theta = operator
                    H += (-1j * theta / 2 * self.Sy[qubit]) * self.dt / self.Tgate
                elif name == 'Rz':
                    theta = operator
                    H += (-1j * theta / 2 * self.Sz[qubit]) * self.dt / self.Tgate
                elif name == 'SingleQubitOperator':
                    a = operator[0, 0]
                    b = operator[0, 1]
                    c = operator[1, 0]
                    d = operator[1, 1]
                    H += Qobj(sp.linalg.logm(
                        (a + d) / 2 * self.qI + (a - d) / 2 * self.Sz[qubit] + (b + c) / 2 * self.Sx[qubit] + (
                                c - b) / 2 / 1j * self.Sy[qubit]),
                        dims=[[2 for i in range(self.N)], [2 for i in range(self.N)]]) * self.dt / self.Tgate
                elif name == 'MultiAncillaQubitOperator':
                    n = int(math.log2(operator.data.shape[0]))
                    bigOp = tensor([Qobj(np.array([[1, 0], [0, 1]])) if i > 0 else Qobj(sp.linalg.logm(operator),
                                                                                        dims=[[2 for i in range(n)],
                                                                                              [2 for i in range(n)]])
                                    for i in range(self.N - n + 1)])  # MAY CAUSE ERRORS
                    H += bigOp * self.dt / self.Tgate
                elif name == 'MultiSensorQubitOperator':
                    n = int(math.log2(operator.data.shape[0]))
                    bigOp = tensor(
                        [Qobj(np.array([[1, 0], [0, 1]])) if self.N - n - i - 1 >= 0 else sp.linalg.logm(operator) for i
                         in range(self.N - n + 1)])  # MAY CAUSE ERRORS
                    H *= bigOp * self.dt / self.Tgate
                elif name == 'm':  # measures only one qubit
                    state = self.state.ptrace(qubit)
                    p = list(np.real(np.diag(state.data.toarray())))
                    # print(p)
                    if np.random.rand() < np.real(p[0]):
                        return '0', p
                    else:
                        return '1', p
                else:
                    print('command not found')

            start_ind = len(self.history[0][0])

            # apply GATE action with decoherence on every Qubit
            if (self.dephase or self.amplitude_damp):
                U = Qobj(H).expm()
                for i in range(int(num_gates * self.Tgate / self.dt)):
                    self.state = U * self.state * U.dag()
                    for qubitIndex in range(self.N):
                        self.__decoherenceTS(qubitIndex)
                    self.__collect()  # collect data if flag is True

            # apply GATE without decoherence
            else:
                H = H * int(num_gates * self.Tgate / self.dt)
                U = Qobj(H).expm()
                self.state = U * self.state * U.dag()
                self.__collect()  # collect data if flag is True

            end_ind = len(self.history[0][0])
            self.__update_bloch(start_ind, end_ind)



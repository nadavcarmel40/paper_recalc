
from Constants import *
from Utils import *

#################################################################################
##############                  register class                   ################
#################################################################################

class BasicQuantumRegisterProj():

    def __init__(self, N, rho_0, w01s=None):
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
        dt = 20e-9 # the time to activate a gate (the size of time step in the simulation) is 20 nanoseconds

        self.state = rho_0
        self.projState = rho_0

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
        self.U0 = Qobj(-1j * self.H0.data / hbar * dt).expm()


        # Error section
        self.gatePhaseError = False
        self.controlQubitError = False
        self.constantUerror = False
        self.constantUerrorFFed = False
        self.randomUerror = False
        self.randomUerrorFFed = False

        n = int(2**self.N)
        nonH = Qobj(np.random.normal(scale=1, size=(n,n))+1j*np.random.normal(scale=1, size=(n,n)), dims=[[2 for i in range(self.N)],[2 for i in range(self.N)]])

        self.constantUNoise = 1/2*(nonH+nonH.dag())

        self.sigmaForError = 0

    def setError(self, gatePhaseError=False, controlQubitError=False, constantUerror=False, constantUerrorFFed=False, randomUerror=False, randomUerrorFFed=False):
        self.gatePhaseError = gatePhaseError
        self.controlQubitError = controlQubitError
        self.constantUerror = constantUerror
        self.constantUerrorFFed = constantUerrorFFed
        self.randomUerror = randomUerror
        self.randomUerrorFFed = randomUerrorFFed

    #################################################################################
    ###############          define 1-qubit gates                     ###############
    #################################################################################

    def __applyU(self, U):
        self.state = U * self.state * U.dag()
        self.projState = U * self.projState * U.dag()

    def X(self, qubitIndex):
        """
        create H matrix for X gate
        """
        phase = np.pi
        if self.gatePhaseError:
            phase = np.pi + np.random.normal(loc=meanForError, scale=self.sigmaForError)
        return -1j * self.Sx[qubitIndex] * phase / 2

    def Y(self, qubitIndex):
        """
        create H matrix for Y gate
        """
        phase = np.pi
        if self.gatePhaseError:
            phase = np.pi + np.random.normal(loc=meanForError, scale=self.sigmaForError)
        return -1j * self.Sy[qubitIndex] * phase / 2

    def Z(self, qubitIndex):
        """
        create H matrix for Z gate
        """
        phase = np.pi
        if self.gatePhaseError:
            phase = np.pi + np.random.normal(loc=meanForError, scale=self.sigmaForError)
        return -1j * self.Sz[qubitIndex] * phase / 2

    def S(self, qubitIndex):
        """
        create H matrix for S gate
        """
        phase = np.pi
        if self.gatePhaseError:
            phase = np.pi + np.random.normal(loc=meanForError, scale=self.sigmaForError)
        return -1j * self.Sz[qubitIndex] * phase / 4

    def T(self, qubitIndex):
        """
        create H matrix for T gate
        """
        phase = np.pi
        if self.gatePhaseError:
            phase = np.pi + np.random.normal(loc=meanForError, scale=self.sigmaForError)
        return -1j * self.Sz[qubitIndex] * phase / 8

    def H(self, qubitIndex):
        """
        create H matrix for H gate
        """
        phase = np.pi
        if self.gatePhaseError:
            phase = np.pi + np.random.normal(loc=meanForError, scale=self.sigmaForError)
        return -1j * phase / 2 * (2 ** (self.N - 1)) * 2 * (self.Sx[qubitIndex] + self.Sz[qubitIndex]).unit()

    #################################################################################
    ###############          define full 2-qubit gates               ################
    #################################################################################

    def __addControlQubit(self, H, q):
        """
        adds control qubit q to gate defined by hamiltonian H
        """
        error = 0
        if self.controlQubitError:
            error = np.abs(np.random.normal(loc=meanForError, scale=self.sigmaForError))
            if error>1:
                error=1
        return ((1-error)*(self.qI - self.Sz[q]) + error*(self.qI + self.Sz[q])) / 2 * H

    def iSWAP(self, q1, q2):
        """
        does iSWAP gate between the qubits, small time step
        TGate assumed to be max(Tx,Ty) in default
        ############################################################# GATE DOESNT WORK
        """
        phase = np.pi
        if self.gatePhaseError:
            phase = np.pi + np.random.normal(loc=meanForError, scale=self.sigmaForError)
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
        if self.gatePhaseError:
            phase = np.pi + np.random.normal(loc=meanForError, scale=self.sigmaForError)
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
        F=np.cumsum(p1/np.sum(p1))
        x = rand()
        i=1
        while x>F[i]:
            i+=1
        n = math.log2(len(p))
        res="{0:b}".format(i-1)
        while len(res)<n:
            res = '0'+res

        #update self state by measurment
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

    def update(self, rho):
        """
        updates self.state to be rho
        """
        if self.state.data.shape == rho.data.shape:
            self.state = rho
            self.projState = rho
        else:
            print("rho is not of the right dims")

    #################################################################################
    ##############          do a list of gates on the qubit          ################
    #################################################################################

    def run(self, commands, beta=0.2, showHistogram=False):
        """
        do a list of gates on the qubit.
        commands - list of commands that is a list of lists of lists, each list is of the form
            [('c',q1,q2), ('c',q1,q2)] where 'c' is a command and q1,q2 are the qubits involved
            (q2 optional), and each list as in description represents one time cut by gates
        returns - list of measurments as [(qi,m)..] where qi is the measured qubit and m is the
        measurment result.
        decoherence_mode = 0: run with no decoherence
        decoherence_mode = 1: run with random gate phase errors
        decoherence_mode = 2: run with random control qubit errors
        decoherence_mode = 3: run with 1+2
        """
        for timecut in commands:
            H = self.H0
            for commandrep in timecut:
                command = commandrep[0]
                q1 = commandrep[1]
                if len(commandrep)==3:
                    q2 = commandrep[2]
                if command == 'x':
                    H += self.X(q1)
                elif command == 'y':
                    H += self.Y(q1)
                elif command == 'z':
                    H += self.Z(q1)
                elif command == 's':
                    H += self.S(q1)
                elif command == 't':
                    H += self.T(q1)
                elif command == 'h':
                    H += self.H(q1)
                elif command == 'CZ':
                    H += self.__addControlQubit(self.Z(q2),q1)
                elif command == 'iSWAP':
                    H += self.iSWAP(q1,q2)
                elif command == 'CX':
                    H += self.__addControlQubit(self.X(q2),q1)
                elif command == 'R2inv':
                    H += self.__addControlQubit(self.Rkinv(2,q2),q1)
                elif command == 'R3inv':
                    H += self.__addControlQubit(self.Rkinv(3,q2),q1)
                elif command == 'm':
                    measurments, p = self.measure(q1, showHistogram=showHistogram)
                elif command.split()[0] == 'CU':
                    u = commandrep[-1]
                    n = int(math.log2(u.data.shape[0]))
                    n_count = self.N-n
                    if self.randomUerror:
                        u = addNoise2G(u,beta)
                    k = int(command.split()[1])
                    if self.constantUerror:
                        tempnoise = Qobj(self.constantUNoise[:2**n,:2**n], dims=[[2 for i in range(n)],[2 for i in range(n)]])
                        noise = tempnoise*norm(np.max(u))
                        u = u+beta*noise
                    temp = self.powU2(u,k)
                    if self.randomUerrorFFed:
                        temp = addNoise2G(temp, beta)
                    if self.constantUerrorFFed:
                        tempnoise = Qobj(self.constantUNoise[:2**n,:2**n], dims=[[2 for i in range(n)],[2 for i in range(n)]])
                        noise = tempnoise*norm(np.max(u))
                        temp = temp+beta*noise*1j
                    tempbig = tensor([qeye(2) if self.N - n - i - 1 >= 0 else temp for i in range(
                        self.N - n + 1)])  # creating U every time instead of just use the prev round might be problomatic with timing
                    H += self.__addControlQubit(tempbig, q1)
                elif command == 'applyOperator':
                    self.projState = q1 * self.projState * q1.dag() # q1 is now operator
                    self.state = self.state.unit()
            U = Qobj(H).expm()
            self.__applyU(U)
        return None
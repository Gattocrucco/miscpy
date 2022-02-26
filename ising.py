# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg
from scipy import linalg
import numba

__doc__ = """

Module for computing the 1D quantum Ising model hamiltonian:

    H = -J \sum_i \sigma^z_i \sigma^z_{i+1} +
        -h \sum_i \sigma^z_i +
        -g \sum_i \sigma^x_i.

The base used is the tensor product of the bases of \sigma^z_i. So the `i`-th
component of a numeric vector is the coefficient for the base vector

    \ket{\psi_i} = \\bigotimes_{k=0}^{n-1} \ket{\sigma_z, i_k}

where i_k is the `k`-th binary digit of `i` (starting from the least
significative), and \ket{\sigma_z, 0 or 1} are the eigenvectors of \sigma_z,
0 for spin up and 1 for spin down.

Functions
---------
hamiltonian :
    Compute the full hamiltonian as a sparse matrix.
hamiltonian_vector_product :
    As the name says, but also hamiltonian-matrix product.
HVP :
    Constructs a scipy.sparse.linalg.LinearOperator for the hamiltonian.
diagonalize :
    Diagonalize fully or partially the hamiltonian.
magnetization :
    Compute a matrix element of the magnetization operator.
parity_projection :
    Project a vector on the even or odd subspace.

Classes
-------
ModifiedLanczos :
    (Too) simple algorithm for computing the ground and first excited levels.

"""

def hamiltonian(n, h=0, g=0, J=1, dtype='float32', databuf=None):
    """
    Compute the hamiltonian matrix for the 1D Ising model:
    
    H = -J \sum_i \sigma^z_i \sigma^z_{i+1} - h \sum_i \sigma^z_i - g \sum_i \sigma^x_i
    
    Parameters
    ----------
    n : integer
        The number of spins.
    h : number, default: 0
        The longitudinal field.
    g : number, default: 0
        The transverse field.
    J : number, default: 1
        The coupling.
    dtype : numpy data type (default float32)
        The data type of the result. Ignored if `databuf` is specified.
    databuf : (2^n, n+1) C-order numpy array or None (default None)
        If specified, it is used as data buffer for the output matrix.
    
    Returns
    -------
    H : 2^n x 2^n symmetric CSR sparse matrix (scipy.sparse.csr_matrix)
        The hamiltonian in the tensor product basis. The entries are real.
    """
    # Check input
    n = int(n)
    assert 0 <= n <= 20
    for x in (g, h, J):
        assert np.isscalar(x)
        assert np.isfinite(x)
    dtype = np.dtype(dtype)
    
    # Data buffer
    if databuf is None:
        databuf = np.empty((2 ** n, n + 1), dtype=dtype)
    else:
        assert isinstance(databuf, np.ndarray)
        assert databuf.shape == (2 ** n, n + 1)
        assert databuf.flags['C_CONTIGUOUS']
        assert databuf.flags['WRITEABLE']
    
    # Build empty sparse matrix
    H = sparse.csr_matrix((2 ** n, 2 ** n), dtype=databuf.dtype)
    
    # Indices arrays
    indices = np.empty(databuf.shape, dtype=H.indices.dtype)
    indptr = np.arange(0, 1 + (n + 1) * 2 ** n, n + 1, dtype=H.indptr.dtype)
    assert len(indptr) == 1 + 2 ** n
    
    # Compute nonzero entries
    _hamiltonian(n, J, h, g, databuf, indices)
    
    # Flatten
    databuf.shape = 2 ** n * (n + 1)
    indices.shape = 2 ** n * (n + 1)
    
    # Fill sparse matrix
    H.data = databuf
    H.indices = indices
    H.indptr = indptr
    H.has_canonical_format = True
    
    return H

@numba.jit(nopython=True, cache=True)
def _hamiltonian(n, J, h, g, data, indices):
    """
    n = number of spins
    J = coupling
    h = longitudinal field
    g = transverse field
    data = (2 ** n, n + 1) array
    indices = (2 ** n, n + 1) integer array
    """
    terms = -g * np.ones(n + 1, dtype=data.dtype)
    terms_idxs = np.empty(n + 1, dtype=indices.dtype)
    
    for row in range(2 ** n):
        # coupling
        different_neighbors = _popcount(row ^ _cycle_left(row, n))
        equal_neighbors = n - different_neighbors
        terms[n] = -J * (equal_neighbors - different_neighbors)
        terms_idxs[n] = row
        
        # longitudinal field
        down_spin = _popcount(row)
        up_spin = n - down_spin
        terms[n] += -h * (up_spin - down_spin)
        
        # transverse field
        for j in range(n):
            col = row ^ (1 << j) # flip bit j
            terms_idxs[j] = col
        
        # write entries
        sorted_idxs = np.argsort(terms_idxs)
        data[row] = terms[sorted_idxs]
        indices[row] = terms_idxs[sorted_idxs]

def hamiltonian_vector_product(psi, h=0, g=0, J=1, out=None):
    """
    Compute the product of the 1D Ising hamiltonian with a vector or matrix.
    The hamiltonian is:
    
    H = -J \sum_i \sigma^z_i \sigma^z_{i+1} - h \sum_i \sigma^z_i - g \sum_i \sigma^x_i
    
    Parameters
    ----------
    psi : array
        A real or complex vector. The length must be a power of 2. If
        multidimensional, the first axis is the vector index, and the other
        are broadcasted, i.e. it behaves like a matrix multiplication.
    h : number (default 0)
        The longitudinal field.
    g : number (default 0)
        The transverse field.
    J : number (default 1)
        The coupling.
    out : None (default) or array
        Array with the same shape as `psi` where the result is written.
        If not specified, the returned vector has the same data type as `psi`.
    
    Returns
    -------
    out : array
        H @ psi (has the same shape as `psi`).
    scp : scalar or array
        psi.conj().T @ H @ psi. It is an array if `psi` is not 1D (the scalar
        product is computed for all columns of `psi`).
    norm : scalar or array
        psi.conj().T @ psi, the squared norm of `psi`.
    """
    # Check input
    assert isinstance(psi, np.ndarray)
    assert len(psi.shape) >= 1
    float_n = np.log2(len(psi))
    n = int(float_n)
    assert n == float_n
    assert 0 <= n <= 20
    for x in (g, h, J):
        assert np.isscalar(x)
        assert np.isfinite(x)
    
    # Output array
    if out is None:
        out = np.empty_like(psi)
    else:
        assert isinstance(out, np.ndarray)
        assert out.shape == psi.shape
        
    # Choose dtype for scalar product (to avoid losing numerical precision)
    sdtype = _real_accumulator_dtype(out.dtype)
    
    # Call actual function (compiled with numba)
    scp, norm = np.zeros((2, 1) + psi.shape[1:], dtype=sdtype)
    _hvp(out, n, psi, h, g, J, scp, norm)
    return out, scp[0], norm[0]

def _real_accumulator_dtype(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.int64
    elif np.issubdtype(dtype, np.floating):
        return np.float64
    elif np.issubdtype(dtype, np.complexfloating):
        return np.float64
    else:
        raise ArgumentError(dtype)

@numba.jit(nopython=True, cache=True)
def _hvp(out, n, psi, h, g, J, s, ni):
    """
    out = output vector, not required to be 0. CAN NOT BE the same as input!
    n = the length of vectors is 2 ** n
    psi = input vector
    h = longitudinal field
    g = transverse field
    J = coupling
    s = accumulator for hermitian product
    ni = accumulator for input normalization
    """
    for i in range(len(out)):
        num1 = _popcount(i)
        num0 = n - num1
        
        num1couple = _popcount(i ^ _cycle_left(i, n))
        num0couple = n - num1couple
        
        out[i] = -(J * (num0couple - num1couple) + h * (num0 - num1)) * psi[i]

        for j in range(n):
            ij = i ^ (1 << j) # flip bit j
            out[i] -= g * psi[ij]
        
        s += np.real(np.conj(psi[i]) * out[i])
        ni += np.real(np.conj(psi[i]) * psi[i])

@numba.jit('intp(intp)', nopython=True)
def _popcount(n):
    """
    Count the number of binary 1s in `n`.
    """
    count = 0
    while n:
        n &= (n - 1)
        count += 1
    return count

@numba.jit('intp(intp, uint)', nopython=True)
def _cycle_left(x, n):
    """
    Cycle `x` left over the `n` least significant bits, assuming more
    significant bits are zero. It must be `n` <= #bits - 2, e.g. n <= 62 with
    64 bit.
    """
    x <<= 1
    msb = (x & (1 << n)) >> n # get the most significant bit
    x |= msb # copies the most significant bit to the least significant
    x &= ~(1 << n) # clears the most significant bit
    
    return x

def HVP(n, h=0, g=0, J=1, parity=None, dtype=None):
    """
    Makes a scipy.sparse.linalg.LinearOperator for the Ising 1D Hamiltonian
    with `n` spins.
    
    Parameters
    ----------
    n : integer
        Number of spins.
    h : number
        Longitudinal field.
    g : number
        Transverse field.
    J : number
        Coupling.
    parity : None or str
        If 'even' or 'odd', the result is projected on that parity.
    dtype : None or numpy datatype
        If None (default), the returned object has "official" datatype float64
        but it will actually use the same datatype as the inputs it is applied
        to. Instead, if `dtype` is specified, the output will always have that
        type.
    
    Returns
    -------
    H : LinearOperator
        An operator that can be used in sparse linalg algorithms.
    """
    # Check input
    assert n == int(n) and n >= 0
    for x in h, g, J:
        assert np.isscalar(x) and np.isfinite(x)
    assert parity in (None, 'even', 'odd')
    
    # Define multiplication
    if dtype is None:
        mult = lambda v: hamiltonian_vector_product(v, h, g, J)[0]
    else:
        dtype = np.dtype(dtype)
        mult = lambda v: hamiltonian_vector_product(v, h, g, J, np.empty_like(v, dtype=dtype))[0]
    
    # Eventually add parity projection
    if parity:
        oldmult = mult
        mult = lambda v: parity_projection(oldmult(v), parity, inplace=True)[0]
    
    # Adjoint multiplications
    if parity:
        rmult = lambda v: oldmult(parity_projection(v, parity)[0])
    else:
        rmult = mult
    
    # Build LinearOperator
    return slinalg.LinearOperator(
        shape=(2 ** n,) * 2,
        matvec=mult,
        rmatvec=rmult,
        matmat=mult,
        rmatmat=rmult,
        dtype='float64' if dtype is None else dtype
    )
    # specify explicitly dtype because otherwise the constructor does a
    # vector product to see the result type

@numba.jit(nopython=True, cache=True)
def _fill_random(v):
    for i in range(len(v)):
        v[i] = np.random.normal()

class ModifiedLanczosState:
    """
    Represents a state of the modified Lanczos algorithm, which consists
    of a vector and the multiplication of the hamiltonian by the same vector.
    
    Members
    -------
    psi : array
        A vector.
    Hpsi : array
        hamiltonian @ psi
    psisq : scalar
        psi.conj().T @ psi (the squared norm of psi)
    psiHpsi : scalar
        psi.conj().T @ hamiltonian @ psi (the matrix element)
    eigenv : scalar (readonly)
        psiHpsi / psisq (if psi is an eigenstate, this is its energy)
    
    Methods
    -------
    normalize :
        Normalize the vector to unit norm. All the members are rescaled
        appropriately.
    project :
        Take the even or odd part of the vector. This assumes the hamiltonian
        commutes with parity.
    """
    
    def __setattr__(self, name, value):
        assert name in {'psi', 'Hpsi', 'psisq', 'psiHpsi'}
        super().__setattr__(name, value)
    
    @property
    def eigenv(self):
        return self.psiHpsi / self.psisq
    
    def normalize(self):
        assert self.psisq > 0
        norm = np.sqrt(self.psisq)
        self.psi /= norm
        self.Hpsi /= norm
        self.psiHpsi /= self.psisq
        self.psisq = 1
    
    def project(self, parity):
        self.psi, self.psisq = parity_projection(self.psi, parity, inplace=True)
        self.Hpsi, _ = parity_projection(self.Hpsi, parity, inplace=True)
        self.psiHpsi = np.vdot(self.psi, self.Hpsi)

class ModifiedLanczos:
    """
    Class to find the ground state of the 1D Ising model using a modified
    Lanczos algorithm. It is not efficient, it is just used as a reference
    and cross-check because the algorithm is simple.
    
    Members
    -------
    state : ModifiedLanczosState
        The state of the algorithm. Has members `psi` and `eigenv` which
        represent the current estimate of the ground state and its energy.
    params : tuple
        The tuple (h, g, J) of the model parameters.
    nit : integer (readonly)
        Current number of iterations of the algorithm.
    
    Methods
    -------
    set_params :
        Set the model parameters.
    initialize :
        Initialize the algorithm manually.
    iteration :
        Do one iteration of the algorithm manually.
    run :
        Run the complete algorithm with initialization until convergence.
    """
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, state):
        assert isinstance(state, (ModifiedLanczosState, type(None)))
        self._state = state
    
    @property
    def params(self):
        """
        The tuple of parameters (h, g, J).
        """
        return self._params
    
    @params.setter
    def params(self, value):
        value = tuple(value)
        assert len(value) == 3
        for x in value:
            assert np.isscalar(x) and np.isfinite(x)
        self._params = value
    
    def set_params(self, h=0, g=0, J=1):
        self.params = (h, g, J)
    
    def __init__(self, n, h=0, g=0, J=1):
        """
        Initialize the object for `n` spins with the specified parameters.
        The parameters can the be changed at any time, but the number of spins
        can not.
        """
        assert n == int(n) and n >= 2
        self.n = int(n)
        self.params = (h, g, J)
        self.state = None
    
    @property
    def nit(self):
        """
        Number of iterations. Reset by initialize().
        """
        return self._nit
    
    def initialize(self, dtype='float32', parity=None):
        """
        Initialize the algorithm with a random vector and specified data type
        (default single precision). The interation counter is reset to 0.
        
        Parameters
        ----------
        dtype : numpy dtype (default float32)
            The numerical type used for the vectors. The scalars always use
            the highest precision possible, e.g. with float32 vectors the
            scalars are float64.
        parity: None or str
            If 'even' or 'odd', the generated random vector has the given
            parity.
        """
        s = ModifiedLanczosState()
        s.psi = np.empty(2 ** self.n, dtype=dtype)
        _fill_random(s.psi)
        s.Hpsi, s.psiHpsi, s.psisq = hamiltonian_vector_product(s.psi, *self.params)
        if parity:
            s.project(parity)
        s.normalize()
        self.state = s
        self._nit = 0
        
    def iteration(self):
        """
        Do one iteration of the algorithm. The state is not updated, the new
        state is only returned, and the iteration counter is not increased.
        
        If the current state is already an eigenstate up to numerical
        precision, the iteration is not performed and None is returned.
        """
        assert self.state
        
        # make s.Hpsi orthogonal to s.psi and store result in s1
        s0 = self.state
        s1 = ModifiedLanczosState()
        s1.psi = s0.Hpsi - s0.eigenv * s0.psi
        s1.Hpsi, s1.psiHpsi, s1.psisq = hamiltonian_vector_product(s1.psi, *self.params)
        if s1.psisq == 0:
            return None
        H01 = np.sqrt(s1.psisq)
        s1.normalize()
    
        # diagonalize subspace matrix
        H00 = s0.psiHpsi
        H11 = s1.psiHpsi
        H = np.array([[H00, H01], [H01, H11]])
        w, V = linalg.eigh(H)
        v = V[:, np.argmin(w)]
    
        # compute minimum energy eigenvector in subspace
        news = ModifiedLanczosState()
        news.psi   = v[0]      * s0.psi   + v[1]      * s1.psi
        news.Hpsi  = v[0]      * s0.Hpsi  + v[1]      * s1.Hpsi
        news.psisq = v[0] ** 2 * s0.psisq + v[1] ** 2 * s1.psisq
        news.psiHpsi = np.min(w)
        assert np.allclose(news.psisq, 1)
        assert np.allclose(news.psiHpsi, v[0] ** 2 * H00 + 2 * v[0] * v[1] * H01 + v[1] ** 2 * H11)
        
        return news
        
        # Note on normalizations: if s0 was not normalized, the result
        # would be correct but not normalized. If you don't normalize s1, then
        # you have to use H01 = s1.psisq. But without normalizing the
        # convergence is way slower. `news` comes out already normalized but
        # might lose normalization after a lot of iterations.
    
    def run(self, rtol=1e-8, atol=1e-8, parity=None, maxit=None, **kwargs):
        """
        Run the algorithm. If it has not been already explicitly initialized,
        the algorithm is initialized with a random vector. If you call run()
        multiple times, the iteration count is cumulatively increased. At
        least on iteration is done even if convergence was satisfied in the
        previous call.
        
        Parameters
        ----------
        rtol, atol : scalars
            The relative and absolute tolerance used to check at each iteration
            if the eigenvalue has stopped changing.
        parity : None or str
            If 'even' or 'odd', initialize with given parity and reproject
            after each iteration. Use it when h=0 to find the lowest level
            with given parity, and to accelerate convergence as the ground and
            first excited level get closer.
        maxit : None or integer
            Maximum number of iterations. Default: 100 * n.
        **kwargs:
            Additional arguments are passed to `initialize` in case it is
            called.
        
        Returns
        -------
        eigenv : scalar
            The ground level energy.
        psi : vector
            The ground state.
        
        The result can also be accessed from the `state` member.
        """
        # check input
        for x in rtol, atol:
            assert np.isscalar(x) and np.isfinite(x) and x >= 0
        minimum = 2 * np.finfo(np.float64).eps
        assert rtol >= minimum or atol >= minimum
        # The dtype for this is always float64 because the hermitian products
        # are always computed in double precision regardless of the vectors
        # data types.
        
        # eventually initialize
        if not self.state:
            self.initialize(parity=parity, **kwargs)
        
        stop = False
        bound = self._nit + (maxit if maxit else 100 * self.n) 
        while not stop and self._nit < bound:
            news = self.iteration()
            if news is None:
                break
            if parity:
                news.project(parity)
                news.normalize()
        
            # check convergence
            if np.allclose(self.state.eigenv, news.eigenv, rtol=rtol, atol=atol):
                stop = True
        
            # shift states for stop or next cycle
            self.state = news
            
            self._nit += 1
    
        return self.state.eigenv, self.state.psi

def diagonalize(n, h=0, g=0, J=1, parity=None, eigenvectors=True, which=1, dtype='float64', **kwargs):
    """
    Diagonalize the 1D Ising hamiltonian.
    
    Parameters
    ----------
    n : integer
        Number of spins.
    h : number
        Longitudinal field.
    g : number
        Transverse field.
    J : number
        Coupling.
    parity : None or str
        If 'even' or 'odd' find only the eigenstates with given parity. Can not
        be used when `which` is 'all'. Makes no sense when h != 0.
    eigenvectors : bool
        If False compute only the eigenvalues. If True (default) return also
        the eigenvectors.
    which : int or str
        If a number, it is the number of lower energy eigenvectors computed.
        If 'all' a full diagonalization is performed. The default is 1, i.e.
        only the ground state.
    dtype : numpy datatype
        Datatype used in the computation and in the returned arrays (default
        float64).
    
    Keyword arguments
    -----------------
    Additional keyword arguments are passed to `scipy.linalg.eigh` when
    `which == 'all'` or to `scipy.sparse.linalg.eigsh` when `which` is an
    integer.
    
    Returns
    -------
    eigenvalues : 1D  array
        Array of the eigenvalues.
    U : 2D array
        Matrix of normalized eigenvectors (the columns), returned if
        `eigenvectors` is True.
    """
    # Check input
    eigenvectors = bool(eigenvectors)
    dtype = np.dtype(dtype)
    
    # Full diagonalization with dense matrix
    if which == 'all':
        assert parity is None
        assert n <= 12
        H = hamiltonian(n, h, g, J, dtype=dtype).toarray()
        return linalg.eigh(
            H,
            eigvals_only=not eigenvectors,
            overwrite_a=True,
            check_finite=False,
            **kwargs
        )
    
    # Partial diagonalization with ARPACK
    elif isinstance(which, int):
        assert 0 < which < min(2 * n, 2 ** n)
        H = HVP(n, h, g, J, parity=parity, dtype=dtype)
        return slinalg.eigsh(
            H,
            k=which,
            which='SA',
            return_eigenvectors=eigenvectors,
            **kwargs
        )
    
    else:
        raise ArgumentError(which)

def magnetization(psi1, psi2, abs=False, average=True, direction='z'):
    """
    Compute the matrix element of the magnetization with the two input vectors.
    
        M = 1/n \sum_i \sigma^z_i
        avg(M) = psi1.conj().T @ M @ psi2
    
    The computation is done on the first axis of the input arrays and
    vectorized on the others.
    
    Parameters
    ----------
    psi1, psi2: (2 ** n, ...) array
        Respectively left and right side vector.
    abs : bool (default False)
        Compute the absolute value of the magnetization. Not supported for
        direction == 'x'.
    average : bool (default True)
        If False, do not divide the result by the number of spins `n`.
    direction : str (default 'z')
        If 'x', the magnetization is along x, i.e.
    
            M = 1/n \sum_i \sigma^x_i.
    
    Returns
    -------
    out : scalar or array
        psi1.conj().T @ M @ psi2 (with |M| if `abs` and multiplied by `n` if
        `average` is False).
    """
    # Check input
    abs = bool(abs)
    average = bool(average)
    
    direction = str(direction)
    assert direction in ('z', 'x')
    assert direction == 'z' or not abs
    
    for psi in psi1, psi2:
        assert isinstance(psi, np.ndarray)
        assert len(psi.shape) >= 1
        assert np.issubdtype(psi.dtype, np.number)
    
    assert len(psi1) == len(psi2)
    float_n = np.log2(len(psi1))
    n = int(float_n)
    assert n == float_n
    
    np.broadcast(psi1, psi2)
    
    # Type for result
    anysubdtype = lambda dtype: any(np.issubdtype(psi.dtype, dtype) for psi in [psi1, psi2])
    if anysubdtype(np.complexfloating):
        dtype = np.complex128
    elif anysubdtype(np.floating):
        dtype = np.float64
    elif anysubdtype(np.integer):
        dtype = np.float64 if average else np.int64
    
    # Call compiled function
    out = np.zeros((1,) + np.broadcast(psi1, psi2).shape[1:], dtype=dtype)
    if direction == 'z':
        _magnetization_z(n, psi1, psi2, abs, average, out)
    else:
        _magnetization_x(n, psi1, psi2, average, out)
    return out[0]

@numba.njit(cache=True)
def _magnetization_z(n, psi1, psi2, abs, avg, out):
    """
    n = number of spins
    psi1 = left side
    psi2 = right side
    abs = wether to take the absolute value
    avg = wether to divide by n
    out = output array set to zero
    """
    for i in range(len(psi1)):
        spindown = _popcount(i)
        spinup = n - spindown
        mag = spinup - spindown
        if abs:
            mag = np.abs(mag)
        if avg:
            mag /= n
        out += np.conj(psi1[i]) * psi2[i] * mag

@numba.njit(cache=True)
def _magnetization_x(n, psi1, psi2, avg, out):
    """
    n = number of spins
    psi1 = left side
    psi2 = right side
    avg = wether to divide by n
    out = output array set to zero
    """
    factor = 1/n if avg else 1
    for i in range(len(psi1)):
        for j in range(n):
            ij = i ^ (1 << j)
            out += np.conj(psi1[i]) * psi2[ij] * factor

def parity_projection(psi, parity, inplace=False, norm=True):
    """
    Compute the even or the odd part of `psi`. The parity operator is
        
        P = \prod_i \sigma^x_i.
    
    Parameters
    ----------
    psi : (2 ** n, ...) array
        The vector. Computation is vectorized on trailing axes.
    parity : str
        'even' or 'odd'.
    inplace : bool (default False)
        If True, the result is written directly in `psi`.
    norm : bool (default True)
        If True, the normalization of `psi` is preserved in the output. If
        False, the computation can be done in integer arithmetic.
    
    Returns
    -------
    out : array
        Array with the same shape and type as `psi`. It is actually `psi` if
        `inplace` is True.
    norm : scalar or array
        Squared norm of `out`.
    """
    # Check input
    assert isinstance(psi, np.ndarray)
    assert len(psi.shape) >= 1
    assert len(psi) >= 1
    assert np.issubdtype(psi.dtype, np.number)
    
    float_n = np.log2(len(psi))
    n = int(float_n)
    assert n == float_n
    
    parity = {'even': True, 'odd': False}[parity]
    
    inplace = bool(inplace)
    assert not inplace or psi.flags['WRITEABLE']
    
    norm = bool(norm)
    
    out = psi if inplace else np.empty_like(psi)
    sdtype = _real_accumulator_dtype(out.dtype)
    outnorm = np.zeros((1,) + out.shape[1:], dtype=sdtype)
    _parity_projection(n, psi, parity, norm, out, outnorm)
    return out, outnorm[0]
    
@numba.njit(cache=True)
def _parity_projection(n, psi, parity, norm, out, outnorm):
    """
    n = number of spins
    psi = 2 ** n vector
    parity = bool
    norm = bool
    out = 2 ** n vector (empty)
    outnorm = array (initialized to zero)
    """
    mask = (1 << n) - 1
    factor = 0.5 if norm else 1
    sign = 1 if parity else -1
    for i in range(len(psi)):
        j = (~i) & mask
        if i <= j: # condition needed when `psi` is `out`
            out[i] = (psi[i] + sign * psi[j]) * factor
        else:
            out[i] = sign * out[j]
        outnorm += np.real(np.conj(out[i]) * out[i])
    
if __name__ == '__main__':
    import unittest
    
    np.random.seed(20200306)
    
    class TestPopCount(unittest.TestCase):
        
        def test_random(self):
            for n in np.random.randint(1 + np.iinfo(np.intp).max, size=1000):
                self.assertEqual(_popcount(n), bin(n).count('1'))
        
        def test_zero(self):
            self.assertEqual(_popcount(0), 0)
    
    def cycle_left(x, n):
        binx = f'{{:0>{n}}}'.format(bin(x)[2:])
        return int(binx[1:] + binx[0], base=2)
    
    class TestCycle(unittest.TestCase):
        
        def test_random(self):
            n = np.random.randint(np.iinfo(np.intp).bits - 1, size=1000)
            x = np.random.randint((1 + np.iinfo(np.intp).max) // 2, size=1000)
            x = np.bitwise_and(x, 2 ** n - 1)
            for x, n in zip(x, n):
                self.assertEqual(_cycle_left(x, n), cycle_left(x, n))
        
        def test_zero(self):
            self.assertEqual(_cycle_left(0, 0), 0)
    
    class TestHamiltonian(unittest.TestCase):
        
        def test_zero(self):
            m = hamiltonian(0).toarray()
            self.assertEqual(m.shape, (1, 1))
            self.assertEqual(m[0, 0], 0)
        
        def test_symmetric(self):
            for _ in range(100):
                n = np.random.randint(1, 6)
                h, g, J = np.random.randn(3)
                H = hamiltonian(n, h, g, J).toarray()
                self.assertTrue(np.allclose(H, H.T))
        
        def test_canonical_format(self):
            for _ in range(100):
                n = np.random.randint(1, 6)
                h, g, J = np.random.randn(3)
                H = hamiltonian(n, h, g, J)
                del H._has_sorted_indices
                del H._has_canonical_format
                self.assertTrue(H.has_canonical_format)
        
        def test_same(self):
            for n in range(9):
                h, g, J = np.random.randn(3)
                
                H1 = hamiltonian(n, h, g, J, dtype='float64').toarray()
                self.assertTrue(np.allclose(H1, H1.T))
                
                H2 = np.empty((2 ** n,) * 2)
                psi = np.zeros(2 ** n)
                for i in range(2 ** n):
                    psi[i] = 1
                    hamiltonian_vector_product(psi, h, g, J, out=H2[i])
                    psi[i] = 0
                self.assertTrue(np.allclose(H2, H2.T))
                
                self.assertTrue(np.allclose(H1, H2))
    
    class TestHVP(unittest.TestCase):
        
        def test_product(self):
            for _ in range(100):
                n = np.random.randint(1, 6)
                h, g, J = np.random.randn(3)
                
                H = hamiltonian(n, h, g, J)
                psi = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
                psi1 = H @ psi
                psi2, s, n = hamiltonian_vector_product(psi, h, g, J)
                
                self.assertTrue(np.allclose(psi1, psi2))
                self.assertTrue(np.allclose(np.vdot(psi2, psi), s))
                self.assertTrue(np.allclose(np.vdot(psi, psi2), s))
                self.assertTrue(np.allclose(np.vdot(psi, psi), n))
                
        def test_matmat(self):
            for _ in range(100):
                n = np.random.randint(1, 6)
                h, g, J = np.random.randn(3)
                
                psi = np.random.randn(2 ** n, 2) + 1j * np.random.randn(2 ** n, 2)
                out1, _, _ = hamiltonian_vector_product(psi, h, g, J)
                out2 = np.empty_like(psi)
                for i in range(2):
                    hamiltonian_vector_product(psi[:, i], h, g, J, out2[:, i])

                self.assertTrue(np.allclose(out1, out2))
        
        def test_operator(self):
            for _ in range(100):
                n = np.random.randint(1, 6)
                h, g, J = np.random.randn(3)
                
                psi = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
                psi1 = HVP(n, h, g, J).matvec(psi)
                psi2, _, _ = hamiltonian_vector_product(psi, h, g, J)
                
                self.assertTrue(np.allclose(psi1, psi2))
        
        def test_adjoint(self):
            for _ in range(100):
                n = np.random.randint(1, 6)
                params = np.random.randn(3)
                
                psi = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
                H = HVP(n, *params)
                psi1 = H @ psi
                psi2 = H.H @ psi
                
                self.assertTrue(np.allclose(psi1, psi2))
    
        def test_parity_adjoint(self):
            for _ in range(100):
                n = np.random.randint(1, 6)
                params = np.random.randn(3)
                
                for parity in 'even', 'odd':
                    psi = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
                    H = HVP(n, *params, parity=parity)
                    psi1 = H.H @ psi
                    psi2, _ = parity_projection(psi, parity)
                    psi2, _, _ = hamiltonian_vector_product(psi2, *params)
                
                    self.assertTrue(np.allclose(psi1, psi2))

    class TestDiagonalize(unittest.TestCase):
        
        def test_ground(self):
            """
            Test modified Lanczos against ARPACK. This may fail because
            the modified Lanczos is crappy, just try again.
            """
            for _ in range(100):
                n = np.random.randint(2, 6)
                params = np.random.randn(3)
                
                w1, v1 = diagonalize(n, *params, which=1, dtype='f8', eigenvectors=True)
                w2, v2 = ModifiedLanczos(n, *params).run(rtol=1e-9, atol=1e-9, dtype='f8', maxit=np.inf)
                
                self.assertTrue(np.allclose(w1, w2, rtol=1e-4))
                x = np.abs(v2 / v1.reshape(-1))
                delta = np.abs(np.mean(x) - 1)
                sigma = max(np.std(x, ddof=1) / np.sqrt(len(x)), 1e-4)
                self.assertTrue(delta < 5 * sigma)
    
    class TestMagnetization(unittest.TestCase):
        
        def test_hground(self):
            """
            The ground state with h=0 must have 0 magnetization along z.
            """
            for _ in range(100):
                n = np.random.randint(2, 7)
                g = np.random.randn()
                
                # ground state for the case g=0
                v0 = np.zeros(2 ** n)
                v0[0] = 1
                v0[-1] = np.sign(g)
                
                _, v = diagonalize(n, g=g, which=1, dtype='f8', eigenvectors=True, v0=v0)
                v = v.reshape(-1)
                
                m = magnetization(v, v, average=True)
                self.assertTrue(np.abs(m) < 1e-8)
        
        def test_gground(self):
            """
            A ground state with g=0 must have 0 magnetization along x, apart
            if n=1.
            """
            for _ in range(100):
                n = np.random.randint(2, 10)
                even, odd = np.random.randn(2)
                v = np.zeros(2 ** n)
                v[[0, -1]] = [even + odd, even - odd]
                m = magnetization(v, v, direction='x')
                self.assertTrue(np.abs(m) < 1e-8)
                        
        def test_zfield(self):
            """
            The ground state with h >> J, g must be completely z-magnetized.
            """
            for _ in range(100):
                n = np.random.randint(2, 7)
                g = np.random.randn()
                hsign = 2 * np.random.randint(2) - 1
                h = hsign * 100 * (5 + np.random.randn())
                
                _, v = diagonalize(n, g=g, h=h, which=1, dtype='f8', eigenvectors=True)
                v = v.reshape(-1)
                
                m = magnetization(v, v, average=True)
                absm = magnetization(v, v, abs=True, average=True)
                self.assertTrue(np.allclose(m, hsign, rtol=1e-4))
                self.assertTrue(np.allclose(absm, 1, rtol=1e-4))
    
        def test_xfield(self):
            """
            The ground state with g >> J, h must be completely x-magnetized.
            """
            for _ in range(100):
                n = np.random.randint(2, 7)
                h = np.random.randn()
                gsign = 2 * np.random.randint(2) - 1
                g = gsign * 100 * (5 + np.random.randn())
                
                _, v = diagonalize(n, g=g, h=h)
                v = v.reshape(-1)
                m = magnetization(v, v, direction='x')
                self.assertTrue(np.allclose(m, gsign, rtol=1e-4))

    class TestParity(unittest.TestCase):
        
        def test_sign(self):
            """
            Check sign bug.
            """
            a = np.array([1, 0, 0, -1])
            self.assertTrue(np.all(a == parity_projection(a, 'odd')[0]))
            self.assertTrue(np.all(0 == parity_projection(a, 'even')[0]))
        
        def test_ground(self):
            """
            The ground state with h=0 has parity as the sign of g if n is odd
            and even is n is even. The first excited state has opposite parity.
            """
            for _ in range(100):
                n = np.random.randint(2, 10)
                g = np.random.randn()
                g = np.sign(g) * 0.5 + g
                # We need g not too small otherwise the first excited state may
                # be too close to the ground state to be distinguished
                # numerically.
                
                w, V = diagonalize(n, g=g, which=2)
                
                # ground
                v = V[:, np.argmin(w)]
                veven, _ = parity_projection(v, 'even')
                vodd, _ = parity_projection(v, 'odd')
                wrong = vodd if g > 0 or n % 2 == 0 else veven
                self.assertTrue(np.allclose(wrong, 0))
                
                # first excited
                v = V[:, np.argmax(w)]
                veven, _ = parity_projection(v, 'even')
                vodd, _ = parity_projection(v, 'odd')
                wrong = veven if g > 0 or n % 2 == 0 else vodd
                self.assertTrue(np.allclose(wrong, 0))
        
        def test_norm(self):
            """
            Check the even and odd components add up to the vector and they
            are orthogonal.
            """
            for _ in range(100):
                n = np.random.randint(1, 10)
                v = np.random.randn(2 ** n)
                vsq = np.vdot(v, v)
                veven, neven = parity_projection(v, 'even')
                vodd, nodd = parity_projection(v, 'odd')
                self.assertTrue(np.allclose(v, veven + vodd))
                self.assertTrue(np.allclose(vsq, neven + nodd))
        
        def test_inplace(self):
            """
            Check it works inplace.
            """
            for _ in range(100):
                n = np.random.randint(1, 10)
                v = np.random.randn(2 ** n)
                vcopy = np.copy(v)
                for parity in 'even', 'odd':
                    out, _ = parity_projection(v, parity, inplace=False)
                    parity_projection(v, parity, inplace=True)
                    self.assertTrue(np.allclose(v, out))
                    v[:] = vcopy
        
        def test_commutator(self):
            """
            PH(h) = H(-h)P,
            P = P+ - P-.
            """
            for _ in range(100):
                n = np.random.randint(1, 10)
                h, g, J = np.random.randn(3)
                psi = np.random.randn(2 ** n)
                
                psi1, _, _ = hamiltonian_vector_product(psi, h, g, J)
                psi1e, _ = parity_projection(psi1, 'even')
                psi1o, _ = parity_projection(psi1, 'odd')
                psi1 = psi1e - psi1o
                
                psi2e, _ = parity_projection(psi, 'even')
                psi2o, _ = parity_projection(psi, 'odd')
                psi2 = psi2e - psi2o
                psi2, _, _ = hamiltonian_vector_product(psi2, -h, g, J)
                    
                self.assertTrue(np.allclose(psi1, psi2))

    unittest.main()

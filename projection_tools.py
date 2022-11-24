# Authors: Stefanos Papanikolaou <stefanos.papanikolaou@mail.wvu.edu>
# BSD 2-Clause License
# Copyright (c) 2019, PapStatMechMat
# All rights reserved.
# How to cite SeaPy:
# S. Papanikolaou, Data-Rich, Equation-Free Predictions of Plasticity and Damage in Solids, (under review in Phys. Rev. Materials) arXiv:1905.11289 (2019)

def get_gev_vector(target_psd_matrix, noise_psd_matrix):
    """
    Returns the GEV beamforming vector.
    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    bins, sensors, _ = target_psd_matrix.shape
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex)
    for f in range(bins):
        try:
            eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :],
                                        noise_psd_matrix[f, :, :])
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = eig(target_psd_matrix[f, :, :],
                                       noise_psd_matrix[f, :, :])
        beamforming_vector[f, :] = eigenvecs[:, np.argmax(eigenvals)]
    return beamforming_vector 

def transform(self, X, y):
        """transform function"""
        XMat = np.array(X)
        yMat = np.array(y)

        if XMat.shape[0] != yMat.shape[0]:
            yMat = yMat.T
        assert XMat.shape[0] == yMat.shape[0]

        XMat -= XMat.mean(axis=0)
        Sw, Sb = calc_Sw_Sb(XMat, yMat)
        evals, evecs = eig(Sw, Sb)

        np.ascontiguousarray(evals)
        np.ascontiguousarray(evecs)

        idx = np.argsort(evals)
        idx = idx[::-1]
        evecs = evecs[:, idx]

        self.W = evecs[:, :self.n_components]
        X_transformed = np.dot(XMat, self.W)

        return X_transformed 


def eigenbasis(se, nb):
    # generate number sector
    ns1 = se.model.numbersector(nb)

    # get the size of the basis
    ns1size = ns1.basis.len  # length of the number sector basis
    # G1i = range(ns1size)    # our Greens function?

    # self energy
    # sigma = self.sigma(nb, phi)

    # Effective Hamiltonian
    H1n = ns1.hamiltonian

    # Complete diagonalization
    E1, psi1r = linalg.eig(H1n.toarray(), left=False)
    psi1l = np.conj(np.linalg.inv(psi1r)).T
    # psi1l = np.conj(psi1r).T

    # check for dark states (throw a warning if one shows up)
    # if (nb > 0):
    #     Setup.check_for_dark_states(nb, E1)

    return E1, psi1l, psi1r 

def contruct_ellipse_parallel(pars):
    
    Coor,cm,A_i,Vr,dims,dist,max_size,min_size,d=pars
    dist_cm = coo_matrix(np.hstack([Coor[c].reshape(-1, 1) - cm[k]
                                                for k, c in enumerate(['x', 'y', 'z'][:len(dims)])]))
    Vr.append(dist_cm.T * spdiags(A_i.toarray().squeeze(),
                                  0, d, d) * dist_cm / A_i.sum(axis=0))

    if np.sum(np.isnan(Vr)) > 0:
        raise Exception('You cannot pass empty (all zeros) components!')

    D, V = eig(Vr[-1])

    dkk = [np.min((max_size**2, np.max((min_size**2, dd.real)))) for dd in D]

    # search indexes for each component
    return np.sqrt(np.sum([(dist_cm * V[:, k])**2 / dkk[k] for k in range(len(dkk))], 0)) <= dist
#%% threshold_components 

def get_gevd_vals_vecs(target_psd_matrix, noise_psd_matrix):
    """
    Returns the eigenvalues and eigenvectors of GEVD
    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of eigen values  with shape (bins, sensors)
             eigenvectors (bins, sensors, sensors)
    """
    bins, sensors, _ = target_psd_matrix.shape
    eigen_values = np.empty((bins, sensors), dtype=np.complex)
    eigen_vectors = np.empty((bins, sensors, sensors), dtype=np.complex)
    for f in range(bins):
        try:
            eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :],
                                        noise_psd_matrix[f, :, :])
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = eig(target_psd_matrix[f, :, :],
                                       noise_psd_matrix[f, :, :])
               
        eigen_values[f,:] = eigenvals
        eigen_vectors[f, :] = eigenvecs
    return eigen_values, eigen_vectors # values in increasing order

def isNormal(A, method = 'definition'):

    # use Schur inequality to determine whether it's normal
    if method == 'Schur':
        # initialize eigenValue
        eigenValue = la.eig(A)[0]

        if abs(np.sum(eigenValue**2) - la.norm(A, 'fro')**2) < 0.00001:
            return True
        else:
            return False
    # use definition
    else:
        if abs((A.conjugate().T.dot(A) - A.dot(A.conjugate().T)).all()) < 0.00001:
            return True
        else:
            return False 

def get_esystem(basis,traj_edges,test_set=None,delay=1):
    """

    """
    if test_set is None:
        test_set = basis
    # Calculate Generator, Stiffness matrix
    L = get_generator(basis,traj_edges,test_set=test_set,delay=delay,dt_eff=1)
#    L = get_transop(basis,traj_edges,test_set=test_set,delay=delay)
    S = get_stiffness_mat(basis,traj_edges,test_set=test_set,delay=delay)
    # Calculate, sort eigensystem
    evals, evecs_l, evecs_r = spl.eig(L,b=S,left=True,right=True,overwrite_a=False,overwrite_b=False)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs_l = evecs_l[:,idx]
    evecs_r = evecs_r[:,idx]
    # Expand eigenvectors into real space.
    expanded_evecs_l = np.dot(test_set,evecs_l)
    expanded_evecs_r = np.dot(basis,evecs_r)
    return evals, expanded_evecs_l, expanded_evecs_r 

def FLQTQEB(engine,app):
    '''
    This method calculates the Floquet quasi-energy bands.
    '''
    if app.path is None:
        result=zeros((2,engine.nmatrix+1))
        result[:,0]=array(xrange(2))
        result[0,1:]=angle(eig(engine.evolution(ts=app.ts.mesh('t')))[0])/app.ts.volume('t')
        result[1,1:]=result[0,1:]
    else:
        rank,mesh=app.path.rank(0),app.path.mesh(0)
        result=zeros((rank,engine.nmatrix+1))
        result[:,0]=mesh if mesh.ndim==1 else array(xrange(rank))
        for i,paras in app.path('+'):
            result[i,1:]=angle(eig(engine.evolution(ts=app.ts.mesh('t'),**paras))[0])/app.ts.volume('t')
    name='%s_%s'%(engine,app.name)
    if app.savedata: savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result 



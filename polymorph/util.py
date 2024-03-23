import numpy as np
import copy

def abc2xyz(a=0, b=0, c=0, alpha=90, beta=90, gamma=90):
    xx = a
    yy = b*np.sin(gamma/180*np.pi)
    
    xy = b*np.cos(gamma/180*np.pi)
    xz = c*np.cos(beta/180*np.pi)
    if b == 0:
        yz = 0
    else:
        yz = (b*c*np.cos(alpha/180*np.pi) - xy*xz) / (np.sqrt(b**2-xy**2))
    
    zz = np.sqrt(c**2 - xz**2 - yz**2)
    
    return [xx,yy,zz,xy,xz,yz]

def xyz2abc(xx,yy,zz,xy,xz,yz):
    a = xx
    b = np.sqrt(
        yy**2 + xy**2
    )
    c = np.sqrt(
        zz**2 + xz**2 + yz**2
    )
    alpha = np.arccos(
    (xy*xz+yy*yz)/(b*c)
    )/np.pi*180
    beta = np.arccos(xz/c)/np.pi*180
    gamma = np.arccos(xy/b)/np.pi*180
    
    return [a,b,c,alpha,beta, gamma]


def translation(monomer, vector):
    m = copy.deepcopy(monomer)
    m.coords += np.array(vector)
    m.ends += np.array(vector)
    return m

def trans_along_vec(monomer, n):
    vector = monomer.vector
    m = translation(monomer,vector*n)
    return m

def rotation(p,q):
    p=np.array(p)/np.linalg.norm(np.array(p))
    q=np.array(q)/np.linalg.norm(np.array(q))

    n=np.cross(p,q)/np.linalg.norm(np.cross(p,q))
    angle=np.arccos(np.dot(p,q))

    N = np.array([
        [0, -n[2], n[1]],
        [n[2], 0, -n[0]],
        [-n[1], n[0], 0]
    ])

    R = np.eye(3) + np.sin(angle)*N + (1-np.cos(angle))*np.dot(N,N)

    return R


def treat_ends(box,N):
    box = copy.deepcopy(box)
    
    vector = box.molecules[0].monomers[0].vector

    for i,p in enumerate(box.molecules):
        moves = np.random.randint(-N, N+1)

        x = moves*vector[0]
        y = moves*vector[1]
        z = moves*vector[2]
        box.molecules[i] = p.move(x,y,z)
    
    return box

def random_orientation(box,angle,N,safe_dist=4.0):
    '''
    introduce orientational disorder in pi-pi stacking direction
    
    box: target box to treat
    angle: in degree, the angle to rotate
    N: N chains will be grouped
    save_dist: additional distance between each groups 
    '''
    
    box = copy.deepcopy(box)

    counter = 0

    for i,p in enumerate(box.molecules):

        (x,y,z) = list(p.geo_center)

        if i%N == 0:
            (x0,y0,z0) = (x,y,z)

        p = p.move(-x0,-y0,-z0)
        symbol = (-1)**counter
        for m in p.monomers:
            m.rotation_along_x(symbol*angle)

        box.molecules[i] = p.move(x0,y0+safe_dist*counter,z0)

        if (i+1)%N == 0:
            counter += 1

    return box

def treat_overlap(box, tol=0.3):
    from scipy.spatial import distance_matrix

    box = copy.deepcopy(box)
    nmol = len(box.molecules)

    counter = 0
    for i in range(nmol):
        pi = box.molecules[i]
        coordsi = pi.coords

        for j in range(i+1,nmol):
            pj = box.molecules[j]
            for k,m in enumerate(pj.monomers):
                coordsm = m.coords
    
                distances = distance_matrix(coordsi, coordsm)

                treat = np.where(distances>tol, 0, 1)
                treat = np.sum(treat, axis=0)
                counter += np.sum(treat)
                treat = np.vstack(
                    [treat,treat,treat]
                    )

                rand = np.random.rand(len(coordsm),3)*4*tol - 2*tol
                box.molecules[j].monomers[k].coords = coordsm + rand*np.transpose(treat)
    print(f"treated {counter} coords")
    return box

def exchange_molecules(box1, box2, Nexc, seed=False):
    
    N1 = len(box1.molecules)
    N2 = len(box2.molecules)
    
    seq1 = np.arange(N1)
    seq2 = np.arange(N2)
    
    if seed:
        np.random.seed(12345)
        sel1 = np.random.choice(seq1, size=Nexc, replace=False)
        np.random.seed(54321)
        sel2 = np.random.choice(seq2, size=Nexc, replace=False)
    
    else:
        sel1 = np.random.choice(seq1, size=Nexc, replace=False)
        sel2 = np.random.choice(seq2, size=Nexc, replace=False)
    
    for i in range(Nexc):
        p1 = copy.deepcopy(
                box1.molecules[sel1[i]]
            )
        p2 = copy.deepcopy(
                box2.molecules[sel2[i]]
            )
    
        x,y,z = list(p2.geo_center - p1.geo_center)
        p1 = p1.move(x,y,z)
        p2 = p2.move(-x,-y,-z)
        
        box1.molecules[sel1[i]] = p2
        box2.molecules[sel2[i]] = p1

def exchange_selected_molecules(box1, box2, sel):

    Nexc = len(sel)

    for i in range(Nexc):
        p1 = copy.deepcopy(
                box1.molecules[sel[i]]
            )
        p2 = copy.deepcopy(
                box2.molecules[sel[i]]
            )
    
        x,y,z = list(p2.geo_center - p1.geo_center)
        p1 = p1.move(x,y,z)
        p2 = p2.move(-x,-y,-z)
        
        box1.molecules[sel[i]] = p2
        box2.molecules[sel[i]] = p1

def get_gro_fixed_line( _filename_):
    ''' reading gromacs gro fixed lines'''

    _mol_       =   []
    _mtype_     =   []
    g_names     =   []
    _type_      =   []
    _xyz_       =   []
    _x_         =   []
    _y_         =   []
    _z_         =   []
    _corrupt = True
    with open(_filename_, 'r')  as indata:
        read_flag = False
        at=0
        at_num = 0
        
        _buffer = []
        for j_line in indata:
            if read_flag:
                at+=1
                mtype = j_line[5:10].strip(' ')
                
                _mol_.append( j_line[:5].lstrip(' '))
                _mtype_.append(mtype)
                _type_.append(j_line[10:15].lstrip(' '))
                _x_.append( float( j_line[20:28]) )
                _y_.append( float( j_line[28:36]) )
                _z_.append( float( j_line[36:44]) )
                
                if _buffer==[]:
                    _buffer = [ mtype, at]
                
                elif mtype != _buffer[0]:
                    
                    _buffer += [at-1]
                    g_names.append( _buffer)
                    _buffer = [ mtype, at]
                
                if at == at_num:
                    read_flag = False
                    g_names.append(_buffer + [at])
                    
            elif j_line.startswith(';'):
                pass
            elif at_num == 0:
                #j_line = indata.next()
                j_line = next(indata)
                at_num = int( j_line)
                read_flag = True
            elif at == at_num:
                box_xyz_hi = [float(x) for x in j_line.split(';')[0].split()]
                if len( box_xyz_hi) in [ 3, 9]:
                    _corrupt = False
                    
                
    if at_num != len(_type_):
        print('Atom number mismatch in .gro file')
        return False, 0 ,0
    elif _corrupt:
        print('Corrupt .gro file box definition')
        return False, 0,0
    else:
        _xyz_ = np.array([ _x_, _y_, _z_])
        _xyz_ = np.transpose(_xyz_)*10
        return True, [np.array(_mol_,dtype=int), _mtype_, _type_, _xyz_, g_names], np.array(box_xyz_hi)*10

def find_label(input_str):
    #num = ''.join([x for x in input_str if x.isdigit()])
    label = ''.join([x for x in input_str if x.isalpha()])
    
    return label
    
    

if __name__ == "__main__":
    a, b, c = get_gro_fixed_line("H:\PSHJ\initial model\genpolymorph\polymorph\idhj_1_1.gro")
import numpy as np
import copy
from polymorph.util import abc2xyz, rotation

class Polymer:
    def __init__(self, monomer, n, name, finite=True, reorentation = False):

        self.n = n
        self.name = name
        self.finite = finite
        
        monomers = []
        if reorentation:
            monomer.reorentation()

        for i in range(n):
            if i >= 1:
                m = monomer.trans_along_vec(i)
            else:
                m = monomer
            monomers.append(m)
        self.monomers = monomers
        
    def append(self, monomer):
        if self.finite:
            print('Can not append monomer for an endcapped polymer chain!')
        else:
            self.n += 1
            self.monomers.append(monomer)
        
    def to_gjf(self,filename='polymer.gjf'):
        style = "{:5s} {:10.6f} {:10.6f} {:10.6f}\n"
        if self.finite:
            self.endcapping()
        
        with open(filename,'w') as f:
            f.write("# HF/sto-3g\n\ncommand\n\n0 1\n")
            for i in range(self.n):
                m = self.monomers[i]
                for a in range(m.atom):
                    f.write(style.format(
                        m.labels[a], m.coords[a,0], m.coords[a,1], m.coords[a,2]
                    ))
            f.write('')
    
    def to_gro(self,filename='polymer.gro', a=0, b=0, c=0, alpha=90, beta=90, gamma=90):
        # resnum, resname, atomname, atomnumber, x, y, z
        style = "{:>5d}{:<5s}{:>5s}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}\n"

        if self.finite:
            self.endcapping()
        
        (xx,yy,zz,xy,xz,yz) = abc2xyz(a,b,c,alpha,beta,gamma)
        
        with open(filename,'w') as f:
            f.write("{}\n".format(self.name))
            f.write(" {}\n".format(self.atom))
                
            na = 1
            for i in range(self.n):
                m = self.monomers[i]
                for a in range(m.atom):
                    f.write(style.format(
                        i+1, self.name, m.atom_names[a], na, 
                        m.coords[a,0]/10, m.coords[a,1]/10, m.coords[a,2]/10
                    ))
                    na += 1
                    
            f.write('{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}\n'.format(
                xx,yy,zz,0,0, xy, 0, xz, yz))
    
    def move(self, x=0, y=0, z=0):
        p = copy.deepcopy(self)
        for i in range(p.n):
            p.monomers[i] = p.monomers[i].move(x=x,y=y,z=z)
        return p
        
    def endcapping(self):
        m1 = self.monomers[0]
        m2 = self.monomers[-1]
        
        if not m1.is_head:
            m1.append_H(0)
            
        if not m2.is_tail:
            m2.append_H(1)


    @property
    def atom(self):
        monomers = self.monomers
        na = 0
        for i in range(self.n):
            na += monomers[i].atom
        return na
    
    @property
    def size(self):
        region = self.region
        return region[1,:] - region[0,:]
        
    @property
    def region(self):
        coords = self.coords
        return np.vstack([np.min(coords, axis=0), 
                          np.max(coords, axis=0)])
    
    @property
    def geo_center(self):
        return np.average(self.coords, axis=0)
        

    @property
    def coords(self):
        coords = []
        for m in self.monomers:
            coords.append(m.coords)
        
        return np.vstack(coords)
    
class Monomer:
    def __init__(self):
        self.is_head = False
        self.is_tail = False
        pass
    
        
    @classmethod
    def from_file(cls,filename):
        
        infile = open(filename,'r')
        labels = []
        coords = []

        lines = infile.readlines()

        na = 0
        for line in lines:
            label,x,y,z = line.split()[:]
            labels.append(label)
            coords.append([float(x),float(y),float(z)])
            na += 1
        coords = np.array(coords)

        m = cls()  

        m.labels = labels[1:na-1]
        m.coords = coords[1:na-1]
        m.ends = np.zeros((2,3))
        m.ends[0] = coords[0]
        m.ends[1] = coords[-1]
        m.length = np.linalg.norm(m.vector)
        m.atom_names = [ labels[a]+str(a) for a in range(1,na-1)]
        
        return m
    
    @classmethod
    def from_dict(cls, molinfo):
        m = cls()
        m.labels = molinfo['labels']
        m.coords = molinfo['coords']
        m.atom_names = molinfo['atom_names']
        m.ends = np.zeros((2,3))
        
        return m
    
    
    @property
    def atom(self):
        return len(self.coords)
    
    @property
    def vector(self):
        return self.ends[1] - self.coords[0]
    
    @property
    def mol_plane(self):
        x = self.coords[:,0]
        y = self.coords[:,1]
        z = self.coords[:,2]
        x0 = np.mean(x)
        y0 = np.mean(y)
        z0 = np.mean(z)
        x = x - x0
        y = y - y0
        z = z - z0

        A = np.array([[sum(x * x), sum(x * y), sum(x * z)],
                      [sum(x * y), sum(y * y), sum(y * z)],
                      [sum(x * z), sum(y * z), sum(z * z)]])
        [D, X] = np.linalg.eig(A)

        return X[:,2]
    
    
    def move(self, x=0, y=0, z=0):
        vector = np.array([x,y,z])
        return self.translation(vector)
        
    def reorentation(self):
        '''
        Standard orentation:
        long axis along x
        molecular plane along y
        '''
        
        coords = self.coords
        ends = self.ends
        
        # rotate along long axis
        x = np.array([1,0,0])
        Rx = rotation(self.vector, x)
        
        for i in range(self.atom):
            coord=np.dot(Rx,coords[i,:3])
            coords[i,:3] = coord
        ends[0] = np.dot(Rx,ends[0])
        ends[1] = np.dot(Rx,ends[1])
        
        # rotate along short axis
        y = np.array([0,1,0])
        norm_vec = self.mol_plane
        Ry = rotation(norm_vec, y)
        
        for i in range(self.atom):
            coord=np.dot(Ry,coords[i,:3])
            coords[i,:3] = coord
        ends[0] = np.dot(Ry,ends[0])
        ends[1] = np.dot(Ry,ends[1])       
        
        # rotate along long axis, again!
        vector = self.vector
        Rx = rotation(vector, x)        
        for i in range(self.atom):
            coord=np.dot(Rx,coords[i,:3])
            coords[i,:3] = coord
        ends[0] = np.dot(Rx,ends[0])
        ends[1] = np.dot(Rx,ends[1])
        
        # translate to (0,0,0)
        self.coords = coords-ends[0]
        self.ends = ends-ends[0]

    def rotation_along_x(self, angle, deg=True):
        if deg:
            theta = angle/180*np.pi
        else:
            theta = angle

        R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)]
        ])

        coords = self.coords
        ends = self.ends
        for i in range(self.atom):
            coord=np.dot(R,coords[i,:3])
            coords[i,:3] = coord
            ends[0] = np.dot(R,ends[0])
            ends[1] = np.dot(R,ends[1])

    def rotation_along_z(self, angle, deg=True):
        if deg:
            theta = angle/180*np.pi
        else:
            theta = angle

        R = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        coords = self.coords
        ends = self.ends
        for i in range(self.atom):
            coord=np.dot(R,coords[i,:3])
            coords[i,:3] = coord
            ends[0] = np.dot(R,ends[0])
            ends[1] = np.dot(R,ends[1])


    def trans_along_vec(self, n):
        vector = self.vector
        m = self.translation(vector*n)
        return m
    
    def translation(self, vector):
        m = copy.deepcopy(self)
        m.coords += np.array(vector)
        m.ends += np.array(vector)
        return m

    def append_H(self,pos=0):
        '''
        pos: 0-left; 1-right
        '''
        pos = int(pos)

        if not pos:
            coord = self.coords
            end = self.ends[0] 
            self.coords = np.vstack([end,coord])
            self.labels.insert(0,'H')
            self.is_head = True
            self.atom_names.insert(0,'HV')

        else:
            coord = self.coords
            end = self.ends[1] 
            self.coords = np.vstack([coord,end])
            self.labels.append('H')
            self.is_tail = True
            self.atom_names.append('HV')
        
    def to_gjf(self, filename='monomer.gjf'):
        style = "{:10s} {:10.6f} {:10.6f} {:10.6f}\n"
        
        with open(filename,'w') as f:
            f.write("# HF/sto-3g\n\ncommand\n\n0 1\n")
            for a in range(self.atom):
                coord = self.coords[a]
                label = self.labels[a]
                f.write(style.format(
                    label, coord[0], coord[1], coord[2]
                ))
            f.write('')

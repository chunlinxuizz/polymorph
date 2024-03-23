import numpy as np
import copy
from polymorph.util import (
    abc2xyz, 
    xyz2abc, 
    get_gro_fixed_line, 
    find_label,
)
from polymorph.polymer import Polymer, Monomer

class Box:
    def __init__(self, a=0, b=0, c=0, alpha=90, beta=90, gamma=90):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.molecules = []
        self.nbox = np.array([1,1,1])
        
        
    @classmethod
    def gen_box(cls, molecule, nbox=[1,1,1], abc=[0,0,0,90,90,90]):
        if len(abc) == 3:
            (a,b,c) = abc
            alpha = 90
            beta = 90
            gamma = 90
        elif len(abc) == 6:
            (a,b,c,alpha,beta,gamma) = abc
        else:
            abc = [0,0,0,90,90,90]
        
        if a*b*c == 0:
            (a,b,c) = list(molecule.size)
            
        box = cls(a*nbox[0],b*nbox[1],c*nbox[2],alpha,beta,gamma)
        
        xx,yy,zz,xy,xz,yz = abc2xyz(a,b,c,alpha,beta,gamma)
        
        abc = np.array([
            [xx, xy, xz],
            [0,yy,yz],
            [0,0,zz]
        ])

        for i in range(nbox[0]):
            for j in range(nbox[1]):
                for k in range(nbox[2]):
                    
                    ABC = copy.deepcopy(abc*np.array([i,j,k]))
                    m = copy.deepcopy(molecule)
                    for n in range(3):
                        m = m.move(
                                ABC[0,n], ABC[1,n], ABC[2,n]
                            )
                    box.append(m)
                    
        box.nbox = np.array(nbox)
        
        return box
    
    @classmethod
    def from_gro(cls,filename):
        
        _ok_flag_, groinfo, _xyz_ = get_gro_fixed_line(filename)
        
        _mol_ = groinfo[0]
        names = groinfo[1]
        atom_names = groinfo[2]
        coords = groinfo[3]
        na = len(coords)
        
        labels = [find_label(x) for x in atom_names]
        
        _, start_end = np.unique(_mol_, return_index=True)
        start_end = start_end.tolist()
        start_end.append(na)
        
        if len(_xyz_) > 3:
            (a,b,c,alpha,beta, gamma) = tuple(
                  xyz2abc(_xyz_[0],_xyz_[1],_xyz_[2],_xyz_[5],_xyz_[7],_xyz_[8])
                             )
            BOX = cls(a,b,c,alpha,beta, gamma)
        
        else:
            BOX = cls(_xyz_[0],_xyz_[1],_xyz_[2])
        
        atom_counter = 0
        nmol = len(np.unique(_mol_))
        
        for p in range(nmol):
            molinfo = {}
            
            start = start_end[p]
            end = start_end[p+1]
            molinfo['labels'] = labels[start:end]
            molinfo['atom_names'] = atom_names[start:end]
            molinfo['coords'] = coords[start:end,:]
            
            m1 = Monomer.from_dict(molinfo)

            name = names[atom_counter]
            p1 = Polymer(m1,1,name,finite=False,reorentation = False)

            BOX.append(p1)
            
            atom_counter = start_end[p+1]
            
        return BOX
            
    
    @classmethod
    def add(cls, box1, box2):
        a1, b1, c1 = list(box1.abc[:3])
        a2, b2, c2 = list(box2.abc[:3])
        
        Rx, Ry, Rz = list(np.abs(box2.region[0] - box1.region[0]))
        a = max(a1, a2+Rx)
        b = max(b1, b2+Ry)
        c = max(c1, c2+Rz)
        
        box = cls(a, b, c)
        
        for p in box1.molecules:
            box.append(p)
        for p in box2.molecules:
            box.append(p)
        return box
    
    @classmethod
    def addz(cls,box1,box2):
        a1, b1, c1 = list(box1.abc[:3])
        a2, b2, c2 = list(box2.abc[:3])
        a = max(a1,a2)
        b = max(b1,b2)
        c = c1+c2
        box = cls(a, b, c)

        for p in box1.molecules:
            box.append(p)
        for p in box2.molecules:
            box.append(p)
        return box
    
    @classmethod
    def merge(cls, boxes):
        
        BOX = boxes[0]
        for i in range(1, len(boxes)):
            BOX = cls.add(BOX, boxes[i])

        return BOX
    
    @classmethod
    def extend_chain_length(cls, box, na=1):
        '''
        extend all polymer chains along lattice a
        '''
        vec_a = box.lattice[:,0]
        abc = box.abc
        
        BOX = cls(abc[0]*na,abc[1],abc[2],abc[3],abc[4],abc[5])
        
        for p, polymer in enumerate(box.molecules):
            BOX.append(polymer)

            for i in range(1,na):
                vec = vec_a*i
                p1 = polymer.move(vec[0],vec[1],vec[2])
                [ BOX.molecules[p].append(m) for m in p1.monomers ]
         
        return BOX

    
    @classmethod
    def super_box(cls, box, na=1, nb=1, nc=1):

        #box = cls.extend_chain_length(box, na)
        lattice = box.lattice
        abc = box.abc
        
        BOX = cls(abc[0]*na,abc[1]*nb,abc[2]*nc,abc[3],abc[4],abc[5])
        a_vec = lattice[:,0]
        b_vec = lattice[:,1]
        c_vec = lattice[:,2]

        for p in box.molecules:
            for k in range(nc):
                for j in range(nb):
                    for i in range(na):
                        vec = a_vec*i + b_vec*j + c_vec*k
                        p1 = p.move(vec[0],vec[1],vec[2])
                        BOX.append(p1)
        
        return BOX
    
    def append(self, molecule):
        if molecule.finite:
            molecule.endcapping()
        self.molecules.append(molecule)
    
    def to_gro(self,filename='box.gro'):
        # resnum, resname, atomname, atomnumber, x, y, z
        style = "{:>5d}{:<5s}{:>5s}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}\n"
        
        with open(filename,'w') as f:
            f.write("{}\n".format('Generated by PolyMorph'))
            f.write(" {}\n".format(self.atom))
                
            na = 1
            nr = 1
            
            for t in self.mol_types:
                for p in self._type(t):
                    for j in range(p.n):
                        m = p.monomers[j]

                        for a in range(m.atom):
                            f.write(
                                style.format(
                                nr, p.name, m.atom_names[a], na%100000, 
                                m.coords[a,0]/10, m.coords[a,1]/10, m.coords[a,2]/10
                                )
                            )
                            na += 1
                            
                        nr += 1
                        
            (xx,yy,zz,xy,xz,yz) = self.xyz
            f.write('{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}\n'.format(
                xx/10,yy/10,zz/10,0,0, xy/10, 0, xz/10, yz/10))
        
        
    def move(self, x=0, y=0, z=0):
        box = copy.deepcopy(self)
        for i in range(box.molecule):
            box.molecules[i] = box.molecules[i].move(x=x,y=y,z=z)
        return box
    
    def origin(self):
        x,y,z = list(self.region[0])
        return self.move(-x,-y,-z)
     
    def recenter(self):
        x,y,z = list(self.center)
        x0,y0,z0 = list(np.average(self.coords, axis=0))
        return self.move(x-x0,y-y0,z-z0)
    
    def inversion(self):
        box = self.origin().recenter()
        x,y,z = list(self.center)
        
        box = box.move(-x,-y,-z)
        
        for p in box.molecules:
            for m in p.monomers:
                m.coords = m.coords*(-1)
        
        box = box.move(x,y,z)
        
        return box
        
    def _type(self, name):
        molecules = []
        for p in self.molecules:
            if p.name == name:
                molecules.append(p)
                
        return molecules
    
    
    @property
    def center(self):
        xx,yy,zz,xy,xz,yz = self.xyz
        x = (xx + xy + xz)/2
        y = (yy + yz)/2
        z = zz/2
        return np.array([x,y,z])
    
    
    @property
    def molecule(self):
        return len(self.molecules)
    
    @property
    def monomer(self):
        m = 0
        for p in self.molecules:
            m += p.n
        return m
    
    @property
    def atom(self):
        na = 0
        for p in self.molecules:
            na += p.atom
        return na
    
    @property
    def abc(self):
        return np.array([self.a, self.b, self.c,
                         self.alpha, self.beta, self.gamma])
    
    @property
    def xyz(self):
        (a,b,c,alpha,beta,gamma) = list(self.abc)
        return np.array(
            abc2xyz(a,b,c,alpha,beta,gamma)
            )
    
    @property
    def lattice(self):
        xyz = self.xyz
        return np.array([
                [xyz[0],xyz[3],xyz[4]],
                [0, xyz[1], xyz[5]],
                [0, 0, xyz[2]]
            ])
    
    
    @property
    def region(self):
        regions = []
        for p in self.molecules:
            regions.append(p.region)
        regions = np.vstack(regions)
        return np.vstack([np.min(regions, axis=0), 
                          np.max(regions, axis=0)])
    
    @property
    def size(self):
        region = self.region
        return region[1,:] - region[0,:]
        
    @property
    def coords(self):
        coords = []
        for p in self.molecules:
            coords.append(p.coords)
        return np.vstack(coords)
        
    @property
    def mol_types(self):
        names = []
        for p in self.molecules:
            if p.name not in names:
                names.append(p.name)
        return sorted(names)

    @property
    def xx(self):
        return self.a

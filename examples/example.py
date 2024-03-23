import numpy as np
import sys,os
from polymorph.polymer import Polymer, Monomer
from polymorph.box import Box, treat_ends, exchange_molecules

def mixed_interface():
    
    Nswap = 10
    gap = 18
    NL = 6
    pipi = 5.0
    Npp = 20
    
    # PBTTT
    m1 = Monomer.from_file('pbttt.xyz')
    p1 = Polymer(m1,7,'BTT',finite=True)
    
    # PDPPSe
    m2 = Monomer.from_file('pdppse.xyz')
    p2 = Polymer(m2,3,'DPP',finite=True)
    
    ibtt = Box.gen_box(p1,nbox=(1,Npp,1),abc=[100., pipi, 20]).recenter()
    ibtt = treat_ends(ibtt, 3)
    
    idpp = Box.gen_box(p2,nbox=(1,Npp,1),abc=[100., pipi, 25]).recenter()
    idpp = treat_ends(idpp, 1).move(z =20+gap)
    
    exchange_molecules(ibtt, idpp, Nswap)
    
    int1 = Box.add(ibtt,idpp)
    
    int2 = int1.inversion()
    
    btt = Box.gen_box(p1,nbox=(1,Npp,NL),abc=[100., pipi, 20]).recenter()
    btt = treat_ends(btt,3)
    
    dpp = Box.gen_box(p2,nbox=(1,Npp,NL),abc=[100., pipi, 25]).recenter()
    dpp = treat_ends(dpp,1)
    #  pdppse
    #-------------      gap
    #  int1      |    20+25+gap
    #-------------      gap
    #  pbttt     |     NL*20
    #-------------      gap
    #  int2      |    20+25+gap
    #-------------      gap
    #  pdppse    |     NL*25
    #-------------
    
    int1 = int1.move(z=NL*25 + 45 + NL*20 + 4*gap)
    btt = btt.move(z=NL*25 + 45 + 3*gap)
    int2 = int2.move(z=NL*25 + gap)
    dpp = dpp.move( z=0 )
    
    BOX = Box.merge([dpp,btt,int1,int2])
    
    BOX.a = 100
    BOX.b = pipi*Npp
    BOX.c = (NL+2)*(25+20) + 6*gap
    
    BOX.to_gro("polymorph.gro")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    mixed_interface()
import sys
sys.path.append(".")
sys.path.append("..")

import flowermd # wrapper we use on top of hoomd
import gsd # to work with simulation data files
import gsd.hoomd # to work with simulation data files directly coming from hoomd sims
import hoomd # MD simulation engine used for initializing and running sims
import mbuild as mb # library used to build Molecule objects
import numpy as np # common library used for a plethora of things, such as indexing, numeric constants, linear algebra, etc.
import warnings # used to provide the user warnings rather than errors
from flowermd.base import Pack, Simulation, System, Molecule # importing useful functions from flowermd to simplify initializing and running 
from flowermd.library import LJChain # lennard jones polymer chains with scalable length
from flowermd.library.forcefields import BeadSpring # forcefield template used in order to define interactions within sim
from flowermd.utils import get_target_box_number_density # used to define final density to shrink box to in order to minimize space
from mbuild.compound import Compound # base compound object to create molecule geometry
from mbuild.lattice import Lattice # lattice object to initialize lattice spacing and points
import unyt as u # module used for unit definitions
warnings.filterwarnings('ignore') # ignore warnings

class Flake(System):
    def __init__(
        self,
        x_repeat,
        y_repeat,
        n_layers,
        base_units=dict(),
        periodicity=(True, True, False),
    ):
        surface = mb.Compound(periodicity=periodicity)
        a = 3**.5
        lattice = Lattice(
            lattice_spacing=[a,a,a],
            lattice_vectors=  [[a,0,0],[a/2,3/2,0],[0,0,1]],
            lattice_points={"A": [[1/3,1/3,0], [2/3, 2/3, 0]]},
        ) # define lattice vectors, points, and spacings for flakes
        Flakium = Compound(name="F", element="F") # defines an atom that will be used to populate lattice points
        layers = lattice.populate(
            compound_dict={"A": Flakium}, x=x_repeat, y=y_repeat, z=n_layers
        ) # populates the lattice using the previously defined atom for every "A" site, repeated in all x,y, and z directions
        surface.add(layers) # adds populated flake lattice layers to the 'surface' compound, which represents our flake structure 
        surface.freud_generate_bonds("F", "F", dmin=0.9, dmax=1.1) # generates bonds depending on input distance range, scales with lattice
        surface_mol = Molecule(num_mols=1, compound=surface) # wraps into a Molecule object, creating "1" instance of this molecule

        super(Flake, self).__init__(
            molecules=[surface_mol],
            base_units=base_units,
        )

    def _build_system(self):
        return self.all_molecules[0]

ff = BeadSpring(
    r_cut=2**(1/6),  # r_cut value defines the radius in which a given particle will interact with another.
    beads={
        "A": dict(epsilon=1.0, sigma=1.0),  # chains, epsilon = well depth, defines strength of attractive forces between two molecules
        "F": dict(epsilon=1.0, sigma=1.0),  # flakes, sigma = distance between two particles where PE is zero
    },
    bonds={
        "F-F": dict(r0=1.0, k=1000),
        "A-A": dict(r0=1.0, k=1000.0),  # r0 = equilibrium distance of bonded particles, k = stiffness constant
    },
    angles={
        "A-A-A": dict(t0=2* np.pi / 3., k=100.0),   
        "F-F-F": dict(t0=2 * np.pi / 3., k=5000),
    },
    dihedrals={
        "A-A-A-A": dict(phi0=0.0, k=0, d=-1, n=2), # do not worry about dihedrals
        "F-F-F-F": dict(phi0=0.0, k=500, d=-1, n=2),
    }
)




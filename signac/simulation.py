import time
import warnings
import flowermd
import gsd
import gsd.hoomd
import hoomd
import mbuild as mb
import numpy as np
import unyt as u
from flowermd.base import Pack, Simulation, System, Molecule
from flowermd.library import LJChain
from flowermd.library.forcefields import BeadSpring
from flowermd.utils import get_target_box_number_density
from mbuild.compound import Compound
from mbuild.lattice import Lattice
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

N_chains = 100          # number of polymer chains
initial_dens = 0.001    # initial packing density
final_dens = 0.3       # final packing density after shrink
N_flakes = 10           # number of flakes
chain_length = 10       # chain length
dt = 0.0005             # time step
temp = 5.0              # kT for production run

write_frequency = 50000                     # trajectory/log write frequency
shrink_steps = int(1e6)                     # steps for shrink phase
sim_start = int(shrink_steps / write_frequency)  # for analysis notebook
steps = int(25e6)                            # steps for production NVT run
flake_repeat = 5                            # number of lattice repeats in x,y for flakes


# Device: change to hoomd.device.GPU() if desired
device = hoomd.device.GPU()


# ---------------------------------------------------------------------------
# Flake geometry
# ---------------------------------------------------------------------------


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
        a = 3 ** 0.5

        lattice = Lattice(
            lattice_spacing=[a, a, a],
            lattice_vectors=[[a, 0, 0], [a / 2, 3 / 2, 0], [0, 0, 1]],
            lattice_points={"A": [[1 / 3, 1 / 3, 0], [2 / 3, 2 / 3, 0]]},
        )  # define lattice vectors, points, and spacings for flakes

        Flakium = Compound(
            name="F", element="F"
        )  # defines an atom that will be used to populate lattice points

        layers = lattice.populate(
            compound_dict={"A": Flakium},
            x=x_repeat,
            y=y_repeat,
            z=n_layers,
        )  # populates lattice for every "A" site, repeated in x, y, z

        surface.add(
            layers
        )  # adds populated flake lattice layers to the 'surface' compound

        surface.freud_generate_bonds(
            "F", "F", dmin=0.9, dmax=1.1
        )  # generates bonds based on distance range

        surface_mol = Molecule(
            num_mols=1,
            compound=surface,
        )  # wraps into a Molecule object (one flake)

        super(Flake, self).__init__(
            molecules=[surface_mol],
            base_units=base_units,
        )

    def _build_system(self):
        return self.all_molecules[0]


# ---------------------------------------------------------------------------
# Forcefield: Weeks–Chandler–Andersen-like BeadSpring
# ---------------------------------------------------------------------------

ff = BeadSpring(
    r_cut=2 ** (1 / 6),  # r_cut defines the radius within which particles interact
    beads={
        "A": dict(
            epsilon=1.0,
            sigma=1.0,
        ),  # chains, epsilon = well depth, strength of attraction
        "F": dict(
            epsilon=1.0,
            sigma=1.0,
        ),  # flakes, sigma = distance where PE is zero
    },
    bonds={
        "F-F": dict(r0=1.0, k=1000),
        "A-A": dict(
            r0=1.0,
            k=1000.0,
        ),  # r0 = equilibrium distance, k = stiffness
    },
    angles={
        "A-A-A": dict(t0=2 * np.pi / 3.0, k=100.0),
        "F-F-F": dict(t0=2 * np.pi / 3.0, k=5000),
    },
    dihedrals={
        # do not worry about dihedrals for chains here
        "A-A-A-A": dict(phi0=0.0, k=0, d=-1, n=2),
        "F-F-F-F": dict(phi0=0.0, k=500, d=-1, n=2),
    },
)

# ---------------------------------------------------------------------------
# Build system
# ---------------------------------------------------------------------------

kg_chain = LJChain(
    lengths=chain_length,
    num_mols=N_chains,
)  # polymer chains

sheet = Flake(
    x_repeat=flake_repeat,
    y_repeat=flake_repeat,
    n_layers=1,
    periodicity=(False, False, False),
)  # flakes (non-periodic in all directions)

system = Pack(
    molecules=[
        Molecule(compound=sheet.all_molecules[0], num_mols=N_flakes),
        kg_chain,
    ],
    density=initial_dens,
    packing_expand_factor=6,
    seed=2,
)  # pack chains + flakes into initial box

snapshot = system.hoomd_snapshot           # extract bead count
n_beads = snapshot.particles.N

target_box = get_target_box_number_density(
    density=final_dens * u.Unit("nm**-3"),
    n_beads=n_beads,
)  # final box size for target number density


# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------

gsd_file = f"{N_chains}_{chain_length}mer{N_flakes}f_{dt}dt_{final_dens}_dens.gsd"
log_file = f"{N_chains}_{chain_length}mer{N_flakes}f_{dt}dt_{final_dens}_dens.txt"
start_file = f"{N_chains}_{chain_length}mer{N_flakes}f_{dt}dt_{final_dens}_dens_start.txt"


# ---------------------------------------------------------------------------
# Run simulation: shrink -> NVT
# ---------------------------------------------------------------------------

sim = Simulation(
    initial_state=system.hoomd_snapshot,
    forcefield=ff.hoomd_forces,
    device=device,
    dt=dt,
    gsd_write_freq=int(write_frequency),
    log_file_name=log_file,
    gsd_file_name=gsd_file,
)

start_shrink = time.time()
sim.run_update_volume(
    final_box_lengths=target_box,
    kT=6.0,
    n_steps=shrink_steps,
    tau_kt=100 * sim.dt,
    period=10,
    thermalize_particles=True,
)
end_shrink = time.time()

start_run = time.time()
sim.run_NVT(
    n_steps=steps,
    kT=temp,
    tau_kt=dt * 100,
)
end_run = time.time()

sim.flush_writers()
# allow files to fully close
del sim


# ---------------------------------------------------------------------------
# Save helpful metadata to a text file
# ---------------------------------------------------------------------------

flake_mol = sheet.all_molecules[0]
beads_per_flake = flake_mol.n_particles

with open(start_file, "w") as f:
    f.write(f"{sim_start}\n")
    f.write(f"{beads_per_flake}\n")
    f.write(f"{(end_run - start_run) + (end_shrink - start_shrink)}\n")

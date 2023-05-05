import numpy as np
import stdpopsim
import os
from dataclasses import dataclass
import multiprocessing.pool as mp
from functools import partial


@dataclass
class MsprimeSimulation:
    seed: int
    num_reps: int
    sp_name: str
    model_name: str
    sims_root_path: str
    chrom: str = None
    length: int = None
    sample_size: int = 10
    n_workers: int = 8
    engine: str = "msprime"

    def __post_init__(self):
        # Generating seeds
        rng = np.random.default_rng(self.seed)
        self.seed_array = rng.integers(1, 2**31, self.num_reps)
        # Path parameters
        if self.chrom is None:
            contig = self.length
        else:
            assert self.chrom is not None
            contig = self.chrom
        self.sims_path = (
            f"{self.sims_root_path}{self.sp_name}/{contig}/{self.model_name}/"
        )
        os.makedirs(self.sims_path, exist_ok=True)  # ensuring that sims_path exists
        self.ts_paths = [
            f"{self.sims_path}/sim_{seed}.trees" for seed in self.seed_array
        ]
        # Objects to be used in the simulation
        self.engine = stdpopsim.get_engine(self.engine)
        self.species = stdpopsim.get_species(self.sp_name)
        self.model = self.species.get_demographic_model(self.model_name)
        self.contig = self.species.get_contig(
            chromosome=self.chrom,
            mutation_rate=self.model.mutation_rate,
            length=self.length,
        )
        self.samples = {pop.name: self.sample_size for pop in self.model.populations}


def _run(seed, sim):
    tspath = f"{sim.sims_path}/sim_{seed}.trees"
    if not os.path.exists(tspath):
        ts = sim.engine.simulate(sim.model, sim.contig, sim.samples, seed=seed)
        ts.dump(tspath)
    return tspath


def run_sims(sim):
    run = partial(_run, sim=sim)
    with mp.ThreadPool(sim.n_workers) as p:
        results = p.map(run, sim.seed_array)
    assert list(results) == sim.ts_paths

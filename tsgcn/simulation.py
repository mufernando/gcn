import numpy as np
import stdpopsim
import os
from dataclasses import dataclass
import multiprocessing.pool as mp


@dataclass
class MsprimeSimulation:
    seed: int
    num_reps: int
    sp_name: str
    chrom: str
    model_name: str
    sims_root_path: str
    sample_size: int = 10
    n_threads: int = 6
    engine: str = "msprime"

    def __post_init__(self):
        # Generating seeds
        rng = np.random.default_rng(self.seed)
        self.seed_array = rng.integers(1, 2**31, self.num_reps)
        # Path parameters
        self.sims_path = (
            f"{self.sims_root_path}{self.sp_name}/{self.chrom}/{self.model_name}/"
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
            self.chrom, mutation_rate=self.model.mutation_rate
        )
        self.samples = {pop.name: self.sample_size for pop in self.model.populations}
        self.run()

    def run_sim(self, seed):
        tspath = f"{self.sims_path}/sim_{seed}.trees"
        if not os.path.exists(tspath):
            ts = self.engine.simulate(self.model, self.contig, self.samples, seed=seed)
            ts.dump(tspath)
        return tspath

    def run(self):
        with mp.ThreadPool(self.n_threads) as p:
            results = list(p.imap(self.run_sim, self.seed_array))
        assert results == self.ts_paths

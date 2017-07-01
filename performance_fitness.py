from magenta.models.performance_rnn import performance_model
from magenta.models.performance_rnn import performance_sequence_generator
import magenta


class FitnessFunction:

    def __init__(self, bundle_file):
        config = performance_model.default_configs['performance_with_dynamics']
        self.generator = performance_sequence_generator.PerformanceRnnSequenceGenerator(
            model=performance_model.PerformanceRnnModel(config),
            details=config.details,
            bundle=magenta.music.read_bundle_file(bundle_file)
        )

        self.generator.initialize()

    def evaluate_fitness(self):
        pass

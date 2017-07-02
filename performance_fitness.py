from magenta.models.performance_rnn.performance_lib import PerformanceEvent
from magenta.protobuf import music_pb2
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
        sequence = [
            PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=60),
            PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=0),
            PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=127),
            PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=72),
            PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=0),
            PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=127),
            PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT, event_value=10),
            PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT, event_value=1),
            PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT, event_value=100),
            PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=5),
            PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=1),
            PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=16),
        ]
        return self.generator._model.performance_log_likelihood(sequence)

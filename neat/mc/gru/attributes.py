from neat.attributes import BaseAttribute, FloatAttribute # Assuming FloatAttribute exists
from random import choice, gauss, random, uniform, randint
from neat.activations import ActivationFunctionSet

class GRUCell:
    def __init__(self,config,activation_function_def):
        for param_name, param_info in config.items():
            param_type, default_value = param_info
            setattr(self, param_name, default_value)

        for param_name, param_info in config.items():
            param_type, default_value = param_info
            if "_w" in param_name or "_bias" in param_name:
                if self.init_type == "gaussian":
                    value = gauss(self.init_mean, self.init_stdev)
                    value = min(max(value, self.min_value), self.max_value)
                elif self.init_type == "uniform":
                    min_value = max(self.min_value,(self.init_mean - (2 * self.init_stdev)))
                    max_value = min(self.max_value,(self.init_mean + (2 * self.init_stdev)))
                    value = uniform(min_value, max_value)
                else:
                    raise RuntimeError(f"Unknown init_type {self.init_type_name!r} for {self.init_type_name!s}")
                if param_type == int:
                    value = int(value)
                setattr(self,param_name,value)
            elif "_activation" in param_name:
                if param_type == str:
                    setattr(self, param_name, activation_function_def.get(default_value))
                else:
                    raise RuntimeError(f"Invalid type for activation function: {param_type}")
        self.update_vector = 0
        self.reset_vector = 0
        self.last_candidate_vector = 0
        self.candidate_vector = 0

    def mutate(self, config):
        for param_name, param_info in config.items():
            param_type, default_value = param_info
            if "_w" in param_name or "_bias" in param_name:
                if self.init_type == "gaussian":
                    value = gauss(self.init_mean, self.init_stdev)
                    value = min(max(value, self.min_value), self.max_value)
                elif self.init_type == "uniform":
                    min_value = max(self.min_value, (self.init_mean - (2 * self.init_stdev)))
                    max_value = min(self.max_value, (self.init_mean + (2 * self.init_stdev)))
                    value = uniform(min_value, max_value)
                else:
                    raise RuntimeError(
                        f"Unknown init_type {self.init_type!r} for {self.init_type!s}")
                if param_type == int:
                    value = int(value)
                setattr(self, param_name, value)

    def distance(self, other):
        """
        Calculate distance between two GRU cells.
        """
        d = 0.0  # Use float for distance calculation

        l = abs(self.max_value - self.min_value)

        # Calculate distance for numerical parameters (weights and biases)
        d += abs(self.input_to_reset_w - other.input_to_reset_w)
        d += abs(self.input_to_update_w - other.input_to_update_w)
        d += abs(self.input_to_candidate_w - other.input_to_candidate_w)

        d += abs(self.hidden_to_reset_w - other.hidden_to_reset_w)
        d += abs(self.hidden_to_update_w - other.hidden_to_update_w)
        d += abs(self.hidden_to_candidate_w - other.hidden_to_candidate_w)

        d += abs(self.reset_bias - other.reset_bias)
        d += abs(self.update_bias - other.update_bias)
        d += abs(self.candidate_bias - other.candidate_bias)

        d /= l  # Normalize by the range of values

        # Add 1 to distance if activation functions are different
        if self.reset_activation != other.reset_activation:
            d += 1
        if self.update_activation != other.update_activation:
            d += 1
        if self.candidate_activation != other.candidate_activation:
            d += 1

        return d

    def calc(self, input):
        """
        Calculate the output of the GRU cell given an input.
        This method should implement the GRU update equations.
        """
        self.last_candidate_vector = self.candidate_vector
        self.update_vector = self.update_activation(self.input_to_update_w * input + self.hidden_to_update_w * self.last_candidate_vector + self.update_bias)
        self.reset_vector = self.reset_activation(self.input_to_reset_w * input + self.hidden_to_reset_w * self.last_candidate_vector + self.reset_bias)
        self.candidate_vector = self.candidate_activation(self.input_to_candidate_w * input + self.hidden_to_candidate_w * (self.reset_vector * self.last_candidate_vector) + self.candidate_bias)
        return (1 - self.update_vector) * self.last_candidate_vector + self.update_vector * self.candidate_vector

class GRUAttribute(BaseAttribute):
    """
    Represents a GRU (Gated Recurrent Unit) memory cell within a NEAT genome.
    This node type has internal weights and biases for its gates and candidate state.
    """
    _config_items = {
        # GRU-specific parameters
        # Input weights (from network connections to GRU gates/candidate)
        "input_to_reset_w": [float, 0],
        "input_to_update_w": [float, 0],
        "input_to_candidate_w": [float, 0],

        # Hidden state weights (recurrent connections within the GRU cell)
        "hidden_to_reset_w": [float, 0],
        "hidden_to_update_w": [float, 0],
        "hidden_to_candidate_w": [float, 0],

        # Biases for the gates and candidate
        "reset_bias": [float, 0],
        "update_bias": [float, 0],
        "candidate_bias": [float, 0],

        # Activation functions for gates and candidate state
        "reset_activation": [str, 'sigmoid'], # Typically sigmoid
        "update_activation": [str, 'sigmoid'], # Typically sigmoid
        "candidate_activation": [str, 'tanh'], # Typically tanh

        # Default float attribute config for these internal weights/biases
        "init_mean": [float, 0.0],
        "init_stdev": [float, 1.0],
        "init_type": [str, 'gaussian'],
        "mutate_rate": [float, 0.05],
        "max_value": [float, 5.0], # Example bounds
        "min_value": [float, -5.0], # Example bounds
    }

    def init_value(self, config):
        return GRUCell(self._config_items,config.activation_defs)
    def mutate_value(self, value, config):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name)

        r = random()
        if r < mutate_rate:
            value.mutate(self._config_items)

        return value

    def validate(self, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        if max_value < min_value:
            raise RuntimeError("Invalid min/max configuration for {self.name}")
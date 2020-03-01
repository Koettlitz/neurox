use std::cell::RefCell;
use std::rc::Rc;
use crate::{sigmoid, sigmoid_prime};

const DEFAULT_BIAS: f64 = 0.0;
const DEFAULT_WEIGHT: f64 = 1.0;

pub type Layer = Vec<Rc<RefCell<Neuron>>>;

pub struct NeuronCore {
    pub bias: f64,
    pub weights: Vec<f64>
}

impl NeuronCore {
    fn process(&self, input: f64, i_previous: usize) -> f64 {
        self.weights[i_previous] * input + self.bias
    }

    fn process_derivative_bias(&self, neuron: &Neuron, input: &Vec<f64>, expected: f64,
                               layer: usize, index: usize) -> f64 {
        self.weights[neuron.index] * neuron.cost_derivative_bias(input, expected, layer, index)
    }

    fn process_derivative_weight(&self, neuron: &Neuron, input: &Vec<f64>, expected: f64,
                                 layer: usize, index_a: usize, index_b: usize) -> f64 {
        self.weights[neuron.index] * neuron.cost_derivative_weight(input, expected, layer,
                                                                   index_a, index_b)
    }
}

/// Represents a neuron of a neural network.
/// Each neuron contains the index of its layer. The first layer does not contain any neurons
/// because there is not much computation to do except sigmoiding the input values.<br>
/// The input layer (which contains no neurons) has the index 0, so the first layer, that
/// actually contains neurons has an index of 1.
pub struct Neuron {
    pub core: NeuronCore,
    layer: usize,
    index: usize,
    is_output: bool,
    previous: Option<Layer>,
    cache: RefCell<Option<f64>>
}

impl Neuron {
    /// Creates a new neuron, that is not in the first hidden layer<br>
    /// <code>layer</code> the layer of the neuron (not 0 or 1)<br>
    /// <code>index</code> the index of the neuron inside the layer<br>
    /// <code>is_output</code> flag if the neuron is in the output layer (last layer)
    /// <code>previous</code> pointer to the previous layer of neurons
    pub fn new(layer: usize, index: usize, is_output: bool, previous: Layer) -> Neuron {
        let weights = vec![DEFAULT_WEIGHT; previous.len()];
        Neuron {
            core: NeuronCore { bias: DEFAULT_BIAS, weights },
            layer,
            index,
            is_output,
            previous: Some(previous),
            cache: RefCell::new(None)
        }
    }

    /// Creates a new neuron in the first hidden layer (layer with index 1)<br>
    /// <code>index</code> the index of the neuron inside the layer
    /// <code>is_output</code> flag if the neuron is in the output layer (last layer)
    /// <code>input_layer_size</code> the size of the input layer (number of input values)
    pub fn new_first_hidden(index: usize, is_output: bool, input_layer_size: usize) -> Neuron {
        let weights = vec![DEFAULT_WEIGHT; input_layer_size];
        Neuron {
            core: NeuronCore { bias: DEFAULT_BIAS, weights },
            layer: 1,
            index,
            is_output,
            previous: None,
            cache: RefCell::new(None)
        }
    }

    fn calc_output_for(&self, input: &Vec<f64>) -> f64 {
        let core = &self.core;
        let result = if let Some(prev) = &self.previous {
            prev.iter()
                .map(|n| {
                    core.process(sigmoid(n.borrow().output_for(input)), n.borrow().index)
                })
                .sum()
        } else {
            input.iter().enumerate()
                        .map(|(i, value)| core.process(sigmoid(*value), i))
                        .sum()
        };

        result
    }

    fn part_derivative(&self, input: &Vec<f64>, expected: f64) -> f64 {
        let output = self.output_for(input);
        if self.is_output {
            2.0 * (expected - sigmoid(output)) * -sigmoid_prime(output)
        } else {
            sigmoid_prime(output)
        }
    }

    /// Calculates the output for the given input without the sigmoid activation function.
    /// The result is cached and used at the next calls of this function
    /// To delete the cache call reset()
    pub fn output_for(&self, input: &Vec<f64>) -> f64 {
        *self.cache.borrow_mut().get_or_insert(self.calc_output_for(input))
    }

    /// Get the derivative of the cost function for a bias of the neuron
    /// in <code>layer</code> at <code>index</code>
    /// <br>
    /// <b>expected</b> is the expected outcome<br>
    /// <b>layer</b> and <b>index</b> are the coordinates of the neuron,
    /// whose bias should be derived with respect to
    pub fn cost_derivative_bias(&self, input: &Vec<f64>, expected: f64,
                                layer: usize, index: usize) -> f64 {

        // First part of derivative
        let part = self.part_derivative(input, expected);

        if layer == self.layer { part }
        else {
            if let Some(prev) = &self.previous {
                // If derivative with respect to directly previous layer neurons bias: Process just that neuron
                if layer == self.layer - 1 {
                        let neuron = &prev[index];
                        part * self.core.process_derivative_bias(&neuron.borrow(), input,
                                                                 expected, layer, index)

                // If derivative with respect to further previous layer: Process all prev neurons
                } else if layer < self.layer {
                    let core = &self.core;
                    part * prev.iter()
                               .map(|n| core.process_derivative_bias(&n.borrow(), input,
                                                                     expected, layer, index))
                               .sum::<f64>()
                } else { panic!("Cannot calculate derivative with respect to bias of neuron of higher layer. \
                                self.layer: {}, layer of request: {}", self.layer, layer) }
            } else {
                panic_on_derivative_call_without_previous_neurons("bias", layer)
            }
        }
    }

    /// Get the derivative of the cost function for a weight between the neuron
    /// in <b>layer</b> at <b>index_a</b> and the neuron in <code><b>layer</b> + 1</code>
    /// at <code>index_b</code>
    pub fn cost_derivative_weight(&self, input: &Vec<f64>, expected: f64,
                              layer: usize, index_a: usize, index_b: usize) -> f64 {
        let part = self.part_derivative(input, expected);
        if let Some(prev) = &self.previous {

            // Weight of connection from previous to this neuron
            if layer == self.layer - 1 {
                let neuron = &prev[index_a];
                part * sigmoid(neuron.borrow().output_for(input))

                // Weight of connection to previous neuron (from even previouser neuron)
            } else if layer == self.layer - 2 {
                let neuron = &prev[index_b];
                part * neuron.borrow().cost_derivative_weight(input, expected, layer,
                                                     index_a, index_b)
            } else if layer < self.layer {
                let core = &self.core;
                part * prev.iter()
                           .map(|n| core.process_derivative_weight(&n.borrow(), input, expected,
                                                                   layer, index_a, index_b))
                           .sum::<f64>()
            } else {
                panic!("Could not calculate derivative with respect to weight of connection \
                       from neuron in layer {} to neuron of the next layer.", layer)
            }
        } else {
            panic_on_derivative_call_without_previous_neurons("weight", layer)
        }
    }

    /// Resets the cached output value
    pub fn reset(&self) {
        *self.cache.borrow_mut() = None;
    }
}

fn panic_on_derivative_call_without_previous_neurons(respect: &str, layer: usize) -> ! {
    panic!("Wrong setup of neurons: Requested to calculate derivative with respect to {} of the \
           connection to previous layer {}, but there are no previous hidden neurons to this one.",
           respect, layer)
}

#[cfg(test)]
mod test {
    use crate::test::*;
    use super::*;
    use std::rc::Rc;
    use std::cell::RefCell;

    const TINY_DELTA: f64 = 0.0001;
    const INPUT: [f64; 8] = [13.1, -17000.0, 0.5, 64.0, -3.0, -512.0, 8000.0, 1.0];
    const TARGET_VALUE: f64 = 0.5;

    fn setup() -> Neuron {
        let hidden_layer = (0..8).map(|i| {
            Rc::new(RefCell::new(Neuron::new_first_hidden(i, false, INPUT.len())))
        }).collect();

        Neuron::new(2, 0, true, hidden_layer)
    }

    #[test]
    fn test_cost_derivative_bias() {
        let neuron = setup();

        let previous = neuron.previous.as_ref().expect("Wrong test setup: Output neuron had no predecessors.");
        for (i, n) in previous.iter().enumerate() {
            let expected = expected_cost_derivative(n, &neuron);
            let actual = neuron.cost_derivative_bias(&INPUT.to_vec(), TARGET_VALUE, 1, i);
            println!("Expected: `{}` - Actual: `{}`", expected, actual);
            assert_approx_eq(expected, actual, "Cost derivative unequal");
        }
    }

    fn cost_fn(neuron: &Neuron) -> f64 {
        let output = sigmoid(neuron.output_for(&INPUT.to_vec()));
        (TARGET_VALUE - output).powi(2)
    }

    fn expected_cost_derivative(target_neuron: &Rc<RefCell<Neuron>>, output_neuron: &Neuron) -> f64 {
        target_neuron.borrow_mut().core.bias -= TINY_DELTA;
        let before = cost_fn(output_neuron);
        reset_recursively(output_neuron);

        target_neuron.borrow_mut().core.bias += 2.0 * TINY_DELTA;
        let after = cost_fn(output_neuron);

        reset_recursively(output_neuron);
        target_neuron.borrow_mut().core.bias -= TINY_DELTA;

        println!("after: {} - before: {}", after, before);
        (after - before) / (2.0 * TINY_DELTA)
    }

    fn reset_recursively(neuron: &Neuron) {
        neuron.reset();
        if let Some(prev) = &neuron.previous {
            for (i, n) in prev.iter().enumerate() {
                if i == prev.len() - 1 { reset_recursively(&n.borrow()); }
                else { n.borrow().reset(); }
            }
        }
    }
}

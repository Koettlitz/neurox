use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashMap;
use super::{sigmoid, sigmoid_prime};

const DEFAULT_BIAS: f64 = 1.0;
const DEFAULT_WEIGHT: f64 = 1.0;

pub type Coords = (usize, usize);
pub type Layer = Rc<RefCell<Vec<Neuron>>>;

struct NeuronCore {
    bias: f64,
    weights: HashMap<Coords, f64>
}

impl NeuronCore {
    fn process(&self, input: f64, coords: Coords) -> f64 {
        self.weights[&coords] * input + self.bias
    }

    fn process_derivative_bias(&self, neuron: &mut Neuron, input: &Vec<f64>, expected: f64,
                               layer: usize, index: usize) -> f64 {
        self.weights[&neuron.coords()] * neuron.cost_derivative_bias(input, expected, layer, index)
    }

    fn process_derivative_weight(&self, neuron: &mut Neuron, input: &Vec<f64>, expected: f64,
                                 layer: usize, index_a: usize, index_b: usize) -> f64 {
        self.weights[&neuron.coords()] * neuron.cost_derivative_weight(input, expected, layer,
                                                                       index_a, index_b)
    }
}

pub struct Neuron {
    core: NeuronCore,
    layer: usize,
    index: usize,
    is_output: bool,
    previous: Option<Layer>,
    cache: Option<f64>
}

impl Neuron {
    /// TODO Docs
    pub fn new(layer: usize, index: usize, is_output: bool, previous: Layer) -> Neuron {
        let weights = previous.borrow().iter()
                              .map(|n| (n.coords(), DEFAULT_WEIGHT))
                              .collect();
        Neuron {
            core: NeuronCore { bias: DEFAULT_BIAS, weights },
            layer,
            index,
            is_output,
            previous: Some(previous),
            cache: None
        }
    }

    /// TODO Docs
    pub fn new_second_layer(index: usize, is_output: bool, input_layer_size: usize) -> Neuron {
        let weights = (0..input_layer_size).map(|i| ((0, i), DEFAULT_WEIGHT)).collect();
        Neuron {
            core: NeuronCore { bias: DEFAULT_BIAS, weights },
            layer: 1,
            index,
            is_output,
            previous: None,
            cache: None
        }
    }

    fn coords(&self) -> Coords { (self.layer, self.index) }

    fn calc_output_for(&mut self, input: &Vec<f64>) -> f64 {
        let core = &self.core;
        let result = if let Some(prev) = &self.previous {
            prev.borrow_mut().iter_mut()
                .map(|n| core.process(n.output_for(input), n.coords()))
                .sum()
        } else {
            input.iter().enumerate()
                 .map(|(i, value)| sigmoid(core.process(*value, (0, i))))
                 .sum()
        };

        self.cache = Some(result);
        result
    }

    fn part_derivative(&mut self, input: &Vec<f64>, expected: f64) -> f64 {
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
    fn output_for(&mut self, input: &Vec<f64>) -> f64 {
        if let Some(v) = self.cache { v }
        else { self.calc_output_for(input) }
    }

    /// Get the derivative of the cost function for a bias of the neuron
    /// in <code>layer</code> at <code>index</code>
    /// <br>
    /// <b>expected</b> is the expected outcome<br>
    /// <b>layer</b> and <b>index</b> are the coordinates of the neuron,
    /// whose bias should be derived with respect to
    fn cost_derivative_bias(&mut self, input: &Vec<f64>, expected: f64,
                            layer: usize, index: usize) -> f64 {

        // First part of derivative
        let part = self.part_derivative(input, expected);

        if layer == self.layer { part }
        else {
            if let Some(prev) = &self.previous {
                // If derivative with respect to directly previous layer neurons bias: Process just that neuron
                if layer == self.layer - 1 {
                        let neuron = &mut prev.borrow_mut()[index];
                        part * self.core.process_derivative_bias(neuron, input,
                                                                 expected, layer, index)

                // If derivative with respect to further previous layer: Process all prev neurons
                } else if layer < self.layer {
                    let core = &self.core;
                    part * prev.borrow_mut().iter_mut()
                               .map(|n| core.process_derivative_bias(n, input,
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
    /// in <b>layer</b> at <b>index_a</b> and the neuron in <code><b>layer</b> - 1</code>
    /// at <code>index_b</code>
    fn cost_derivative_weight(&mut self, input: &Vec<f64>, expected: f64,
                              layer: usize, index_a: usize, index_b: usize) -> f64 {
        let part = self.part_derivative(input, expected);
        if let Some(prev) = &mut self.previous {

            // Weight of connection from previous to this neuron
            if layer == self.layer - 1 {
                let neuron = &mut prev.borrow_mut()[index_a];
                part * neuron.output_for(input)

                // Weight of connection to previous neuron (from even previouser neuron)
            } else if layer == self.layer - 2 {
                let neuron = &mut prev.borrow_mut()[index_b];
                part * neuron.cost_derivative_weight(input, expected, layer,
                                                     index_a, index_b)
            } else if layer < self.layer {
                let core = &self.core;
                part * prev.borrow_mut().iter_mut()
                    .map(|n| core.process_derivative_weight(n, input, expected,
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
    fn reset(&mut self) {
        self.cache = None;
    }
}

fn panic_on_derivative_call_without_previous_neurons(respect: &str, layer: usize) -> ! {
    panic!("Wrong setup of neurons: Requested to calculate derivative with respect to {} of the \
           connection to previous layer {}, but there are no previous hidden neurons to this one.",
           respect, layer)
}

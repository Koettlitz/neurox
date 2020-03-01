mod neurons;

use std::cell::RefCell;
use std::rc::Rc;
use neurons::*;
use crate::neural_network::LearningRate::*;

const DEFAULT_LEARNING_RATE: f64 = 0.001;

/// Neural network, that contains the Neurons with their weights and biases
/// Use [`process`] to get the output of the neural network for a given input.
/// Use [`learn`] to adjust the weights and biases of the neurons to make the network learn.
///
/// [`process`]: ../../neural_network/struct.NeuralNetwork.html#process
/// [`learn`]: ../../neural_network/struct.NeuralNetwork.html#learn
pub struct NeuralNetwork<C: Fn(f64) -> f64> {
    layers: Vec<Layer>,
    pub learning_rate: LearningRate<C>
}

impl<C: Fn(f64) -> f64> NeuralNetwork<C> {
    pub fn new(layer_sizes: Vec<usize>) -> NeuralNetwork<C> {
        NeuralNetwork::with_learning_rate(layer_sizes, Fixed(DEFAULT_LEARNING_RATE))
    }

    pub fn with_learning_rate(layer_sizes: Vec<usize>, learning_rate: LearningRate<C>) -> NeuralNetwork<C> {
        let mut layers = Vec::new();
        let mut previous: Option<Layer> = None;
        for (i_layer, size) in layer_sizes.iter().skip(1).enumerate() {
            let mut layer: Layer = Vec::new();
            for i_neuron in 0..*size {
                let neuron = if let Some(p) = &previous {
                    Neuron::new(i_layer, i_neuron, i_layer == layer_sizes.len(), p.clone())
                } else {
                    Neuron::new_first_hidden(i_neuron, i_layer == layer_sizes.len(), layer_sizes[0])
                };

                layer.push(Rc::new(RefCell::new(neuron)));
            }

            layers.push(layer.clone());
            previous = Some(layer);
        }

        NeuralNetwork { layers, learning_rate }
    }

    /// Let the network process the given input to get the output values
    pub fn process(&self, input: &Vec<f64>) -> Vec<f64> {
        self.layers.last().expect("Neural Network was empty.").iter()
            .map(|n| n.borrow_mut().output_for(input)).collect()
    }

    /// Adjust the neurons weights and biases to make the network learn.
    pub fn learn(&self, input: &Vec<f64>, expected: &Vec<f64>) {
        // adjust biases
        for (i_layer, layer) in self.layers.iter().enumerate() {
            let layer_size = layer.len();
            for i_neuron in 0..layer_size {
                let cost = self.cost_derivative_bias(input, expected, i_layer, i_neuron);
                (&layer[i_layer]).borrow_mut().core.bias -= cost * self.learning_rate.for_cost(cost);
            }
        }

        // adjust weights
        for (i_layer, layer) in self.layers.iter().skip(1).enumerate() {
            let prev = &self.layers[i_layer - 1];

            for i_neuron_prev in 0..prev.len() {
                for (i_neuron, n) in layer.iter().enumerate() {
                    let cost = self.cost_derivative_weight(input, expected, i_layer - 1, i_neuron_prev, i_neuron);
                    n.borrow_mut().core.weights[i_neuron_prev] -= cost * self.learning_rate.for_cost(cost);
                }
            }
        }

        // Clear cached values
        self.layers.iter().flat_map(|l| l.iter())
            .for_each(|n| n.borrow_mut().reset());
    }

    fn cost_derivative_bias(&self, input: &Vec<f64>, expected: &Vec<f64>, i_layer: usize, i_neuron: usize) -> f64 {
        self.layers.last().expect("Neural Network was empty.").iter().enumerate()
            .map(|(i, n)| n.borrow_mut().cost_derivative_bias(input, expected[i], i_layer, i_neuron))
            .sum()
    }

    fn cost_derivative_weight(&self, input: &Vec<f64>, expected: &Vec<f64>, layer: usize, index_a: usize, index_b: usize) -> f64 {
        self.layers.last().expect("Neural Network was empty.").iter().enumerate()
            .map(|(i, n)| n.borrow_mut().cost_derivative_weight(input, expected[i], layer, index_a, index_b))
            .sum()
    }
}

pub enum LearningRate<C: Fn(f64) -> f64> {
    Fixed(f64),
    ByCost(C)
}

impl<C: Fn(f64) -> f64> LearningRate<C> {
    pub fn for_cost(&self, cost: f64) -> f64 {
        match self {
            Fixed(lr) => *lr,
            ByCost(c) => c(cost)
        }
    }
}

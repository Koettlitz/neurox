use std::cell::RefCell;
use std::rc::Rc;
use crate::neurons::*;

pub struct NeuralNetwork {
    layers: Vec<Layer>
}

impl NeuralNetwork {
    pub fn new(layer_sizes: Vec<usize>) -> NeuralNetwork {
        let mut layers = Vec::new();
        let mut previous: Option<Layer> = None;
        for (i_layer, size) in layer_sizes.iter().enumerate() {
            let layer: Layer = Rc::new(RefCell::new(Vec::new()));
            for i_neuron in 0..*size {
                let prev = if let Some(p) = &previous {
                    Some(Rc::clone(&p))
                } else { None };

                let neuron = Neuron::new(i_layer, i_neuron,
                                        i_layer == layer_sizes.len(), prev);

                layer.borrow_mut().push(neuron);
            }

            layers.push(Rc::clone(&layer));
            previous = Some(layer);
        }

        NeuralNetwork { layers }
    }
}
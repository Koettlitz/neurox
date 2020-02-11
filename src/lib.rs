mod neurons;

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

fn sigmoid_prime(x: f64) -> f64 {
    let minus_exp = (-x).exp();
    minus_exp / (1.0 + minus_exp).powi(2)
}
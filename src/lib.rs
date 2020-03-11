pub mod neural_network;

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

fn sigmoid_prime(x: f64) -> f64 {
    let minus_exp = (-x).exp();
    minus_exp / (1.0 + minus_exp).powi(2)
}

#[cfg(test)]
mod test {
    use super::{sigmoid, sigmoid_prime};

    pub const FLOATING_PRECISION: u8 = 4;

    #[test]
    fn sigmoid_for_minus_inclines_zero() {
        assert!(sigmoid(-1.0) < 0.5);
        inclines_zero(&sigmoid, false);
    }

    #[test]
    fn sigmoid_zero_is_half() {
        assert_approx_eq(0.5, sigmoid(0.0), "Sigmoid of 0 should be 0.5");
    }

    #[test]
    fn sigmoid_for_plus_inclines_one() {
        let mut sig = sigmoid(1.0);
        assert!(sig > 0.5);

        for i in 2..10 {
            let greater = sigmoid(i as f64 * 2.0);
            assert!(greater > sig);
            assert!(greater < 1.0);
            sig = greater;
        }
    }

    #[test]
    fn sig_prime_for_minus_inclines_zero() {
        assert!(sigmoid_prime(-1.0) < 0.25);
        inclines_zero(&sigmoid_prime, false);
    }

    #[test]
    fn sig_prime_zero_is_quarter() {
        assert_approx_eq(0.25, sigmoid_prime(0.0), "Sigmoid prime of zero must be 0.25");
    }

    #[test]
    fn sig_prime_for_plus_inclines_zero() {
        assert!(sigmoid_prime(1.0) < 0.25);
        inclines_zero(&sigmoid_prime, true);
    }

    fn inclines_zero(fun: & dyn Fn(f64) -> f64, from_below: bool) {
        let mut last = fun(-1.0);

        for i in 2..10 {
            let input = if from_below { i as f64 } else { -i as f64 };
            let smaller = fun(input * 2.0);
            assert!(smaller < last);
            assert!(smaller > 0.0);
            last = smaller;
        }
    }

    pub fn approx_eq(a: f64, b: f64, precision: u8) -> bool {
        let max_diff = 10.0_f64.powi(-(precision as i32));
        (a - b).abs() < max_diff
    }

    pub fn assert_approx_eq(a: f64, b: f64, msg: &str) {
        if !approx_eq(a, b, FLOATING_PRECISION) {
            panic!("{} - expected: `{}` - actual: `{}`", msg, a, b);
        }
    }
}
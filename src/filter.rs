use std::f64::consts::PI;

struct Formant {
    a1: f64,
    a2: f64,
    b0: f64,
}

impl Formant {
    fn new(frequency: f64, bandwidth: f64, sample_rate: u32) -> Self {
        let sample_rate = sample_rate as f64;
        let r = (-PI * bandwidth / sample_rate).exp();
        let wc = 2.0 * PI * frequency / sample_rate;

        let b0 = (1.0 - r) * (1.0 - 2.0 * r * (2.0 * wc).cos() + (r * r));
        let a1 = -2.0 * r * wc.cos();
        let a2 = r * r;

        Self { b0, a1, a2 }
    }
}

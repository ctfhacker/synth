pub mod audio;
pub mod oscillator;

use std::time::Duration;

/// Number of harmonics that a voice will produce
const NUM_HARMONICS: usize = 20;

/// The number of samples per second
const SAMPLE_RATE: u32 = 44100;

/// A thing that can be sampled
pub trait Sampler: Send {
    // Get the next sample
    fn sample(&mut self) -> f64;

    fn save(&mut self, filename: &str, time: Duration) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(filename, spec).unwrap();

        let num_samples = (SAMPLE_RATE as f64 * time.as_secs_f64()) as usize;
        for _ in 0..num_samples {
            writer.write_sample(self.sample() as f32);
        }

        writer.finalize().unwrap();
    }
}

use crate::oscillator::{sine_oscillator, Oscillator};

#[derive(Clone)]
struct Voice {
    harmonics: [Oscillator; NUM_HARMONICS],

    /// The current tilt ratio used to determine the amplitudes of harmonics
    /// Usually in the [-0.1, -16.0] range
    tilt: f64,

    /// The ratio of amplitude decrease for each subsequent harmonic
    tilt_ratio: f64,
}

impl Voice {
    pub fn new() -> Self {
        let harmonics = [sine_oscillator(SAMPLE_RATE); NUM_HARMONICS];
        let tilt = -3.0;
        let tilt_ratio = db_to_amplitude(tilt);

        Self {
            harmonics,
            tilt,
            tilt_ratio,
        }
    }

    /// Set the fundamental frequency of this voice
    ///
    /// This will subsequentially set the frequencies of all the harmonics
    /// for the voice as well
    pub fn set_frequency(&mut self, freq: f64) {
        for (i, harmonic) in self.harmonics.iter_mut().enumerate() {
            harmonic.set_frequency(freq * (i + 1) as f64);
        }
    }

    pub fn set_tilt(&mut self, tilt: f64) {
        self.tilt_ratio = db_to_amplitude(tilt)
    }
}

impl Sampler for Voice {
    fn sample(&mut self) -> f64 {
        // Sum the harmonics for this voice applying the tilt roll off ratio
        self.harmonics
            .iter_mut()
            .enumerate()
            .map(|(index, harmonic)| harmonic.sample() * self.tilt_ratio.powi(index as i32))
            .sum()
    }
}

#[derive(Clone)]
struct Quartet {
    bass: Voice,
    bari: Voice,
    lead: Voice,
    tenor: Voice,
}

impl Quartet {
    pub fn new() -> Self {
        let mut bass = Voice::new();
        let mut bari = Voice::new();
        let mut lead = Voice::new();
        let mut tenor = Voice::new();
        bass.set_frequency(220.0);
        bari.set_frequency(220.0 * 1.5);
        lead.set_frequency(220.0 * 2.);
        tenor.set_frequency(220.0 * 5. / 4.);

        Self {
            bass,
            bari,
            lead,
            tenor,
        }
    }
}

impl Sampler for Quartet {
    fn sample(&mut self) -> f64 {
        0.2 * (self.bass.sample() + self.bari.sample() + self.lead.sample() + self.tenor.sample())
    }
}

pub fn db_to_amplitude(db: f64) -> f64 {
    10.0_f64.powf(db / 20.)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut osc = Quartet::new();
        audio::play(osc.clone(), Duration::from_secs(1));

        osc.save("playme.wav", Duration::from_secs(1));
    }
}

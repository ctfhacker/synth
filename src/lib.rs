use rand::distributions::{Distribution, Uniform};

use rustfft::num_complex::Complex32;

pub mod audio;
pub mod oscillator;

use biquad::*;

// pub mod biquad;
// use biquad::Biquad;

pub mod filter;

use std::time::Duration;

/// Number of harmonics that a voice will produce
const NUM_HARMONICS: usize = 8;

/// The number of samples per second
pub const SAMPLE_RATE: u32 = 48000;

timeloop::create_profiler!(Timers);

timeloop::impl_enum!(
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum Timers {
        CloneVoice,
        GenSamples,
        SingleSample,
        Mutate,
        Fft,
        Diff,
    }
);

/// Save a wav file of name `filename` from `input` samples
pub fn save_wav(filename: &str, input: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(filename, spec).unwrap();

    for sample in input {
        writer.write_sample(*sample);
    }

    writer.finalize().unwrap();
}

/// Read and return a .wav file named `filename`
pub fn read_wav(filename: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(filename).unwrap();
    reader.samples::<i16>().map(|x| x.unwrap() as f32).collect()
}

/// A thing that can be sampled
pub trait Sampler: Send {
    // Get the next sample
    fn sample(&mut self) -> f32;

    fn gen_and_save(&mut self, filename: &str, time: Duration) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(filename, spec).unwrap();

        let num_samples = (SAMPLE_RATE as f32 * time.as_secs_f32()) as usize;
        for _ in 0..num_samples {
            writer.write_sample(self.sample());
        }

        writer.finalize().unwrap();
    }

    fn gen_samples_and_save(&mut self, filename: &str, num_samples: usize) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(filename, spec).unwrap();

        for _ in 0..num_samples {
            writer.write_sample(self.sample());
        }

        writer.finalize().unwrap();
    }

    fn gen(&mut self, time: Duration) -> Vec<f32> {
        let mut result = Vec::new();

        let num_samples = (SAMPLE_RATE as f32 * time.as_secs_f32()) as usize;
        for _ in 0..num_samples {
            result.push(self.sample());
        }

        result
    }

    fn gen_samples(&mut self, num_samples: usize, output: &mut Vec<f32>) {
        output.clear();

        for _ in 0..num_samples {
            output.push(self.sample());
        }
    }

    fn gen_samples_complex(&mut self, num_samples: usize, output: &mut Vec<Complex32>) {
        output.clear();

        for _ in 0..num_samples {
            let curr_sample = timeloop::time_work!(Timers::SingleSample, { self.sample() });

            output.push(Complex32::new(curr_sample, 0.0));
        }
    }
}

use crate::oscillator::{sine_oscillator, Oscillator};

#[derive(Clone)]
pub struct Voice {
    harmonics: [Oscillator; NUM_HARMONICS],

    amplitude_table: [f32; NUM_HARMONICS],

    /// Formants to be applied for this voice
    formants: [Option<DirectForm2Transposed<f32>>; 16],

    /// The formant frequency and bandwidth
    formant_freq_bw: [Option<(f32, f32)>; 16],

    /// The fundamental frequency of this voice
    fundamental: f32,

    /// Number of available formants
    num_formants: usize,

    /// The spectral tilt of the voice
    tilt: f32,
}

impl std::fmt::Debug for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fundamental: {:8.2}\n", self.fundamental);
        write!(f, "Tilt: {:6.2}\n", self.tilt);
        for (i, formant) in self.formant_freq_bw.iter().enumerate() {
            let Some((freq, bandwidth)) = formant else {
                break;
            };

            write!(f, "F{}: {freq:8.2} Bandwidth: {bandwidth:8.2}\n", i + 1);
        }

        Ok(())
    }
}

fn rdtsc() -> u64 {
    unsafe { std::arch::x86_64::_rdtsc() }
}

impl Voice {
    pub fn new() -> Self {
        let harmonics = [sine_oscillator(SAMPLE_RATE); NUM_HARMONICS];
        let amplitude_table = [0f32; NUM_HARMONICS];

        let mut res = Self {
            harmonics,
            amplitude_table,
            formants: [None; 16],
            formant_freq_bw: [None; 16],
            fundamental: 0.0,
            num_formants: 0,
            tilt: 0.0,
        };

        res.set_tilt(-3.0);
        res
    }

    pub fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let tilt_range = Uniform::from(-1.0..=1.0f32);
        let freq_range = Uniform::from(-100.0..=100.0f32);
        let fund_range = Uniform::from(0.0..=2.0f32);

        // Reset the harmonic oscillator index
        self.harmonics
            .iter_mut()
            .for_each(|harmonic| harmonic.reset());

        // Adjust the fundamental a smidge
        /*
        if rdtsc() % 8 == 0 {
            let new_fundamental = (self.fundamental * fund_range.sample(&mut rng));
            self.set_frequency(new_fundamental.max(0.0).min(1000.0));
        }
        */

        // Adjust the tilt a smidge
        if rdtsc() % 8 == 0 {
            let mut tilt = (self.tilt + tilt_range.sample(&mut rng)).min(0.0);
            self.set_tilt(tilt);
        }

        // Randomly mutate the formants
        for index in 0..self.formant_freq_bw.len() {
            let Some((mut freq, mut bandwidth)) = self.formant_freq_bw[index] else {
                continue;
            };

            // Randomly choose to mutate this formant
            if rdtsc() & 7 == 0 {
                continue;
            }

            // Sometimes randomly change the formant frequency
            if rdtsc() & 7 == 0 {
                freq += freq_range.sample(&mut rng);
            }

            // Sometimes randomly change the formant bandwidth
            if rdtsc() & 7 == 0 {
                bandwidth += freq_range.sample(&mut rng);
            }

            // Set formant ensuring a positive formant frequency and
            // positive bandwidth
            self.set_formant(index, freq.max(0.0), bandwidth.max(0.0));
        }
    }

    /// Set the fundamental frequency of this voice
    ///
    /// This will subsequentially set the frequencies of all the harmonics
    /// for the voice as well
    pub fn set_frequency(&mut self, freq: f32) {
        for (i, harmonic) in self.harmonics.iter_mut().enumerate() {
            harmonic.set_frequency(freq * (i + 1) as f32);
        }

        self.fundamental = freq;
    }

    /// Set the current tilt for this voice
    pub fn set_tilt(&mut self, tilt: f32) {
        self.tilt = tilt;

        for harmonic in 0..NUM_HARMONICS {
            self.amplitude_table[harmonic] = amplitude_ratio(harmonic as u32 + 1, tilt);
        }
    }

    /// Set a specific formant. F1 is index 0, F2 is index 1, ect.
    pub fn set_formant(&mut self, index: usize, formant: f32, bandwidth: f32) {
        let q = formant / bandwidth;

        let coeffs = Coefficients::<f32>::from_params(
            Type::LowPass,
            (SAMPLE_RATE as f32).hz(),
            (formant as f32).hz(),
            q as f32,
        )
        .unwrap();

        let mut filter = DirectForm2Transposed::<f32>::new(coeffs);
        self.formants[index] = Some(filter);
        self.formant_freq_bw[index] = Some((formant, bandwidth));
    }

    pub fn set_formants(&mut self, formants: &[(f32, f32)]) {
        assert!(
            formants.len() < self.formants.len(),
            "Not enough formants available in voice"
        );

        self.formants = [None; 16];
        self.formant_freq_bw = [None; 16];
        self.num_formants = formants.len();
        for (i, (formant, bandwidth)) in formants.iter().enumerate() {
            let q = formant / bandwidth;

            let coeffs = Coefficients::<f32>::from_params(
                Type::LowPass,
                (SAMPLE_RATE as f32).hz(),
                (*formant as f32).hz(),
                q as f32,
            )
            .unwrap();

            let mut filter = DirectForm2Transposed::<f32>::new(coeffs);
            self.formants[i] = Some(filter);
            self.formant_freq_bw[i] = Some((*formant, *bandwidth));
        }
    }
}

impl Sampler for Voice {
    fn sample(&mut self) -> f32 {
        // Initialize the sample
        let volume_db = -30.0;
        let volume = db_to_amplitude(volume_db);

        // Sum the harmonics for this voice applying the tilt roll off ratio
        let sample: f32 = self
            .harmonics
            .iter_mut()
            .enumerate()
            .map(|(index, harmonic)| harmonic.sample() * self.amplitude_table[index])
            .sum();

        // Apply the formants to this sample
        let mut result = sample as f32;

        for filter in self.formants.iter_mut() {
            let Some(filter) = filter else {
                break;
            };

            result = filter.run(result);
        }

        // Return the result
        result * volume
    }
}

#[derive(Default)]
struct WhiteNoise;

impl Sampler for WhiteNoise {
    // Get the next sample
    fn sample(&mut self) -> f32 {
        let between = Uniform::from(-1.0f32..=1.0f32);
        let mut rng = rand::thread_rng();

        between.sample(&mut rng)
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
    fn sample(&mut self) -> f32 {
        (self.bass.sample() + self.bari.sample() + self.lead.sample() + self.tenor.sample())
    }
}

pub fn db_to_amplitude(db: f32) -> f32 {
    10.0_f32.powf(db / 20.)
}

pub fn amplitude_to_db(amplitude: f32) -> f32 {
    20.0 * amplitude.log10()
}

/// Given a harmonic and tilt, return the amplitude ratio for that harmonic
fn amplitude_ratio(harmonic: u32, tilt: f32) -> f32 {
    // Compute the dB reduction for the nth harmonic
    let reduction_db = tilt * (harmonic as f32).log2();

    // Convert the dB reduction to a linear amplitude ratio
    let amplitude_ratio = 10f32.powf(reduction_db / 20.0);

    /*
    println!(
        "Harmonic {}: reduction = {:.2} dB, amplitude ratio = {:.5}",
        n, reduction_db, amplitude_ratio
    );
    */

    amplitude_ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        // let mut osc = Quartet::new();
        // audio::play(osc.clone(), Duration::from_secs(1));
        // osc.save("playme.wav", Duration::from_secs(1));

        // let mut noise = WhiteNoise::default();
        // let input = noise.gen(Duration::from_secs(1));
        // println!("Noise: {}", input.len());
        // save_wav("whitenoise.wav", &input);

        let mut voice = Voice::new();
        voice.set_tilt(-3.0);
        voice.set_frequency(261.4);
        voice.set_formants(&[(800.0, 50.0)]);
        let mut input = voice.gen(Duration::from_millis(5));

        audio::play(voice.clone(), Duration::from_secs(1));
        // audio::play_input(input, SAMPLE_RATE as f32);

        /*
        let input = voice.gen(Duration::from_secs(1));
        let formants = [(500.0, 50.0), (2000.0, 50.0)];

        let mut filters = [None; 16];
        for (i, (formant, bandwidth)) in formants.iter().enumerate() {
            let mut filter = Biquad::new(*formant, *bandwidth, SAMPLE_RATE);
            filters[i] = Some(filter);
        }

        let mut output = Vec::new();

        for sample in input.iter() {
            let mut result = 0.0;

            for filter in filters.iter_mut() {
                let Some(filter) = filter else {
                    break;
                };
                result += filter.process(*sample as f32);
            }

            output.push(result as f32);
        }

        println!("{output:?}");
        audio::play_input(output, Duration::from_secs(1));
        */

        // voice.save("test.wav", Duration::from_secs(1));
    }
}

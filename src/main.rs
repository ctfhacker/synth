use rustfft::num_complex::Complex32;
use std::time::Duration;
use synth::*;

use plotters::prelude::*;

/// Calculate the root mean square of two Complex numbers using only their real values
fn root_mean_square(x: &[Complex32], y: &[Complex32]) -> f32 {
    assert!(x.len() == y.len());

    let mut sum = 0.0;
    for (x_val, y_val) in x.iter().zip(y.iter()) {
        let diff = (x_val.re - y_val.re).powi(2);
        sum += diff;
    }

    (sum / x.len() as f32).sqrt()
}

fn difference(x: &[Complex32], y: &[Complex32]) -> f32 {
    assert!(x.len() == y.len());

    let mut sum = 0.0;
    for (x_val, y_val) in x.iter().zip(y.iter()) {
        let diff = (x_val.re - y_val.re).abs();
        sum += diff;
    }

    sum
}

fn loss_func(x: &[Complex32], y: &[Complex32]) -> f32 {
    root_mean_square(x, y)
    // difference(x, y)
}

fn plot_fft(filename: &str, data: &[rustfft::num_complex::Complex<f32>]) {
    let sampling_rate = synth::SAMPLE_RATE as f32;

    let filename = format!("{filename}.png");
    let root = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_magnitude_db = data
        .iter()
        .map(|c| 20.0 * c.norm().max(1e-5).log10())
        .fold(f32::MIN, f32::max);

    let mut max_freq = sampling_rate / 2.0; // Nyquist frequency
    max_freq = 3000.0;
    let mut chart = ChartBuilder::on(&root)
        .caption("FFT Magnitude Plot", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_ranged(0.0..max_freq, -100.0..max_magnitude_db)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Frequency (Hz)")
        .y_desc("Magnitude (dB)")
        .draw()
        .unwrap();

    let half_data_len = data.len() / 2; // We'll plot only up to the Nyquist frequency
    let freq_resolution = sampling_rate / data.len() as f32;

    // Convert the amplitude data to dB
    let series_data: Vec<_> = (0..half_data_len)
        .map(|i| {
            let freq = i as f32 * freq_resolution;
            (freq, 20.0 * data[i].norm().max(1e-5).log10())
        })
        .collect();

    chart
        .draw_series(LineSeries::new(series_data, &BLUE))
        .unwrap();
}

fn generate_fake_test_voice(num_samples: usize) -> Vec<Complex32> {
    // Init the voice used to test with
    let mut voice = Voice::new();
    voice.set_tilt(-1.0);
    voice.set_frequency(261.6);
    voice.set_formants(&[(300.0, 53.3), (2100.0, 53.3)]);

    let mut result = Vec::new();
    voice.gen_samples_complex(num_samples, &mut result);

    result
}

timeloop::create_profiler!(Timers);

fn main() {
    // let input = synth::read_wav("./madde_test_f1_800_bw_53.3_note_261.6_tilt_neg1.wav");
    let input = synth::read_wav("./b4_test.wav");
    println!("Input samples: {}", input.len());

    let num_test_samples = input.len().min(1024 * 16);

    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(num_test_samples);

    let mut wanted_voice_buffer = input
        .iter()
        .take(num_test_samples)
        .map(|x| Complex32::new(*x, 0.0))
        .collect::<Vec<_>>();

    // let mut wanted_voice_buffer = generate_fake_test_voice(num_test_samples);

    // Compute FFT in place
    fft.process(&mut wanted_voice_buffer);

    plot_fft("wanted", &wanted_voice_buffer);

    // Init the voice used to test with
    let mut voice = Voice::new();
    voice.set_tilt(-3.0);
    voice.set_frequency(504.0);
    voice.set_formants(&[
        (500.0, 53.3),
        (1500.0, 50.0),
        (3000.0, 100.0),
        (4000.0, 100.0),
        (5000.0, 100.0),
        (6000.0, 100.0),
        (7000.0, 100.0),
    ]);

    // Initialize the best distance and voice variables
    let mut best_distance = f32::MAX;
    let mut best_voice = voice.clone();

    let mut count = 0;
    let mut timer = std::time::Instant::now();
    let mut start_time = std::time::Instant::now();
    let mut input = Vec::new();

    // Start the global timer for the profiler
    timeloop::start_profiler!();

    'done: loop {
        count += 1;
        if timer.elapsed() >= std::time::Duration::from_secs(1) {
            println!(
                "{:8.2} iters/sec",
                count as f64 / start_time.elapsed().as_secs_f64()
            );
            timer = std::time::Instant::now();

            // timeloop::print!();
        }

        let mut voice = timeloop::time_work!(Timers::CloneVoice, { best_voice.clone() });

        // Make some mutation in the voice to see if it's closer to the wanted test case
        timeloop::time_work!(Timers::Mutate, {
            voice.mutate();
        });

        // Generate the sample to test with
        timeloop::time_work!(Timers::GenSamples, {
            voice.gen_samples_complex(num_test_samples, &mut input);
        });

        // Compute FFT in place
        // fft.process_with_scratch(&mut buffer, &mut scratch);
        timeloop::time_work!(Timers::Fft, {
            fft.process(&mut input);
        });

        // Calculate the distance away from the wanted case
        let distance =
            timeloop::time_work!(Timers::Diff, { loss_func(&input, &wanted_voice_buffer) });

        // If a better voice has been found, keep it
        if distance < best_distance {
            println!("New best! {distance}");
            println!("Voice: {voice:?}");
            best_distance = distance;
            best_voice = voice.clone();

            best_voice.gen_and_save("curr_best.wav", Duration::from_secs(1));

            if best_distance < 0.08 {
                break 'done;
            }

            // plot_fft(&format!("mid_point_{distance:8.6}"), &input);
        }
    }
}

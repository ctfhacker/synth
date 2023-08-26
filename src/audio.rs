use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Device;

use crate::Sampler;

use std::time::Duration;

pub fn play(mut sampler: impl Sampler + 'static, play_time: Duration) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device: Device = host
        .default_output_device()
        .expect("No default device found");

    let config = device.default_output_config().unwrap();
    println!("Config {config:?}");

    let err_fn = |err| eprintln!("Error on playback: {err}");

    let channels = config.channels() as usize;

    let stream = device.build_output_stream(
        &config.into(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for frame in data.chunks_mut(channels) {
                let value = sampler.sample();
                for sample in frame.iter_mut() {
                    *sample = value as f32;
                }
            }
        },
        err_fn,
        None,
    )?;

    stream.play()?;

    std::thread::sleep(play_time);

    Ok(())
}

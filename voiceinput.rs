use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Instant, Duration};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{StreamConfig, SampleRate};
use vosk::{Model, Recognizer};

pub fn voice_input() {
    let host = cpal::default_host();
    let input_device = host.default_input_device().expect("No input device available");

    let supported_config = input_device
        .supported_input_configs()
        .expect("Failed to get supported input configs")
        .next()
        .expect("No supported input configs")
        .with_max_sample_rate();

    let config = StreamConfig {
        channels: supported_config.channels(),
        sample_rate: SampleRate(48000),
        buffer_size: cpal::BufferSize::Default,
    };

    let model_path = "C:/Users/User/RustroverProjects/VoiceAssistant/vosk-model-ru-0.22"; 
    let model = Model::new(model_path).expect("Failed to load model");
    let recognizer = Recognizer::new(&model, 16000.0).expect("Failed to create recognizer");
    let recognizer = Arc::new(Mutex::new(recognizer));

    let should_stop = Arc::new(AtomicBool::new(false));
    let should_stop_clone = Arc::clone(&should_stop);

    let last_sound_time = Arc::new(Mutex::new(Instant::now()));
    let last_sound_time_clone = Arc::clone(&last_sound_time);
    let recognizer_clone = Arc::clone(&recognizer);

    let stream = input_device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {

                let mono_data: Vec<f32> = data.chunks(2).map(|chunk| (chunk[0] + chunk[1]) / 2.0).collect();
                let resampled_data: Vec<f32> = mono_data.iter().step_by(3).cloned().collect();
                let pcm_data: Vec<i16> = resampled_data.iter().map(|&sample| (sample * i16::MAX as f32) as i16).collect();
                let mut recognizer = recognizer_clone.lock().unwrap();

                recognizer.accept_waveform(&pcm_data).expect("Failed to process audio");

                let partial_result = recognizer.partial_result();
                println!("Partial result: {}", partial_result.partial);


                let silence_threshold = 0.1;
                let sum: f32 = mono_data.iter().map(|&sample| sample * sample).sum();
                let rms = (sum / mono_data.len() as f32).sqrt();


                if rms >= silence_threshold {
                    let mut last_sound_time = last_sound_time_clone.lock().unwrap();
                    *last_sound_time = Instant::now();
                }


                let elapsed_silence = last_sound_time_clone.lock().unwrap().elapsed();
                if elapsed_silence >= Duration::from_secs(5) {
                    should_stop_clone.store(true, Ordering::SeqCst);
                }
            },
            move |err| {
                eprintln!("An error occurred in the audio stream: {}", err);
            },
            None,
        )
        .expect("Failed to build input stream");

    stream.play().expect("Failed to start stream");
    println!("Listening for audio...");

    while !should_stop.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    stream.pause().expect("Failed to pause stream");

    let mut binding = recognizer.lock().unwrap();
    let final_result = binding.final_result();
    println!("Final result: {:?}", final_result);
}

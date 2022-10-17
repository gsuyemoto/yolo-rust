
use anyhow::{bail, Result};

use linuxvideo::{
    format::{PixFormat, Pixelformat},
    Device,
};

use std::{env, fs::File, io::Write, path::Path};

// const MODEL: &str           = "models/yolov5s.torchscript";
const MODEL: &str           = "models/mobilenet-v3.pt";
const WIDTH: i32            = 640;
const HEIGHT: i32           = 480;
const WINDOW_NAME: &str     = "YOLO Object Detection";

pub fn main() -> Result<()>{
    // load jit model and put it to cuda
    let mut model = tch::CModule::load_on_device(MODEL, tch::Device::Cpu)?;   
    model.set_eval(); 

    let device = Device::open(Path::new("/dev/video0"))
        .expect("Unable to open camera device");

    let mut capture = device.video_capture(PixFormat::new(WIDTH as u32, HEIGHT as u32, Pixelformat::JPEG))
            .expect("Error opening video capture device");

    println!("negotiated format: {:?}", capture.format());

    let mut stream = capture.into_stream(2)
        .expect("Error creating camera stream");
    
    println!("stream started, waiting for data");

    let path = Path::new("test.jpg");
    let mut file = File::create(path)
        .expect("Error creating file to save .jpg capture");

    stream.dequeue(|buf| {
        file.write_all(&*buf)?;
        println!("wrote file");
        Ok(())
    })
    .expect("Error, failed to write cam cap to file");

    loop {
        let tensor = tch::vision::imagenet::load_image(path)
            .expect("Error loading video capture as Tensor");
        let probabilities = model.forward_ts(&[tensor.unsqueeze(0)])?.softmax(-1, tch::Kind::Float);  

        println!("{:?}", probabilities);
    }
  
    Ok(())
} 

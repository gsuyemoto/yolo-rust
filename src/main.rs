
use anyhow::{bail, Result};
use std::{time, process};

use opencv::{
	highgui,
	prelude::*,
	imgproc,
	videoio, 
    core,
};

const MODEL: &str   = "data/yolov7-tiny.pt";
const WIDTH: i32    = 640;
const HEIGHT: i32   = 480;
const WINDOW_NAME: &str    = "YOLO Object Detection";

pub fn main() -> Result<()>{
    // create empty window named 'frame'
    highgui::named_window(WINDOW_NAME, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(WINDOW_NAME, 640, 480)?;

    // load jit model and put it to cuda
    let mut model = tch::CModule::load(MODEL)?;   
    model.set_eval(); 
    model.to(tch::Device::Cpu, tch::Kind::Float, false);

    // create empty Mat to store image data
    let mut frame = Mat::default();

    // Open the web-camera (assuming you have one)
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    loop {
        // read frame to empty mat 
        cam.read(&mut frame)?;
        
        // resize image
        //let mut resized = Mat::default();   
        ////imgproc::resize(&frame, &mut resized, core::Size{width: WIDTH, height: HEIGHT}, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;
        //// convert bgr image to rgb
        //let mut rgb_resized = Mat::default();  
        ////imgproc::cvt_color(&resized, &mut rgb_resized, imgproc::COLOR_BGR2RGB, 0)?;    
        //imgproc::cvt_color(&frame, &mut rgb_resized, imgproc::COLOR_BGR2RGB, 0)?;    
        //// get data from Mat 
        //let h = resized.size()?.height;
        //let w = resized.size()?.width;   
        let resized_data = frame.data_bytes_mut()?; 
        // convert bytes to tensor
        let tensor = tch::Tensor::of_data_size(resized_data, &[HEIGHT as i64, WIDTH as i64, 3], tch::Kind::Uint8);  
        // normalize image tensor
        let tensor = tensor.to_kind(tch::Kind::Float) / 255;
        // carry tensor to cuda
        // let tensor = tensor.to_device(tch::Device::Cuda(0)); 
        let tensor = tensor.to_device(tch::Device::Cpu); 
        // convert (H, W, C) to (C, H, W)
        let tensor = tensor.permute(&[2, 0, 1]); 
        // add batch dim (convert (C, H, W) to (N, C, H, W)) 
        let normalized_tensor = tensor.unsqueeze(0);   

        // make prediction and time it. 
        let start = time::Instant::now();
        let probabilities = model.forward_ts(&[normalized_tensor])?.softmax(-1, tch::Kind::Float);  
        let duration = start.elapsed();
        println!("Probabilities: {:?}, Duration: {:?}", probabilities,  duration); 

        // show image 
        highgui::imshow(WINDOW_NAME, &frame)?;
        // if button q pressed, abort.
        if highgui::wait_key(5)? == 113 { 
            highgui::destroy_all_windows()?;
            println!("Pressed q. Aborting program.");
            break;
        }
    }
  
    Ok(())
} 


use image;
use std::path::Path;
use std::fs::File;

pub fn export_png(name: String, buf: &[[u8; 3]], width: usize, height: usize) {
    assert!(width * height == buf.len());

    let mut imgbuf = image::ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let px = buf[(height - 1 - y as usize) * width + x as usize];
        *pixel = image::Rgb(px);
    }

    let ref mut out = File::create(&Path::new(&name)).unwrap();
    let _ = image::ImageRgb8(imgbuf).save(out, image::PNG);
}


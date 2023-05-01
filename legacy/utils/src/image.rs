extern crate image;

use self::image::{DynamicImage, GrayImage, ImageFormat, ImageResult, Pixel};
use std::path::Path;

pub fn hash(image: &DynamicImage) -> u64 {
  use std::collections::hash_map::DefaultHasher;
  use std::hash::{Hash, Hasher};
  let mut s = DefaultHasher::new();
  image.to_luma().to_vec().hash(&mut s);
  s.finish()
}

pub fn open_pgm(path: &Path) -> ImageResult<GrayImage> {
  use self::image::DynamicImage::ImageLuma8;

  self::image::open(&path).and_then(|img| match img {
    ImageLuma8(gi) => Ok(gi),
    _ => panic!("Expected PGM."),
  })
}

pub fn open_png_gray(path: &Path) -> ImageResult<GrayImage> {
  self::image::open(&path).map(|img| self::image::imageops::colorops::grayscale(&img))
}

pub fn save(img: &DynamicImage, path_string: &str) -> () {
  use std::fs::File;
  let path = Path::new(path_string);
  img.save(path).unwrap();
}

pub fn threshold_binary(img: &mut GrayImage, threshold: u8, inverse: bool) -> () {
  let min = if inverse { 255 } else { 0 };
  let max = if inverse { 0 } else { 255 };

  for p in img.pixels_mut() {
    *p = image::Luma([if p.channels()[0] > threshold { max } else { min }]);
  }
}

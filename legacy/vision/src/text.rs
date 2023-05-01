extern crate image;

extern crate anna_utils;

use std::collections::HashMap;

use image::{GrayImage, Pixel};

pub type Font<A> = HashMap<u32, (Vec<A>, Vec<GrayImage>)>;

pub fn parse<A: Clone>(font: &Font<A>, text: &mut GrayImage) -> Option<Vec<A>> {
  use image::GenericImage;

  let height = text.height();
  let mut xs: Vec<A> = Vec::new();

  for (b, e) in tokenize(text) {
    // println!("token: {:?} {:?}", b, e);
    let width = 1 + (e - b);
    let ref mut token = text.sub_image(b, 0, width, height).to_image();

    match token_parse_all(font, token) {
      None => {
        return None;
      }
      Some(values) => {
        xs.extend(values);
      }
    }
  }

  if xs.is_empty() {
    None
  } else {
    Some(xs)
  }
}

pub fn token_parse_all<A: Clone>(font: &Font<A>, token: &mut GrayImage) -> Option<Vec<A>> {
  use image::GenericImage;

  let token_width = token.width();
  let token_height = token.height();

  let mut widths: Vec<u32> = font.keys().map(|&u| u).collect();
  widths.sort();
  widths.reverse();

  for &width in widths.iter().skip_while(|&&w| w > token_width) {
    let value = if width < token_width {
      let ref mut token_sub = token.sub_image(0, 0, width, token_height).to_image();
      token_parse(font, token_sub)
    } else {
      token_parse(font, token)
    };

    for a in value {
      return {
        if width < token_width {
          let ref mut token_next =
            token.sub_image(width, 0, token_width - width, token_height).to_image();
          token_parse_all(font, token_next).map(|mut values| {
            values.insert(0, a);
            values
          })
        } else {
          Some(vec![a])
        }
      };
    }
  }

  /*
  use std::time::{SystemTime, UNIX_EPOCH};
  let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
  anna_utils::image::save(&self::image::DynamicImage::ImageLuma8(token.clone()),
                                 &format!("../resources/vision-templates/dominostars/samples/token-{}.pgm", now));
  */

  None
}

pub fn token_parse<A: Clone>(font: &Font<A>, token: &GrayImage) -> Option<A> {
  use template;
  font.get(&token.width()).and_then(|&(ref chars, ref tmpls)| {
    // template::match_best(&token, &tmpls, 0.001).map(|(idx, _, _)| chars[idx].clone())
    // template::match_best(&token, &tmpls, 0.007).map(|(idx, _, _)| chars[idx].clone())
    template::match_best(&token, &tmpls, 0.03).map(|(idx, _, _)| chars[idx].clone())
  })
}

pub fn tokenize(img: &GrayImage) -> Vec<(u32, u32)> {
  let mut tokens: Vec<(u32, u32)> = Vec::new();
  let mut token: Option<u32> = None;

  fn is_defined(img: &GrayImage, x: u32) -> bool {
    (0..img.height()).any(|y| img.get_pixel(x, y).channels()[0] < 255)
  }

  for x in 0..img.width() {
    match token {
      None => {
        if is_defined(img, x) {
          token = Some(x);
        }
      }
      Some(b) => {
        let connected = (0..img.height()).any(|y| {
          let previous = if x > 0 { img.get_pixel(x - 1, y).channels()[0] < 255 } else { false };

          (y < (img.height() - 1) && previous && img.get_pixel(x, y + 1).channels()[0] < 255)
            || (y > 0 && previous && img.get_pixel(x, y - 1).channels()[0] < 255)
            || (previous && img.get_pixel(x, y).channels()[0] < 255)
        });

        if !connected {
          tokens.push((b, x - 1));
          token = if is_defined(img, x) { Some(x) } else { None };
        }
      }
    }
  }

  tokens
}

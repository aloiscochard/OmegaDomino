extern crate image;
extern crate rayon;

use image::GrayImage;

pub fn match_all(
  image: &GrayImage,
  templates: &[GrayImage],
  threshold: f32,
) -> Vec<(usize, f32, (u32, u32))> {
  match_n(image, templates, threshold, None)
}

pub fn match_best(
  image: &GrayImage,
  templates: &[GrayImage],
  threshold: f32,
) -> Option<(usize, f32, (u32, u32))> {
  let mut xs = match_n(image, templates, threshold, None);
  xs.sort_by_key(|&(_, s, _)| (s * 1000.) as u32);
  xs.first().cloned()
}

pub fn match_first(
  image: &GrayImage,
  templates: &[GrayImage],
  threshold: f32,
) -> Option<(usize, f32, (u32, u32))> {
  match_n(image, templates, threshold, Some(1)).first().cloned()
}

/** Templates matching with threshold on score. (0.0 is perfect match) * *
 ** * * * * * * * * * * * * * * * * **/
pub fn match_n(
  image: &GrayImage,
  templates: &[GrayImage],
  threshold: f32,
  n: Option<usize>,
) -> Vec<(usize, f32, (u32, u32))> {
  use std::iter;

  let mut matches = Vec::new();

  let t_width = templates[0].width();
  let t_height = templates[0].height();

  assert!(t_width <= image.width());
  assert!(t_height <= image.height());

  for t in templates.iter().skip(1) {
    assert!(t.width() == t_width);
    assert!(t.height() == t_height);
  }

  let x_max = (image.width() - t_width) + 1;
  let y_max = (image.height() - t_height) + 1;

  let scores_init: Vec<usize> = iter::repeat(0).take(templates.len()).collect();

  for x in 0..x_max {
    for y in 0..y_max {
      let mut scores = scores_init.clone();

      for t_x in 0..t_width {
        for t_y in 0..t_height {
          let image_p = image.get_pixel(x + t_x, y + t_y)[0] as i32;
          for (i, template) in templates.iter().enumerate() {
            scores[i] += (image_p - template.get_pixel(t_x, t_y)[0] as i32).abs() as usize;
          }
        }
      }

      for (i, &score) in scores.iter().enumerate() {
        let score_n = score as f32 / (t_width * t_height * 256) as f32;
        // println!("score_n: {}", score_n);
        if score_n <= threshold {
          matches.push((i, score_n, (x, y)));

          if matches.len() == n.unwrap_or(0) {
            return matches;
          }
        }
      }
    }
  }

  matches
}

pub fn score(image: &GrayImage, template: GrayImage) -> f32 {
  assert!(image.width() == template.width());
  assert!(image.height() == template.height());
  let (_, score, _) = match_first(image, &vec![template], 1.0).unwrap();
  score
}

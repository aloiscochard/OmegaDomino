extern crate image;

extern crate anna_utils;
extern crate anna_vision;

#[test]
fn window_capture_match() {
  use anna_utils::xwindow;
  use anna_vision::template;
  use image::imageops::FilterType;
  use image::DynamicImage::ImageLuma8;
  use std::path::Path;

  let (display, window_root) = xwindow::display_open();

  let window = xwindow::window_find(display, window_root, "WM_NAME", "QEMU").unwrap();
  let (width, height) = xwindow::window_geometry(display, window);
  let ref mut image = xwindow::window_capture(display, window, 0, 0, width, height).unwrap();

  let path = Path::new("../resources/vision-templates/fulltilt/table.pgm");
  let template = self::image::open(&path).unwrap();

  match template {
    ImageLuma8(template) => {
      let ref mut template = template.clone();

      let r = 12;
      let matches12 = {
        let template_r = self::image::imageops::resize(
          template,
          template.width() / r,
          template.height() / r,
          FilterType::Nearest,
        );
        let image_r = self::image::imageops::resize(
          image,
          image.width() / r,
          image.height() / r,
          FilterType::Nearest,
        );

        template::match_first(&image_r, &vec![template_r], 0.14)
      };

      for (_, s, (x, y)) in matches12 {
        println!("12# {:?}", (s, (x * 12, y * 12)));
      }
    }
    _ => {
      panic!("Invalid template format.");
    }
  }

  xwindow::display_close(display);
}

extern crate image;
extern crate x11;

use std::{ffi, ptr};

//use self::image::{DynamicImage, GrayImage};
use self::image::{GrayImage, RgbImage};

use self::x11::xlib::{self, Display, Window, XImage};

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct Bgr8 {
    pub b: u8,
    pub g: u8,
    pub r: u8,
    _pad: u8,
}

pub fn display_open() -> (*mut Display, Window) {
  unsafe {
    let display = xlib::XOpenDisplay(ptr::null());
    let window_root = xlib::XDefaultRootWindow(display);
    (display, window_root)
  }
}

pub fn display_close(display: *mut Display) -> () {
  unsafe {
    xlib::XCloseDisplay(display);
  }
}

/** Capture window content as image, the coordinate are relative to the
 ** window itself. * * * * **/
pub fn window_capture(
  display: *mut Display,
  window: Window,
  x: i32,
  y: i32,
  width: u32,
  height: u32,
) -> Option<GrayImage> {
  use self::image::DynamicImage::ImageRgb8;

  let color_image = window_capture_color(display, window, x, y, width, height);
  match color_image {
    None => None,
    Some(img) => Some(ImageRgb8(img).to_luma())
  }
}

pub fn window_capture_color(
  display: *mut Display,
  window: Window,
  x: i32,
  y: i32,
  width: u32,
  height: u32,
) -> Option<RgbImage> {
  unsafe {
    let zpixmap_ptr =
      xlib::XGetImage(display, window, x, y, width, height, xlib::XAllPlanes(), xlib::ZPixmap);
    if zpixmap_ptr.is_null() {
      None
    } else {
      let zpixmap = &mut *zpixmap_ptr;
      let image = zpixmap_read(*zpixmap);
      xlib::XDestroyImage(zpixmap_ptr);
      Some(image)
    }
  }
}

pub fn window_find(
  display: *mut Display,
  window_root: Window,
  attr: &str,
  name: &str,
) -> Option<Window> {
  let mut root: u64 = 0;
  let mut childrens: *mut xlib::Window = ptr::null_mut();
  let mut childrens_size: u32 = 0;

  unsafe {
    let status = xlib::XQueryTree(
      display,
      window_root,
      &mut root,
      &mut 0,
      &mut childrens,
      &mut childrens_size,
    );
    assert!(status != 0);

    for i in 0..childrens_size {
      let window = *childrens.offset(i as isize);
      let mut props_size: i32 = 0;
      let props = xlib::XListProperties(display, window, &mut props_size);

      for j in 0..props_size {
        let prop = *props.offset(j as isize);
        let prop_name = ffi::CString::from_raw(xlib::XGetAtomName(display, prop));

        if prop_name.to_str().unwrap() == attr {
          let mut prop_value: *mut u8 = ptr::null_mut();
          let mut items_size: u64 = 0;

          let status = xlib::XGetWindowProperty(
            display,
            window,
            prop,
            0,
            65536,
            0,
            xlib::AnyPropertyType as u64,
            &mut 0,
            &mut 0,
            &mut items_size,
            &mut 0,
            &mut prop_value,
          );
          assert!(status == 0);

          let mut vec = Vec::new();

          for k in 0..items_size {
            let value = *prop_value.offset(k as isize);
            vec.push(value as char);
          }

          let window_name: String = vec.into_iter().collect();

          if window_name == name {
            return Some(window);
          }
        }
      }
    }

    for i in 0..childrens_size {
      let window = *childrens.offset(i as isize);
      match window_find(display, window, attr, name) {
        Some(win) => {
          return Some(win);
        }
        _ => { /* NOOP */ }
      }
    }
  }

  None
}

pub fn window_geometry(display: *mut Display, window: Window) -> (u32, u32) {
  let (mut width, mut height) = (0, 0);
  unsafe {
    let status = xlib::XGetGeometry(
      display,
      window,
      &mut 0,
      &mut 0,
      &mut 0,
      &mut width,
      &mut height,
      &mut 0,
      &mut 0,
    );
    assert!(status > 0);
  }
  (width, height)
}

/*
fn zpixmap_read_luma(image: XImage) -> GrayImage {
  let xs: &[Bgr8] = unsafe { std::slice::from_raw_parts(image.data as *const _,
                                  image.width as usize * image.height as usize) };

  unimplemented!()
}
*/

fn zpixmap_read(image: XImage) -> RgbImage {
  use self::image::{RgbImage};

  let mut bs = Vec::new();

  for y in 0..image.height {
    for x in 0..image.width {
      let offset = ((x * (image.bits_per_pixel / 8)) + (y * image.bytes_per_line)) as isize;

      unsafe {
        let b = *image.data.offset(offset);
        let g = *image.data.offset(offset + 1);
        let r = *image.data.offset(offset + 2);

        bs.push(r as u8);
        bs.push(g as u8);
        bs.push(b as u8);
      }
    }
  }

  RgbImage::from_raw(image.width as u32, image.height as u32, bs).unwrap()
}

/*
fn zpixmap_read_luma(image: XImage) -> GrayImage {
  let mut bs = Vec::new();

  for y in 0..image.height {
    for x in 0..image.width {
      let offset = ((x * (image.bits_per_pixel / 8)) + (y * image.bytes_per_line)) as isize;

      unsafe {
        let b = *image.data.offset(offset);
        let g = *image.data.offset(offset + 1);
        let r = *image.data.offset(offset + 2);

        let g = (r as f32 * 0.2126) + (g as f32 * 0.7152) + (b as f32 * 0.0722);
        bs.push(g as u8);
      }
    }
  }

  GrayImage::from_raw(image.width as u32, image.height as u32, bs).unwrap()
}
*/

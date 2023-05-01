extern crate image;

extern crate anna_model;
extern crate anna_utils;
extern crate anna_vision;

use image::{GenericImage, GrayImage, Pixel};
use std::collections::HashMap;
use std::path::Path;

use anna_model::{SeatId, Money};

use anna_vision::dominostars::*;
use anna_vision::text;


#[test]
pub fn ps_find_table_cards_get() {

  let path_templates = Path::new(
    "../resources/vision-templates/dominostars/",
  );
  let tmpls = templates_load(path_templates);

  let path_table = path_templates.join("samples").join("abort-1604995522138.pgm");
  let mut table_img = image::open(&path_table).unwrap().to_luma();

  for card_id in 0..5 {
    let card = table_cards_get(&tmpls, &mut table_img, card_id);
    println!("{} - card: {:?}", card_id, card);
  }
}

#[test]
pub fn ps_find_player_cards_process() {
  use std::fs;
  let path_templates = Path::new(
    "../resources/vision-templates/dominostars/",
  );
  let tmpls = templates_load(path_templates);
  let path_input = path_templates.join("samples").join("playercards");
  let paths = fs::read_dir(path_input.clone()).unwrap();

  for res in paths {
    let path = res.unwrap().path();
    let path_str = path.to_string_lossy();

    println!("{:?}", path);
    let path_table = path;
    let mut table_img = image::open(&path_table).unwrap().to_luma();

    for seat_id in 0..6 {
      let state = player_cards_state(&tmpls, &mut table_img, seat_id);
      let cards = if state.is_some() {
        player_cards_get(&tmpls, &mut table_img, seat_id)
      } else { None } ;
      println!("{} - cards: {:?} - state: {:?}", seat_id, cards, state);
    }
  }
}


#[test]
pub fn ps_find_player_cards_get() {

  let path_templates = Path::new(
    "../resources/vision-templates/dominostars/",
  );
  let tmpls = templates_load(path_templates);

  let path_table = path_templates.join("samples/playercards").join("table_playercards_0013.pgm");
  let mut table_img = image::open(&path_table).unwrap().to_luma();

  for seat_id in 0..6 {
    let state = player_cards_state(&tmpls, &mut table_img, seat_id);
    let cards = if state.is_some() {
      player_cards_get(&tmpls, &mut table_img, seat_id)
    } else { None } ;
    println!("{} - cards: {:?} - state: {:?}", seat_id, cards, state);
  }
}

#[test]
pub fn ps_find_player_active() {

  let path_templates = Path::new(
    "../resources/vision-templates/dominostars/",
  );

  let path_table = path_templates.join("samples").join("abort-1605018553489.pgm");
  // let path_table = path_templates.join("samples").join("abort-1604997093259.pgm");
  // let path_table = path_templates.join("samples").join("table_0007.pgm");
  // let path_table = path_templates.join("samples").join("abort-1604936546115.pgm");

  let mut table_img = image::open(&path_table).unwrap().to_luma();
  let active = player_active_find(&mut table_img);

  println!("active: {:?}", active);
}

#[test]
pub fn ps_find_player_funds() {

  let path_templates = Path::new(
    "../resources/vision-templates/dominostars/",
  );
  let tmpls = templates_load(path_templates);

  let path_table = path_templates.join("samples").join("table_0013.pgm");
  let mut table_img = image::open(&path_table).unwrap().to_luma();

  for seat_id in 4..5 {
    let funds = player_funds_get(&tmpls, &mut table_img, seat_id);
    // let state = player_funds_state(&tmpls, &mut table_img, seat_id);
    println!("{} - funds: {:?}", seat_id, funds);
  }
}

#[test]
pub fn ps_find_player_pledge() {

  let path_templates = Path::new(
    "../resources/vision-templates/dominostars/",
  );
  let tmpls = templates_load(path_templates);

  let path_table = path_templates.join("samples").join("table_0008.pgm");
  let mut table_img = image::open(&path_table).unwrap().to_luma();

  for seat_id in 4..5 {
    let pledge = player_pledge_get(&tmpls, &mut table_img, seat_id);
    println!("{} - pledge: {:?}", seat_id, pledge);
  }
}

#[test]
pub fn ps_find_dealer() {

  let path_templates = Path::new(
    "../resources/vision-templates/dominostars/",
  );
  let path_table = path_templates.join("samples").join("table_0001.pgm");

  let mut table_img = image::open(&path_table).unwrap().to_luma();

  let seat_id = dealer_find(&mut table_img);
  println!("seat_id: {:?}", seat_id);
}

#[test]
pub fn ps_process_extract_table() {
  use std::fs;

  let path_templates = Path::new(
    "../resources/vision-templates/dominostars/",
  );
  let path_input = path_templates.join("input");

  let tmpls = templates_load(path_templates);

  let paths = fs::read_dir(path_input.clone()).unwrap();

  for res in paths {
    let path = res.unwrap().path();
    let path_str = path.to_string_lossy();

    let mut screen_img = image::open(&path).unwrap().to_luma();
    let coord = table_find(&tmpls, &screen_img);

    for (x, y) in coord {
      println!("{:?} - {:?}", path_str, (x, y));
      let table_img = screen_img.sub_image(x, y, TABLE_WIDTH as _, TABLE_HEIGHT as _).to_image();
      let table_path = path_input.join(format!("{}.table.pgm",path.file_stem().unwrap().to_string_lossy()));
      println!("{:?}", table_path);
      table_img.save(table_path).unwrap();
    }
  }
}

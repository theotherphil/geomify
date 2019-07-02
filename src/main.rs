//! Based on https://github.com/fogleman/primitive

use image::{Rgb, RgbImage};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::error::Error;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "geomify")]
struct Opt {
    /// Input image path.
    #[structopt(short = "i", long, parse(from_os_str))]
    input: PathBuf,

    /// Output image path.
    #[structopt(short = "o", long, parse(from_os_str))]
    output: PathBuf,

    /// Number of shapes used in approximation of input image.
    #[structopt(short = "n", long)]
    num_shapes: u64,
}

// TODO
// alpha
// more shapes
// hill climbing
// logging/intermediate results
// non-deterministic seeding
// svg output
// whatever else the go version does that's useful
// performance
// other optimisation methods

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    println!("{:#?}", opt);

    let image = image::open(opt.input)?.to_rgb();
    let (width, height) = image.dimensions();

    println!("Input size: ({}, {})", width, height);

    let background = average_image_colour(&image);
    let mut current_image = RgbImage::from_pixel(width, height, background);

    let num_samples = 1000;

    // Totally deterministic for now.
    let mut rng = StdRng::seed_from_u64(1);

    for n in 0..opt.num_shapes {
        println!("Shape: {}", n);

        let mut best_candidate = current_image.clone();
        let mut least_error = std::u64::MAX;

        for _ in 0..num_samples {
            let rect = Rect::generate_random(&mut rng, width, height);
            let rect_colour = average_colour_within_rect(&image, &rect);
            let candidate = draw_rect(&current_image, &rect, rect_colour);

            let error = sum_squared_errors(&image, &candidate);
            if error < least_error {
                least_error = error;
                best_candidate = candidate;
            }
        }

        current_image = best_candidate;
    }

    current_image.save(&opt.output)?;

    Ok(())
}

fn draw_rect(image: &RgbImage, rect: &Rect, colour: Rgb<u8>) -> RgbImage {
    // Shouldn't allocate every time here, but not worrying at
    // all about performance for now.
    //
    // We could also replace a load of code here by using functions
    // from image or imageproc.
    let mut result = image.clone();
    for y in rect.top..(rect.top + rect.height) - 1 {
        for x in rect.left..(rect.left + rect.width) - 1 {
            result.put_pixel(x, y, colour);
        }
    }

    result
}

trait Random {
    fn generate_random<R: Rng>(rng: &mut R, image_width: u32, image_height: u32) -> Self;
}

impl Random for Rect {
    fn generate_random<R: Rng>(rng: &mut R, image_width: u32, image_height: u32) -> Self {
        assert!(image_width > 0 && image_height > 0, "Come on now");
        // The upper bounds here are deliberate - gen_range has
        // an exclusive upper bound, but we need to allow for a rectangle
        // of width and height at least 1.
        //
        // This is different to the approach taken in `primitive`, as
        // 1. they always generate rectangles with dimensions less than 32
        // 2. they generate uniformly in full range and then clamp
        let left = rng.gen_range(0, image_width - 1);
        let top = rng.gen_range(0, image_height - 1);
        let width = rng.gen_range(0, image_width - left);
        let height = rng.gen_range(0, image_width - top);

        Rect { left, top, width, height }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Rect {
    left: u32,
    top: u32,
    width: u32,
    height: u32,
}

fn sum_squared_errors(target: &RgbImage, candidate: &RgbImage) -> u64 {
    assert!(target.dimensions() == candidate.dimensions());
    let mut sum = 0u64;
    for y in 0..target.height() {
        for x in 0..target.width() {
            let t = target.get_pixel(x, y).data;
            let c = candidate.get_pixel(x, y).data;

            let (dr, dg, db) = (
                t[0] as i32 - c[0] as i32,
                t[1] as i32 - c[1] as i32,
                t[2] as i32 - c[2] as i32
            );
            sum += (dr * dr) as u64
                + (dg * dg) as u64
                + (db * db) as u64;
        }
    }
    sum
}

/// We should use integral images here instead. Won't work for
/// non-rectangular shapes, but will make a big different for rects.
fn average_colour_within_rect(image: &RgbImage, rect: &Rect) -> Rgb<u8> {
    let mut avg = [0u64, 0, 0];
    if rect.width == 0 || rect.height == 0 {
        return Rgb([avg[0] as u8, avg[1] as u8, avg[2] as u8]);
    }
    let mut count = 0u64;
    for y in rect.top..(rect.top + rect.height) {
        for x in rect.left..(rect.left + rect.width) {
            let p = image.get_pixel(x, y);
            avg[0] += p.data[0] as u64;
            avg[1] += p.data[1] as u64;
            avg[2] += p.data[2] as u64;
            count += 1;
        }
    }
    avg[0] /= count;
    avg[1] /= count;
    avg[2] /= count;
    Rgb([avg[0] as u8, avg[1] as u8, avg[2] as u8])
}

fn average_image_colour(image: &RgbImage) -> Rgb<u8> {
    let mut avg = [0u64, 0, 0];
    let mut count = 0u64;
    for p in image.pixels() {
        avg[0] += p.data[0] as u64;
        avg[1] += p.data[1] as u64;
        avg[2] += p.data[2] as u64;
        count += 1;
    }
    avg[0] /= count;
    avg[1] /= count;
    avg[2] /= count;
    Rgb([avg[0] as u8, avg[1] as u8, avg[2] as u8])
}

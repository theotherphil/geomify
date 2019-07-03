//! Based on https://github.com/fogleman/primitive

use image::{Rgb, RgbImage};
use imageproc::{
    drawing::draw_filled_rect,
    definitions::Image,
    integral_image::{integral_image, sum_image_pixels},
    rect::Rect
};
use log::{info, debug, trace};
use rand::{Rng, SeedableRng, rngs::StdRng};
use simplelog::{TermLogger, LevelFilter, Config, TerminalMode};
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

    /// Number of starting shapes used when attempting to add
    /// the next shape to the approximation.
    #[structopt(short = "s", long, default_value = "1000")]
    num_samples: u64,

    /// Number of attempts made to improve each random sample
    /// via mutation before giving up.
    #[structopt(short = "a", long, default_value = "10")]
    num_attempts: u64,

    /// Verbosity of logging.
    #[structopt(short = "v", parse(from_occurrences))]
    verbosity: u64,
}

// TODO
// alpha
// more shapes
// hill climbing
// better logging/intermediate results
// non-deterministic seeding
// svg output
// whatever else the go version does that's useful
// performance
// other optimisation methods

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    let level_filter = match opt.verbosity {
        0 => LevelFilter::Info,
        1 => LevelFilter::Debug,
        _ => LevelFilter::Trace
    };
    let config = Config {
        location: None,
        .. Default::default()
    };
    let _ = TermLogger::init(level_filter, config, TerminalMode::Mixed);

    info!("{:#?}", opt);

    let target = image::open(opt.input)?.to_rgb();
    let (width, height) = target.dimensions();
    let target_integral = integral_image::<_, u64>(&target);

    info!("Input size: ({}, {})", width, height);

    let background = average_image_colour(&target);
    let mut current_image = RgbImage::from_pixel(width, height, background);

    // Totally deterministic for now.
    let mut rng = StdRng::seed_from_u64(1);

    for n in 0..opt.num_shapes {
        info!("Shape: {}", n);

        let mut best_candidate = current_image.clone();
        let mut least_error = std::u64::MAX;

        for s in 0..opt.num_samples {
            debug!("Sample: {}", s);

            let rect = Rect::generate_random(&mut rng, width, height);
            let (error, candidate) = hill_climb(
                &mut rng,
                &target,
                &target_integral,
                &current_image,
                opt.num_attempts,
                rect);

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

/// Returns (least error, image with best rect added)
fn hill_climb<R: Rng>(
    rng: &mut R,
    target: &RgbImage,
    target_integral: &Image<Rgb<u64>>,
    current: &RgbImage,
    num_attempts: u64,
    start: Rect
) -> (u64, RgbImage)
{
    let (width, height) = target.dimensions();
    let mut attempts = 0;

    let rect_colour = average_colour_within_rect(&target, &start);
    let candidate = draw_rect(current, &start, rect_colour);
    // Using draw_filled_rect increases the runtime of the following
    // from 1.82s to 5.3s:
    // time cargo run --release -- -i ../mona_lisa.png -n 10 -a 2 -s 50 -o ../result.png
    //let candidate = draw_filled_rect(current, start, rect_colour);

    let mut least_error = sum_squared_errors(&target, &candidate);
    let mut best_candidate = candidate;
    let mut best_rect = start;

    while attempts < num_attempts {
        trace!("Attempt: {}", attempts);

        let mutated = best_rect.mutate(rng, width, height);
        let rect_colour = average_colour_within_rect(&target, &mutated);
        let candidate = draw_rect(current, &mutated, rect_colour);
        // See comment on using draw_rect above
        //let candidate = draw_filled_rect(current, mutated, rect_colour);
        let error = sum_squared_errors(&target, &candidate);

        if error < least_error {
            attempts = 0;
            least_error = error;
            best_rect = mutated;
            best_candidate = candidate;
        } else {
            attempts += 1;
        }
    }

    (least_error, best_candidate)
}

fn draw_rect(image: &RgbImage, rect: &Rect, colour: Rgb<u8>) -> RgbImage {
    // Shouldn't allocate every time here, but not worrying at
    // all about performance for now.
    //
    // We could also replace a load of code here by using functions
    // from image or imageproc.
    let mut result = image.clone();
    for y in rect.top() as u32..(rect.top() as u32 + rect.height()) {
        for x in rect.left() as u32..(rect.left() as u32 + rect.width()) {
            result.put_pixel(x, y, colour);
        }
    }

    result
}

trait Mutate {
    fn mutate<R: Rng>(
        &self,
        rng: &mut R,
        image_width: u32,
        image_height: u32
    ) -> Self;
}

impl Mutate for Rect {
    fn mutate<R: Rng>(
        &self,
        rng: &mut R,
        image_width: u32,
        image_height: u32
    ) -> Self {
        let choice = rng.gen_range(0, 2);
        let (c1, c2): (i32, i32) = (
            rng.gen_range(-10, 10),
            rng.gen_range(-10, 10)
        );
        let (left, top, width, height) = if choice == 0 {
            (
                self.left() + c1,
                self.top() + c2,
                self.width() as i32,
                self.height() as i32
            )
        } else {
            (
                self.left(),
                self.top(),
                self.width() as i32 + c1,
                self.height() as i32 + c2
            )
        };

        let (w, h) = (image_width as i32, image_height as i32);

        let left = clamp(left, 0, w - 2);
        let top = clamp(top, 0, h - 2);
        let width = clamp(width, 1, w - left as i32 - 1);
        let height = clamp(height, 1, h - top as i32 - 1);

        Rect::at(left, top).of_size(width as u32, height as u32)
    }
}

fn clamp(x: i32, min: i32, max: i32) -> i32 {
    if x < min { min } else if x > max { max } else { x }
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
        let width = rng.gen_range(1, image_width - left);
        let height = rng.gen_range(1, image_width - top);

        Rect::at(left as i32, top as i32).of_size(width, height)
    }
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

// fn average_colour_within_rect(integral_image: &Image<Rgb<u64>>, rect: &Rect) -> Rgb<u8> {
//     // The bounds for sum_image_pixels are inclusive
//     let mut avg = sum_image_pixels(
//         integral_image,
//         rect.left,
//         rect.top,
//         rect.left + rect.width - 1,
//         rect.top + rect.height - 1
//     );
//     let count = rect.width as u64 * rect.height as u64;
//     avg[0] /= count;
//     avg[1] /= count;
//     avg[2] /= count;
//     Rgb([avg[0] as u8, avg[1] as u8, avg[2] as u8])
// }

fn average_colour_within_rect(image: &Image<Rgb<u8>>, rect: &Rect) -> Rgb<u8> {
    let mut avg = [0u64, 0, 0];
    if rect.width() == 0 || rect.height() == 0 {
        return Rgb([avg[0] as u8, avg[1] as u8, avg[2] as u8]);
    }
    let mut count = 0u64;
    for y in rect.top()..rect.bottom() + 1 {
        for x in rect.left()..rect.right() + 1 {
            let p = image.get_pixel(x as u32, y as u32);
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

//! Based on https://github.com/fogleman/primitive

use image::{Rgb, RgbImage};
use imageproc::{
    drawing::draw_filled_rect_mut,
    definitions::Image,
    integral_image::{integral_image, integral_squared_image, sum_image_pixels},
    map::map_colors2,
    rect::Rect
};
use log::{info, debug, trace};
use rand::{Rng, thread_rng};
use rayon::prelude::*;
use simplelog::{TermLogger, LevelFilter, Config, TerminalMode};
use std::cmp;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Mutex;
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

    let target = image::open(&opt.input)?.to_rgb();
    let (width, height) = target.dimensions();
    let target_integral = integral_image::<_, u64>(&target);

    info!("Input size: ({}, {})", width, height);

    let background = average_colour_within_rect(
        &target_integral,
        &Rect::at(0, 0).of_size(target.width(), target.height())
    );

    let mut current_image = RgbImage::from_pixel(width, height, background);

    for n in 0..opt.num_shapes {
        info!("Shape: {}", n);

        let current_diff = map_colors2(
            &target,
            &current_image,
            |p, q| Rgb([
                cmp::max(p[0], q[0]) - cmp::min(p[0], q[0]),
                cmp::max(p[1], q[1]) - cmp::min(p[1], q[1]),
                cmp::max(p[2], q[2]) - cmp::min(p[2], q[2])
            ])
        );
        let current_diff_integral_squared = integral_squared_image::<_, u64>(&current_diff);

        let current_error = sum_image_pixels_within_rect(
            &current_diff_integral_squared,
            &Rect::at(0, 0).of_size(target.width(), target.height())
        ).iter().sum();

        let best = Mutex::new(
            (
                Rect::at(0, 0).of_size(1, 1),
                Rgb([0, 0, 0]),
                std::u64::MAX
            )
        );

        (0..opt.num_samples).into_par_iter()
            .for_each(|s| {
                debug!("Sample: {}", s);

                let mut rng = thread_rng();

                let rect = Rect::generate_random(&mut rng, width, height);
                let (error, rect, colour) = hill_climb(
                    &mut rng,
                    &target,
                    &target_integral,
                    opt.num_attempts,
                    rect,
                    current_error,
                    &current_diff_integral_squared
                );

                let mut best = best.lock().unwrap();
                if error < best.2 {
                    best.0 = rect;
                    best.1 = colour;
                    best.2 = error;
                }
            });

        let best = best.lock().unwrap();
        current_image = draw_rect(&current_image, &best.0, best.1);
    }

    current_image.save(&opt.output)?;

    Ok(())
}

/// Returns the sum of squared errors after drawing the given
/// rectangle on the current image with colour equal to the
/// average colour of the target image in that rectangle.
fn sum_squared_errors_after_drawing_rect(
    current_error: u64,
    current_diff_integral_squared: &Image<Rgb<u64>>,
    target: &RgbImage,
    target_integral: &Image<Rgb<u64>>,
    rect: &Rect
) -> u64 {
    let error_in_rect_after_drawing = sum_squares_difference_from_average(
        target,
        target_integral,
        rect
    );
    let channel_errors_in_rect_before_drawing = sum_image_pixels_within_rect(
        current_diff_integral_squared,
        rect
    );
    let error_in_rect_before_drawing: u64 = channel_errors_in_rect_before_drawing
        .iter()
        .sum();

    current_error - error_in_rect_before_drawing + error_in_rect_after_drawing
}

/// Returns (least error, best rect, best rect colour)
fn hill_climb<R: Rng>(
    rng: &mut R,
    target: &RgbImage,
    target_integral: &Image<Rgb<u64>>,
    num_attempts: u64,
    start: Rect,
    current_error: u64,
    current_diff_integral_squared: &Image<Rgb<u64>>
) -> (u64, Rect, Rgb<u8>)
{
    let (width, height) = target.dimensions();
    let mut attempts = 0;

    let mut least_error = sum_squared_errors_after_drawing_rect(
        current_error,
        current_diff_integral_squared,
        target,
        target_integral,
        &start
    );
    trace!("Starting error for hill climb: {}", least_error);
    let mut best_rect = start;

    while attempts < num_attempts {
        trace!("Attempt: {}", attempts);

        let mutated = best_rect.mutate(rng, width, height);
        let error = sum_squared_errors_after_drawing_rect(
            current_error,
            current_diff_integral_squared,
            target,
            target_integral,
            &mutated
        );

        if error < least_error {
            trace!("Improved error for hill climb: {}", error);
            attempts = 0;
            least_error = error;
            best_rect = mutated;
        } else {
            attempts += 1;
        }
    }

    let colour = average_colour_within_rect(target_integral, &best_rect);
    (least_error, best_rect, colour)
}

fn draw_rect(image: &RgbImage, rect: &Rect, colour: Rgb<u8>) -> RgbImage {
    // This is a bit sad. draw_filled_rect in imageproc allocates a
    // fresh output image and calls GenericImage::copy_from to populate
    // it. This performs elementwise copies, because the impl is for any
    // GenericImage and can't be specialised for when the input is actually
    // an ImageBuffer.
    let mut result = image.clone();
    draw_filled_rect_mut(&mut result, *rect, colour);
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

/// Computes the average image colour in the provided rectangle, and returns the
/// sum over the pixels in the rectangle of the square of the differences between
/// the actual and average pixel values.
fn sum_squares_difference_from_average(
    image: &RgbImage,
    integral_image: &Image<Rgb<u64>>, rect: &Rect
) -> u64 {
    let avg = average_colour_within_rect(integral_image, rect);

    if rect.width() == 0 || rect.height() == 0 {
        return 0;
    }

    let mut sum_sq = 0;
    for y in rect.top()..rect.bottom() + 1 {
        for x in rect.left()..rect.right() + 1 {
            let p = image.get_pixel(x as u32, y as u32);
            let (dr, dg, db) = (
                p[0] as i32 - avg[0] as i32,
                p[1] as i32 - avg[1] as i32,
                p[2] as i32 - avg[2] as i32
            );
            sum_sq += (dr * dr) as u64
                    + (dg * dg) as u64
                    + (db * db) as u64;
        }
    }

    sum_sq
}

fn sum_image_pixels_within_rect(integral_image: &Image<Rgb<u64>>, rect: &Rect) -> [u64; 3] {
    // The bounds for sum_image_pixels are inclusive
    sum_image_pixels(
        integral_image,
        rect.left() as u32,
        rect.top() as u32,
        rect.left() as u32 + rect.width() - 1,
        rect.top() as u32 + rect.height() - 1
    )
}

fn average_colour_within_rect(integral_image: &Image<Rgb<u64>>, rect: &Rect) -> Rgb<u8> {
    let mut avg = sum_image_pixels_within_rect(integral_image, rect);
    let count = rect.width() as u64 * rect.height() as u64;
    avg[0] /= count;
    avg[1] /= count;
    avg[2] /= count;
    Rgb([avg[0] as u8, avg[1] as u8, avg[2] as u8])
}

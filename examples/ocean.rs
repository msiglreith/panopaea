
extern crate panopaea;
extern crate panopaea_utils;

use panopaea::ocean::empirical as ocean;

fn main() {
    let parameters = ocean::Parameters {
        water_depth: 100.0,
        fetch: 800.0 * 1000.0,
        wind_speed: 25.0,

        water_density: 1000.0,
        surface_tension: 0.072,
        gravity: 9.81,

        swell: 0.25,
        domain_size: 1000.0,
    };

    let spectrum = ocean::SpectrumTMA {
        jonswap: ocean::SpectrumJONSWAP {
            wind_speed: parameters.wind_speed,
            fetch: parameters.fetch,
            gravity: parameters.gravity,
        },
        depth: parameters.water_depth,
    };

    let mut ocean = ocean::Ocean::new(1<<9);
    let (height_spectrum, omega) = ocean.build_height_spectrum(&parameters, &spectrum);
    let mut displacement = ocean.new_displacement();

    {
        let (img_data, dim) = {
            let mut data = Vec::new();
            for y in 0 .. omega.dim().0 {
                for x in 0 .. omega.dim().1 {
                    let val = &omega[(y, x)];
                    data.push([
                        panopaea_utils::imgproc::transfer(&val, -40.0, 40.0),
                        0,
                        0,
                    ]);
                }
            }
            (data, (omega.dim().1, omega.dim().0))
        };

        panopaea_utils::png::export(
            format!("omega.png").to_owned(),
            &img_data,
            dim);
    }

    {
        let (img_data, dim) = {
            let mut data = Vec::new();
            for y in 0 .. height_spectrum.dim().0 {
                for x in 0 .. height_spectrum.dim().1 {
                    let val = &height_spectrum[(y, x)];
                    data.push([
                        panopaea_utils::imgproc::transfer(&val.re, -0.1, 0.1),
                        panopaea_utils::imgproc::transfer(&val.im, -0.1, 0.1),
                        0,
                    ]);
                }
            }
            (data, (height_spectrum.dim().1, height_spectrum.dim().0))
        };

        panopaea_utils::png::export(
            format!("height_spectrum.png").to_owned(),
            &img_data,
            dim);
    }

    for i in 0..30 {
        ocean.propagate(i as f64 / 2.0f64, &parameters, height_spectrum.view(), omega.view(), displacement.view_mut());

        let (img_data, dim) = {
            let mut data = Vec::new();
            // let (density, _) = vel.split();
            for y in 0 .. displacement.dim().0 {
                for x in 0 .. displacement.dim().1 {
                    let val = &displacement[(y, x)];
                    data.push([
                        panopaea_utils::imgproc::transfer(&val.x, -20.0, 20.0),
                        panopaea_utils::imgproc::transfer(&val.y, -20.0, 20.0),
                        panopaea_utils::imgproc::transfer(&val.z, -20.0, 20.0),
                    ]);
                }
            }
            (data, (displacement.dim().1, displacement.dim().0))
        };

        panopaea_utils::png::export(
            format!("displacement_{}.png", i).to_owned(),
            &img_data,
            dim);
    }
}


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
    };

    let spectrum = ocean::SpectrumTMA {
        jonswap: ocean::SpectrumJONSWAP {
            wind_speed: parameters.wind_speed,
            fetch: parameters.fetch,
            gravity: parameters.gravity,
        },
        depth: parameters.water_depth,
    };


    let height_spectrum = ocean::build_height_spectrum(&parameters, &spectrum, 1000.0f64, 1 << 9);

    let (img_data, dim) = {
        let mut data = Vec::new();
        // let (density, _) = vel.split();
        for y in 0 .. height_spectrum.dim().0 {
            for x in 0 .. height_spectrum.dim().1 {
                let val = &height_spectrum[(y, x)];
                data.push([
                    panopaea_utils::imgproc::transfer(&val.0, -0.1, 0.1),
                    panopaea_utils::imgproc::transfer(&val.1, -0.1, 0.1),
                    panopaea_utils::imgproc::transfer(&0.0, -2.0, 2.0),
                ]);
            }
        }
        (data, (height_spectrum.dim().1, height_spectrum.dim().0))
    };

    panopaea_utils::png::export(
        "height_spectrum.png".to_owned(),
        &img_data,
        dim);
}

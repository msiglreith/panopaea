
extern crate cgmath;
#[macro_use] extern crate gfx;
extern crate gfx_window_glutin;
extern crate glutin;
extern crate panopaea;
extern crate generic_array;
extern crate rayon;

use gfx::{Bind, Device, Factory};
use gfx::traits::FactoryExt;

use panopaea::*;
use panopaea::particle::{Particles, Property};

use cgmath::Transform;
use generic_array::typenum::U2;
use rayon::prelude::*;

pub type ColorFormat = gfx::format::Rgba8;
pub type DepthFormat = gfx::format::DepthStencil;

gfx_defines!{
    vertex Vertex {
        pos: [f32; 2] = "a_Pos",
        color: [f32; 3] = "a_Color",
    }

    constant Locals {
        view: [[f32; 4]; 4] = "u_View",
        proj: [[f32; 4]; 4] = "u_Proj",
        particle_size: f32 = "u_ParticleSize",
    }

    pipeline pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        locals: gfx::ConstantBuffer<Locals> = "Locals",
        out_color: gfx::BlendTarget<ColorFormat> = ("Target0", gfx::state::ColorMask::all(), gfx::preset::blend::ALPHA),
    }
}

impl Property for Vertex {
    type Subtype = Vertex;
    fn new() -> Self {
        Vertex {
            pos: [0.0; 2],
            color: [0.0; 3]
        }
    }
}

fn main() {
    // rayon::initialize(rayon::Configuration::new().num_threads(1));

    let builder = glutin::WindowBuilder::new()
        .with_dimensions(1440, 900);
        // .with_vsync();
    let (window, mut device, mut factory, main_color, _) =
        gfx_window_glutin::init::<ColorFormat, DepthFormat>(builder);
    let mut encoder: gfx::Encoder<_, _> = factory.create_command_buffer().into();

    let vs = factory.create_shader_vertex(include_bytes!("data/particles.vs.glsl")).unwrap();
    let gs = factory.create_shader_geometry(include_bytes!("data/particles.gs.glsl")).unwrap();
    let ps = factory.create_shader_pixel(include_bytes!("data/particles.fs.glsl")).unwrap();
    let pso = factory.create_pipeline_state(
        &gfx::ShaderSet::Geometry(vs, gs, ps),
        gfx::Primitive::PointList,
        gfx::state::Rasterizer::new_fill(),
        pipe::new()
    ).unwrap();

    let vbuf = factory.create_buffer(
        32 * 1024,
        gfx::buffer::Role::Vertex,
        gfx::memory::Usage::Dynamic,
        Bind::empty()
    ).unwrap();

    let mut data = pipe::Data {
        vbuf: vbuf,
        locals: factory.create_constant_buffer(1),
        out_color: main_color,
    };

    let smoothing = 2.0;
    let rest_density = 1.0;
    let timestep = 0.01f32;
    let mass = 3.5f32;
    // let viscosity = 3.5f32;
    let border = 40.0;
    let mut particles = Particles::new();
    pbd::init::<f32, U2>(&mut particles);
    particles.add_property::<Vertex>();

    let mut grid = sph::grid::BoundedGrid::new(math::vector_n::vec2(256, 256), smoothing);

    {
        let mut positions = Vec::new();
        let mut masses = Vec::new();
        for y in 0..40u8 {
            for x in 0..20u8 {
                let mut pos = sph::property::Position::new();
                pos[0] = (x as f32) * smoothing * 0.75;
                pos[1] = (y as f32) * smoothing * 0.75;
                positions.push(pos);
                masses.push(mass);
            }
        }

        particles.add_particles(positions.len())
                 .with::<sph::property::Position<f32, U2>>(&positions)
                 .with::<sph::property::Mass<f32>>(&masses);
    }

    let (width, height) = window.get_inner_size_points().unwrap();
    let aspect = (height as f32) / (width as f32);

    let view = cgmath::Matrix4::one();
    let projection = cgmath::ortho(-20.0, 100.0, -20.0 * aspect, 100.0 * aspect, -5.0, 5.0);

    let mut step = false;
    'main: loop {
        step = false;
        for event in window.poll_events() {
            match event {
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::A)) => { step = true; }
                glutin::Event::KeyboardInput(glutin::ElementState::Released, _, Some(glutin::VirtualKeyCode::A)) => { step = false; }
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
                glutin::Event::Closed => break 'main,
                _ => {},
            }
        }


        let locals = Locals {
            view: view.into(),
            proj: projection.into(),
            particle_size: 0.7,
        };
        encoder.update_constant_buffer(&data.locals, &locals);

        if true {

        // PBF simulation step
        particles
            // Reset acceleration
            .run(sph::reset_acceleration::<f32, U2>)
            // Apply external forces
            .run(|p| {
                let mut accel = p.write_property::<sph::property::Acceleration<f32, U2>>();
                accel.par_iter_mut().for_each(|mut accel| {
                    accel[1] += -10.0; // gravity
                });

                // println!("accel: {:?}", accel);
            })
            .run1(pbd::apply_forces, timestep)
            .run1(pbd::predict_position, timestep)
            // Boundary checks
            .run(|p| {
                let mut position = p.write_property::<pbd::property::PredPosition<f32, U2>>();
                position.par_iter_mut()
                    .for_each(|mut pos| {
                       if pos[1] < 0.0 { pos[1] = 0.0; }
                       if pos[1] > border * smoothing { pos[1] = border * smoothing; }
                       if pos[0] < 0.0 { pos[0] = 0.0; }
                       if pos[0] > border * smoothing { pos[0] = border * smoothing; }
                    });

                // println!("post pred pos: {:?}", p.write_property::<pbd::property::PredPosition<f32, U2>>());
            })
            // Neighbor search
            .run(|p| {
                let mut position = p.write_property::<sph::property::Position<f32, U2>>();
                let mut pred_position = p.write_property::<pbd::property::PredPosition<f32, U2>>();
                let mut velocity = p.write_property::<sph::property::Velocity<f32, U2>>();
                let mut vel_pos = Vec::new();
                for i in 0..position.len() {
                    vel_pos.push((position[i], pred_position[i], velocity[i]));
                }
                vel_pos.sort_by_key(|&(_, ref pred_pos, _)| grid.get_key(pred_pos));
                for i in 0..position.len() {
                    position[i] = vel_pos[i].0;
                    pred_position[i] = vel_pos[i].1;
                    velocity[i] = vel_pos[i].2;
                }
                grid.construct_ranges(pred_position);
            });

        for _ in 0..4 {
            particles
                .run1(pbd::calculate_lambda, (rest_density, smoothing, 0.0001, &grid))
                .run1(pbd::calculate_pos_delta, (rest_density, smoothing, &grid))
                .run(pbd::apply_delta::<f32>)
                .run(|p| {
                    let mut position = p.write_property::<pbd::property::PredPosition<f32, U2>>();
                    // println!("iter: pre pos: {:?}", position);
                    position.par_iter_mut()
                        .for_each(|mut pos| {
                           if pos[1] < 0.0 { pos[1] = 0.0; }
                           if pos[1] > border * smoothing { pos[1] = border * smoothing; }
                           if pos[0] < 0.0 { pos[0] = 0.0; }
                           if pos[0] > border * smoothing { pos[0] = border * smoothing; }
                        });

                    // println!("iter: post pos: {:?}", position);
                });

        }

        particles
            .run1(pbd::update_velocity, timestep)
            .run(pbd::update_position::<f32>);

        // println!("");
        }

        // Update particle vertex data
        particles.run(|p| {
            let mut vertex = p.write_property::<Vertex>();
            let position = p.read_property::<sph::property::Position<f32, U2>>();
            let pred_position = p.read_property::<pbd::property::PredPosition<f32, U2>>();

            for (mut v, pos) in
                    vertex.iter_mut()
                     .zip(position.iter())
            {
                v.pos[0] = pos[0]; v.pos[1] = pos[1];
            }

        });

        let slice = gfx::Slice {
            start: 0,
            end: particles.num_particles() as u32,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };

        encoder.update_buffer(&data.vbuf, particles.read_property::<Vertex>(), 0).unwrap();
        encoder.clear(&data.out_color, [0.5, 0.5, 0.5, 1.0]);
        encoder.draw(&slice, &pso, &data);
        encoder.flush(&mut device);
        window.swap_buffers().unwrap();
        device.cleanup();
    }
}

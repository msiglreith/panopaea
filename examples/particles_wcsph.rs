
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
    rayon::initialize(rayon::Configuration::new().num_threads(1));

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

    let stiffness = 3.0;
    let smoothing = 0.03;
    let rest_density = 1000.0;
    let timestep = 0.0005f32;
    let mass = 0.015f32;
    let viscosity = 3.5f32;
    let border = 40.0;
    let mut particles = Particles::new();
    sph::wcsph::init::<f32, U2>(&mut particles);
    particles.add_property::<Vertex>();

    let mut grid = sph::grid::BoundedGrid::new(math::vector_n::vec2(256, 256), smoothing);

    {
        let mut positions = Vec::new();
        let mut masses = Vec::new();
        for y in 0..30u8 {
            for x in 0..30u8 {
                let mut pos = sph::property::Position::new();
                pos[0] = (x as f32) * smoothing * 0.6;
                pos[1] = (y as f32) * smoothing * 0.6;
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
    let projection = cgmath::ortho(-1.0, 2.0, -1.0 * aspect, 2.0 * aspect, -5.0, 5.0);

    let mut iterations = 0;
    'main: loop {
        for event in window.poll_events() {
            match event {
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
                glutin::Event::Closed => break 'main,
                _ => {},
            }
        }


        iterations += 1;

        let locals = Locals {
            view: view.into(),
            proj: projection.into(),
            particle_size: 0.01,
        };
        encoder.update_constant_buffer(&data.locals, &locals);

        // SPH simulation step
        {
            particles
                // Neighbor search
                .run(|p| {
                    let mut position = p.write_property::<sph::property::Position<f32, U2>>();
                    let mut velocity = p.write_property::<sph::property::Velocity<f32, U2>>();
                    let mut vel_pos = Vec::new();
                    for i in 0..position.len() {
                        vel_pos.push((position[i], velocity[i]));
                    }
                    vel_pos.sort_by_key(|&(ref pos, _)| grid.get_key(pos));
                    for i in 0..position.len() {
                        position[i] = vel_pos[i].0; velocity[i] = vel_pos[i].1;
                    }
                    grid.construct_ranges(position);
                })
                // Reset acceleration
                .run(sph::reset_acceleration::<f32, U2>)
                // Apply external forces
                .run(|p| {
                    let mut accel = p.write_property::<sph::property::Acceleration<f32, U2>>();
                    accel.par_iter_mut().for_each(|mut accel| {
                        accel[1] += -10.0; // gravity
                    });
                })
                // Compute density
                .run1(sph::wcsph::compute_density, (smoothing, &grid))
                // Calculate pressure force
                .run1(sph::wcsph::calculate_pressure, (smoothing, stiffness, rest_density, &grid))
                // Calculate viscosity force
                .run1(sph::wcsph::calculate_viscosity, (smoothing, viscosity, &grid))
                // Integrate position and velocity (apply force)
                .run1(sph::wcsph::integrate_explicit_euler, timestep)
                // Boundary checks
                .run(|p| {
                    let mut position = p.write_property::<sph::property::Position<f32, U2>>();
                    let mut velocity = p.write_property::<sph::property::Velocity<f32, U2>>();
                    position.par_iter_mut()
                        .zip(velocity.par_iter_mut())
                        .for_each(|(mut pos, mut vel)| {
                           if pos[1] < 0.0 { pos[1] = 0.0; vel[1] = -vel[1] * 0.0; }
                           if pos[1] > border * smoothing { pos[1] = border * smoothing; vel[1] = -vel[1] * 0.0; }
                           if pos[0] < 0.0 { pos[0] = 0.0; vel[0] = -vel[0] * 0.0; }
                           if pos[0] > border * smoothing { pos[0] = border * smoothing; vel[0] = -vel[0] * 0.0; }
                        });
                });
        }

        // Update particle vertex data
        particles.run(|p| {
            let mut vertex = p.write_property::<Vertex>();
            let position = p.read_property::<sph::property::Position<f32, U2>>();
            let density = p.read_property::<sph::property::Density<f32>>();
            let accel = p.read_property::<sph::property::Acceleration<f32, U2>>();

            println!("{:?}",density[2]);
            println!("{:?}",accel[2]);

            for ((mut v, pos), &accel) in
                    vertex.iter_mut()
                     .zip(position.iter())
                     .zip(density.iter())
            {
                v.pos[0] = pos[0]; v.pos[1] = pos[1];
                v.color[0] = 0.2 * accel / rest_density; v.color[1] = 0.0; v.color[2] = 0.0;
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

        println!("-----------");
    }
}

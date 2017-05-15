
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
use panopaea::particle::Particles;

use cgmath::Transform;
use generic_array::typenum::U2;

pub type ColorFormat = gfx::format::Srgba8;
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

impl Default for Vertex {
    fn default() -> Vertex {
        Vertex {
            pos: [0.0, 0.0],
            color: [1.0, 1.0, 1.0],
        }
    }
}

fn main() {
    rayon::initialize(rayon::Configuration::new().num_threads(1));

    let builder = glutin::WindowBuilder::new()
        .with_dimensions(1440, 900)
        .with_vsync();
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
    let mut particles = Particles::new();
    sph::wcsph::init::<f32, U2>(&mut particles);
    particles.add_property::<Vertex>();

    let mut grid = sph::grid::BoundedGrid::new(math::vector_n::vec2(64, 64), smoothing);

    let mut positions = Vec::new();
    let mut masses = Vec::new();
    for y in 0..8u8 {
        for x in 0..16u8 {
            let mut pos = sph::property::Position::default();
            ((pos.0).0)[0] = (x as f32) * smoothing * 0.6;
            ((pos.0).0)[1] = (y as f32) * smoothing * 0.6;
            positions.push(pos);
            masses.push(sph::property::Mass(1.0f32));
        }
    }

    particles.add_particles(positions.len())
             .with::<sph::property::Position<f32, U2>>(&positions)
             .with(&masses);

    let (width, height) = window.get_inner_size_points().unwrap();
    let aspect = (height as f32) / (width as f32);

    let view = cgmath::Matrix4::one();
    let projection = cgmath::ortho(-100.0, 100.0, 100.0 * aspect, -100.0 * aspect, -5.0, 5.0);

    'main: loop {
        for event in window.poll_events() {
            match event {
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
                glutin::Event::Closed => break 'main,
                _ => {},
            }
        }

        let locals = Locals {
            view: view.into(),
            proj: projection.into(),
            particle_size: 1.0,
        };
        encoder.update_constant_buffer(&data.locals, &locals);

        particles.run(|p| {
            let mut position = p.write_property::<sph::property::Position<f32, U2>>().unwrap();
            position.sort_by_key(|pos| grid.get_key(pos));
            grid.construct_ranges(position);
        });

        sph::wcsph::compute_density(smoothing, &grid, &mut particles);

        particles.run(|p| {
            let mut vertex = p.write_property::<Vertex>().unwrap();
            let position = p.read_property::<sph::property::Position<f32, U2>>().unwrap();
            let density = p.read_property::<sph::property::Density<f32>>().unwrap();

            for ((mut v, pos), &d) in
                    vertex.iter_mut()
                     .zip(position.iter())
                     .zip(density.iter())
            {
                v.pos[0] = pos[0]; v.pos[1] = pos[1];
                v.color[0] = d.0 / 4.0; v.color[1] = 0.0; v.color[2] = 0.0;
            }
        });

        let slice = gfx::Slice {
            start: 0,
            end: particles.num_particles() as u32,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };

        encoder.update_buffer(&data.vbuf, particles.read_property::<Vertex>().unwrap(), 0).unwrap();
        encoder.clear(&data.out_color, [0.2, 0.2, 0.2, 1.0]);
        encoder.draw(&slice, &pso, &data);
        encoder.flush(&mut device);
        window.swap_buffers().unwrap();
        device.cleanup();
    }
}

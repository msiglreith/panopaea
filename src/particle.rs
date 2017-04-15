
//! Particle system

use std::collections::HashMap;
use std::default::Default;
use std::any::{TypeId};
use mopa;

pub struct Particles {
    num_particles: usize,
    properties: HashMap<TypeId, Box<Storage>>,
}

impl Particles {
    pub fn new() -> Self {
        Particles {
            num_particles: 0,
            properties: HashMap::new(),
        }
    }

    pub fn add_property<T>(&mut self)
        where T: Clone + Default + 'static
    {
        let type_id = TypeId::of::<T>();
        let num_particles = self.num_particles;

        self.properties.entry(type_id).or_insert_with(|| {
            Box::new(vec![T::default(); num_particles])
        });
    }

    pub fn read_property<T>(&self) -> Option<&[T]>
        where T: Clone + Default + 'static {
        unsafe { self.get_property().map(|property| property.as_slice()) }
    }

    pub fn write_property<T>(&mut self) -> Option<&mut [T]>
        where T: Clone + Default + 'static {
        unsafe { self.get_property_mut().map(|property| property.as_mut_slice()) }
    }

    unsafe fn get_property<T>(&self) -> Option<&Vec<T>>
        where T: Clone + Default + 'static
    {
        let type_id = TypeId::of::<T>();
        self.properties.get(&type_id)
                       .and_then(|property| property.downcast_ref())
    }

    unsafe fn get_property_mut<T>(&self) -> Option<&mut Vec<T>>
        where T: Clone + Default + 'static
    {
        let type_id = TypeId::of::<T>();
        self.properties.get(&type_id)
                       .and_then(|property| (*(property as *const _ as *mut Box<Storage>)).downcast_mut()) // TODO: something safer would be appreciated
    }

    pub fn reserve(&mut self, additional: usize) {
        for (_, property) in &mut self.properties {
            property.reserve(additional);
        }
    }

    pub fn add_particles(&mut self, additional: usize) -> Builder {
        self.reserve(additional);
        self.num_particles += additional;
        Builder(self)
    }

    pub fn run<'a, F>(&'a mut self, func: F)
        where F: FnOnce(Processor<'a>) {
        func(Processor(self));
    }

    pub fn num_particles(&self) -> usize {
        self.num_particles
    }
}

pub struct Builder<'a>(&'a mut Particles);

impl<'a> Builder<'a> {
    pub fn with<T>(&mut self, values: &[T]) -> &mut Self
        where T: Clone + Default + 'static
    {
        let num_particles = self.0.num_particles;
        if let Some(mut storage) = unsafe { self.0.get_property_mut::<T>() } {
            debug_assert_eq!(values.len(), num_particles - storage.len());
            storage.extend_from_slice(values);
        }

        self
    }
}

impl<'a> Drop for Builder<'a> {
    fn drop(&mut self) {
        // fill remaining properties with default values
        let num_particles = self.0.num_particles;
        for (_, property) in &mut self.0.properties {
            let remaining_particles = num_particles - property.len();
            if remaining_particles > 0 {
                property.fill(remaining_particles);
            }
        }
    }
}

pub struct Processor<'a>(&'a mut Particles);
impl<'a> Processor<'a> {
    pub fn read_property<T>(&self) -> Option<&[T]>
        where T: Clone + Default + 'static {
        unsafe { self.0.get_property().map(|property| property.as_slice()) }
    }

    pub fn write_property<T>(&self) -> Option<&mut [T]>
        where T: Clone + Default + 'static {
        unsafe { self.0.get_property_mut().map(|property| property.as_mut_slice()) }
    }
}


pub trait Storage : mopa::Any {
    fn len(&self) -> usize;
    fn reserve(&mut self, additional: usize);
    fn fill(&mut self, additional: usize);
}

mopafy!(Storage);

impl<T: Clone + Default + 'static> Storage for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional);
    }

    fn fill(&mut self, additional: usize) {
        self.extend_from_slice(&vec![T::default(); additional])
    }
}

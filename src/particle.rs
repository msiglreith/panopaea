
//! Particle system

use std::collections::HashMap;
use std::default::Default;
use std::marker::PhantomData;
use std::any::{TypeId};
use mopa;

pub trait Property: 'static {
    type Subtype: Clone;
    fn new() -> Self::Subtype;
}

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

    pub fn add_property<T: Property>(&mut self) {
        let type_id = TypeId::of::<T>();
        let num_particles = self.num_particles;

        self.properties.entry(type_id).or_insert_with(|| {
            Box::new((vec![T::new(); num_particles], PhantomData::<T>))
        });
    }

    pub fn read_property<T: Property>(&self) -> Option<&[T::Subtype]> {
        unsafe { self.get_property::<T>().map(|property| property.as_slice()) }
    }

    pub fn write_property<T: Property>(&mut self) -> Option<&mut [T::Subtype]> {
        unsafe { self.get_property_mut::<T>().map(|property| property.as_mut_slice()) }
    }

    unsafe fn get_property<T: Property>(&self) -> Option<&Vec<T::Subtype>> {
        let type_id = TypeId::of::<T>();
        self.properties.get(&type_id)
                       .and_then(|property| property.downcast_ref::<VecStorage<T>>())
                       .map(|&(ref vec, _)| vec)
    }

    unsafe fn get_property_mut<T: Property>(&self) -> Option<&mut Vec<T::Subtype>> {
        let type_id = TypeId::of::<T>();
        self.properties.get(&type_id)
                       .and_then(|property| (*(property as *const _ as *mut Box<Storage>)).downcast_mut::<VecStorage<T>>()) // TODO: something safer would be appreciated
                       .map(|&mut (ref mut vec, _)| vec)
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
    pub fn with<T: Property>(&mut self, values: &[T::Subtype]) -> &mut Self {
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
    pub fn read_property<T: Property>(&self) -> Option<&[T::Subtype]> {
        unsafe { self.0.get_property::<T>().map(|property| property.as_slice()) }
    }

    pub fn write_property<T: Property>(&self) -> Option<&mut [T::Subtype]> {
        unsafe { self.0.get_property_mut::<T>().map(|property| property.as_mut_slice()) }
    }
}


pub trait Storage : mopa::Any {
    fn len(&self) -> usize;
    fn reserve(&mut self, additional: usize);
    fn fill(&mut self, additional: usize);
}

mopafy!(Storage);

type VecStorage<T: Property> = (Vec<T::Subtype>, PhantomData<T>);

impl<T: Property> Storage for VecStorage<T> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    fn fill(&mut self, additional: usize) {
        self.0.extend_from_slice(&vec![T::new(); additional])
    }
}

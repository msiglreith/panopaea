
use particle::{Particles, Property};

type Edges = Particles;
type Faces = Particles;
type HalfEdges = Particles;
type Vertices = Particles;

struct EdgeId(usize);
struct FaceId(usize);
struct HalfEdgeId(usize);
struct VertexId(usize);

struct HalfEdge {
    /// Target vertex
    vertex: VertexId,

    /// Adjacent face
    face: FaceId,
}

struct Vertex {

}

pub struct Mesh {
    edges: Edges,
    faces: Faces,
    half_edges: HalfEdges,
    vertices: Vertices,
}

impl Mesh {
    pub fn new() -> Self {
        let edges = Edges::new();
        let faces = Faces::new();
        let half_edges = HalfEdges::new();
        let vertices = Vertices::new();

        Mesh {
            edges,
            faces,
            half_edges,
            vertices,
        }
    }

    pub fn add_edge_property<T: Property>(&mut self) {
        self.edges.add_property::<T>()
    }

    pub fn add_face_property<T: Property>(&mut self) {
        self.faces.add_property::<T>()
    }

    pub fn add_half_edge_property<T: Property>(&mut self) {
        self.half_edges.add_property::<T>()
    }

    pub fn add_vertex_property<T: Property>(&mut self) {
        self.vertices.add_property::<T>()
    }

    pub fn read_edge_property<T: Property>(&self) -> &[T::Subtype] {
        self.edges.read_property::<T>()
    }

    pub fn read_face_property<T: Property>(&self) -> &[T::Subtype] {
        self.faces.read_property::<T>()
    }

    pub fn read_half_edge_property<T: Property>(&self) -> &[T::Subtype] {
        self.half_edges.read_property::<T>()
    }

    pub fn read_vertex_property<T: Property>(&self) -> &[T::Subtype] {
        self.vertices.read_property::<T>()
    }

    pub fn write_edge_property<T: Property>(&mut self) -> &mut [T::Subtype] {
        self.edges.write_property::<T>()
    }

    pub fn write_face_property<T: Property>(&mut self) -> &mut [T::Subtype] {
        self.faces.write_property::<T>()
    }

    pub fn write_half_edge_property<T: Property>(&mut self) -> &mut [T::Subtype] {
        self.half_edges.write_property::<T>()
    }

    pub fn write_vertex_property<T: Property>(&mut self) -> &mut [T::Subtype] {
        self.vertices.write_property::<T>()
    }
}


#version 150

in vec2 a_Pos;
in vec3 a_Color;

uniform Locals {
    mat4 u_View;
    mat4 u_Proj;
    float u_ParticleSize;
};

out Vertex {
    vec3 color;
} v_Vertex;

void main() {
    gl_Position = u_View * vec4(a_Pos, 0.0, 1.0);
    v_Vertex.color = a_Color;
}

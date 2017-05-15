
#version 150

layout (points) in;
layout (triangle_strip, max_vertices=4) out;

uniform Locals {
    mat4 u_View;
    mat4 u_Proj;
    float u_ParticleSize;
};

in Vertex {
    vec3 color;
} v_Vertex[];

out Vertex {
    vec3 color;
    vec2 uv;
} g_Vertex;

void main()
{
    gl_Position = u_Proj * (gl_in[0].gl_Position + vec4(-u_ParticleSize, -u_ParticleSize, 0, 0));
    g_Vertex.color = v_Vertex[0].color;
    g_Vertex.uv = vec2(-1, -1);
    EmitVertex();

    gl_Position = u_Proj * (gl_in[0].gl_Position + vec4(u_ParticleSize, -u_ParticleSize, 0, 0));
    g_Vertex.color = v_Vertex[0].color;
    g_Vertex.uv = vec2(1, -1);
    EmitVertex();

    gl_Position = u_Proj * (gl_in[0].gl_Position + vec4(-u_ParticleSize, u_ParticleSize, 0, 0));
    g_Vertex.color = v_Vertex[0].color;
    g_Vertex.uv = vec2(-1, 1);
    EmitVertex();

    gl_Position = u_Proj * (gl_in[0].gl_Position + vec4(u_ParticleSize, u_ParticleSize, 0, 0));
    g_Vertex.color = v_Vertex[0].color;
    g_Vertex.uv = vec2(1, 1);
    EmitVertex();
}
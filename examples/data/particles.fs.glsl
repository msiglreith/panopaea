
#version 150

in Vertex {
    vec3 color;
    vec2 uv;
} ps_vertex;

out vec4 Target0;

void main() {
    float alpha = step(0.0, 1.0-dot(ps_vertex.uv, ps_vertex.uv));
    Target0 = vec4(ps_vertex.color, alpha);
}

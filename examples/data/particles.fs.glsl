
#version 150

in Vertex {
    vec3 color;
    vec2 uv;
} g_Vertex;

out vec4 Target0;

void main() {
    float alpha = step(0, 1-dot(g_Vertex.uv, g_Vertex.uv));
    Target0 = vec4(g_Vertex.color, alpha);
}

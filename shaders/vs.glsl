 #version 460

    layout(location = 0) in vec3 position;
    layout(location = 1) in vec4 color;

    layout(location = 0) out vec4 fragColor;

        void main() {
            gl_Position = vec4(position.x /2, position.y/2, position.z/2, 1.0);
            fragColor = color;
        }
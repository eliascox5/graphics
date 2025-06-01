//Function to load and compile GLSL using the shader macros.
use std::env;
use std::fs;
use std::io;
use std::path;
use std::sync::Arc;
use std::vec;
use vulkano::device::Device;
use vulkano::shader::ShaderModule;
use vulkano_shaders;

enum ShaderTypes{
    Vertex,
    Fragment,
}

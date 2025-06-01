use std::{collections::HashMap, io::Read, string, usize};
use byte::{ctx::Endian, *};
use serde::Deserialize;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use serde_json::{error, Result, Value};


pub fn read_buffer_view<T: for<'a> TryRead<'a, Endian>>(data: &Vec<u8>, bv: BufferAccessor) -> Vec<T>{
    //Based on the gltf format ive come up with this 
    //Since the .bin is split into sections, we read just those sections
    //And split it into the size that hte literal takes up
    //for example signed floating point is 4 bytes long
    let byte_size = bv.bit_size / 8;
    let buffer_contents =  data
    .split_at(bv.byteOffset)
    .1
    .split_at(bv.byteLength)
    .0.to_vec();

    let mut value = buffer_contents.chunks( byte_size);
    let final_data: Vec<T> = value.map(|a| a.read_with::<T>(&mut 0, LE).unwrap()).collect();
    return final_data
}


pub struct BufferAccessor{
    pub bit_size: usize,
    pub buffer: usize,
    pub byteLength: usize,
    pub byteOffset: usize,
    pub component_type: usize,
}

#[derive(BufferContents, Vertex, Debug, Clone)]
#[repr(C)]
pub struct Vec4 {
    #[format(R16G16B16A16_SFLOAT)]
    pub position: [f32; 4],
}

#[derive(BufferContents, Vertex, Debug, Clone)]
#[repr(C)]
pub struct three_d {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}
#[derive(BufferContents, Vertex)]
#[repr(C)]

pub struct Vec2 {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}

pub fn serialise_gltf(json: &str){

}  
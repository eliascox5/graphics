use std::{collections::HashMap, io::Read, string, usize};
use byte::*;
use serde::Deserialize;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use serde_json::{error, Result, Value};

pub fn read_buffer_view(data: &Vec<u8>, bv: BufferAccessor){
    //Based on the gltf format ive come up with this 
    //Since the .bin is split into sections, we read just those sections
    //And split it into the size that hte literal takes up
    //for example signed floating point is 4 bytes long
    let byte_size = bv.bit_size / 8;
    let buffer_contents = data
    .split_at(bv.byteOffset)
    .1
    .split_at(bv.byteLength + bv.byteOffset)
    .0.to_vec();

    let mut value = data.chunks( byte_size);
    let floatsi: Vec<f32> = value.map(|a| a.read_with::<f32>(&mut 0, LE).unwrap()).collect();

   let mut vectors: Vec<Vec3> = vec![];

    let v3 = floatsi.chunks(3);
    for v in v3{
        let vs = [v[0], v[1], v[2]];
        vectors.push(Vec3 { position: vs});
    }
    
    println!("The first vectors are {:?}", vectors);
}

pub fn read_vec_3s(vc: &Vec<u8>){
    
}

pub struct BufferAccessor{
    pub bit_size: usize,
    pub buffer: usize,
    pub byteLength: usize,
    pub byteOffset: usize,
    pub target: usize,
}

#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub struct Vec3 {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 3],
}
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Vec2 {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}

pub fn serialise_gltf(json: &str){

}  
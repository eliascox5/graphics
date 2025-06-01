pub struct Settings{
    pub logging: bool,


}
impl Settings{
    pub fn default() -> Settings{
        Settings{logging: false}
    }
}
from glumpy import app, gloo, gl
import os

scalar = 10000

window = app.Window()
vertex = """
uniform int total_frames;
uniform int frame_size; // number of verts in a frame
uniform int index_start; // start index for range
attribute vec3 position;
    void main()
    {
        float index = position.z;
        if(index_start + frame_size < frame_size * total_frames){
            if(index < index_start + frame_size && index >= index_start)
            gl_Position = vec4(position.x, position.y, 0.0, 1.0);
        }
        else{
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
        }
    }
"""
fragment = """
           uniform vec4 color;
           void main() {
               gl_FragColor = color;
           } """


data_files = []

print("Loading body frame data")
if not os.path.isdir("./frames"):
    print("No frame data files detected")
    print("Run the simulation with '1' as the 4th argument to save the frame data")
    exit(0)

path, dirs, files = os.walk("./frames").__next__()
num_frames = len(files)

for i in range(num_frames):
    with open("./frames/" + files[i]) as f:
        content = [x.strip().split(" ") for x in f.readlines()]
        data_files.append(content)
print("Loaded", len(data_files), "frames(", len(data_files[0]) * len(data_files), "vertices)")
print("Frame data loaded, converting to coordinate pairs")
num_bodies = len(data_files[0])
simulation_verts = []
count = 0
for f in data_files:
    for line in f:
        x = float(line[1]) / scalar
        y = float(line[2]) / scalar
        z = count
        count += 1
        simulation_verts.append((x, y, z))
print("Data conversion complete")

quad = gloo.Program(vertex, fragment, count=num_bodies * num_frames)

quad['position'] = simulation_verts
quad["total_frames"] = num_frames
quad["frame_size"] = num_bodies
quad["index_start"] = 0
quad['color'] = 1,0,0,1  # red

wait_index = 0
wait_until = 15

@window.event
def on_draw(dt):
    global wait_index, wait_until
    window.clear()
    quad.draw(gl.GL_POINTS)
    wait_index += 1
    if wait_index >= wait_until:
        wait_index = 0
        quad["index_start"] += num_bodies

app.run()
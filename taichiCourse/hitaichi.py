import taichi as ti

ti.init(ti.cpu)

paused=ti.field(ti.i32,())

G=100
PI=3.141592653

N=1000

m=59

galaxy_size=0.4

planet_radius=3

init_vel=1000

h=1e-6

substepping=10


pos=ti.Vector.field(2,ti.f32,N)
vel=ti.Vector.field(2,ti.f32,N)
force=ti.Vector.field(2,ti.f32,N)

@ti.kernel
def initialize():
    center=ti.Vector([0.5,0.5])
    for i in range(N):
        theta=ti.random()*2*PI
        r=(ti.sqrt(ti.random())*0.7+0.3)*galaxy_size
        offset=r*ti.Vector([ti.cos(theta),ti.sin(theta)])
        pos[i]=center+offset
        vel[i]=[-offset.y,offset.x]
        vel[i]*=init_vel


@ti.kernel
def compute_force():
    for i in range(N):
        force[i]=ti.Vector([0.0,0.0])
    
    for i in range(N):
        p=pos[i]
        for j in range(i):
            diff=p-pos[j]
            r=diff.norm(1e-5)

            f=-G*m*m*(1.0/r)**3*diff
            force[i]+=f
            force[j]+=-f



@ti.kernel
def update():
    dt=h/substepping
    for i in range(N):
        vel[i]+=dt*force[i]/m
        pos[i]+=dt*vel[i]

gui=ti.GUI("N-body problem",(512,512))

initialize()
while gui.running:
    for i in range(substepping):
        compute_force()
        update()
    
    gui.clear(0x112f41)
    gui.circles(pos.to_numpy(),color=0xffffff,radius=planet_radius)
    gui.show()


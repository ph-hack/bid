"""
This module contains all specifications used to create the images with the
background and foreground images taken with the DSLR camera.
"""


from data_creation import SpecList, ForeSpecs, Spec, generate_dataset

specs = SpecList()

# comb 1/54, blur=none, noise=1 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[], depths=[]),
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[3])
    ],
    blur_method='none',
    blur_level=2,
    noise_level=1,
    compression=0,
    scale=0.8
))

# comb 2/54, blur=none, noise=1 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[], depths=[]),
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[-2], depths=[0])
    ],
    blur_method='none',
    blur_level=2,
    noise_level=1,
    compression=5,
    scale=0.65
))

# comb 3/54, blur=none, noise=1 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[], depths=[]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2,1], depths=[3,0])
    ],
    blur_method='none',
    blur_level=2,
    noise_level=1,
    compression=20,
    scale=0.5
))

# comb 4/54, blur=none, noise=2 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[], depths=[]),
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[1], depths=[1])
    ],
    blur_method='none',
    blur_level=2,
    noise_level=2,
    compression=0,
    scale=0.8
))

# comb 5/54, blur=none, noise=2 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[], depths=[]),
        ForeSpecs(positions=[-1], depths=[2]),
        ForeSpecs(positions=[1], depths=[5])
    ],
    blur_method='none',
    blur_level=2,
    noise_level=2,
    compression=5,
    scale=0.65
))

# comb 6/54, blur=none, noise=2 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[], depths=[]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[1, -2], depths=[3,0])
    ],
    blur_method='none',
    blur_level=2,
    noise_level=2,
    compression=20,
    scale=0.5
))

# comb 7/54, blur=none, noise=3 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[], depths=[]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[0], depths=[4]),
    ],
    blur_method='none',
    blur_level=2,
    noise_level=3,
    compression=0,
    scale=0.8
))

# comb 8/54, blur=none, noise=3 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[], depths=[]),
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='none',
    blur_level=2,
    noise_level=3,
    compression=5,
    scale=0.65
))

# comb 9/54, blur=none, noise=3 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[], depths=[]),
        ForeSpecs(positions=[0], depths=[4]),
        ForeSpecs(positions=[-2,1], depths=[3,0]),
    ],
    blur_method='none',
    blur_level=2,
    noise_level=3,
    compression=20,
    scale=0.5
))

# comb 10/54, blur=macro, noise=1 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2], depths=[0]),
    ],
    blur_method='macro',
    blur_level=4,
    noise_level=1,
    compression=0,
    scale=0.8
))

# comb 11/54, blur=macro, noise=1 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[0], depths=[4]),
    ],
    blur_method='macro',
    blur_level=6,
    noise_level=1,
    compression=5,
    scale=0.65
))

# comb 12/54, blur=macro, noise=1 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[-1], depths=[2]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='macro',
    blur_level=8,
    noise_level=1,
    compression=20,
    scale=0.5
))

# comb 13/54, blur=macro, noise=2 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[4]),
        ForeSpecs(positions=[-2,1], depths=[3,0]),
        ForeSpecs(positions=[-1], depths=[2]),
    ],
    blur_method='macro',
    blur_level=2,
    noise_level=2,
    compression=0,
    scale=0.8
))

# comb 14/54, blur=macro, noise=2 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='macro',
    blur_level=5,
    noise_level=2,
    compression=5,
    scale=0.65
))

# comb 15/54, blur=macro, noise=2 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[1, -2], depths=[3,0]),
    ],
    blur_method='macro',
    blur_level=10,
    noise_level=2,
    compression=20,
    scale=0.5
))

# comb 16/54, blur=macro, noise=3 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2], depths=[0]),
    ],
    blur_method='macro',
    blur_level=4,
    noise_level=3,
    compression=0,
    scale=0.8
))

# comb 17/54, blur=macro, noise=3 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[2], depths=[0]),
        ForeSpecs(positions=[0], depths=[4]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='macro',
    blur_level=7,
    noise_level=3,
    compression=5,
    scale=0.65
))

# comb 18/54, blur=macro, noise=3 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[1]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2,1], depths=[5,0]),
    ],
    blur_method='macro',
    blur_level=9,
    noise_level=3,
    compression=20,
    scale=0.5
))

# comb 19/54, blur=hyperfocal, noise=1 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2], depths=[0]),
    ],
    blur_method='hyperfocal',
    blur_level=3,
    noise_level=1,
    compression=0,
    scale=0.8
))

# comb 20/54, blur=hyperfocal, noise=1 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[2], depths=[1]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='hyperfocal',
    blur_level=8,
    noise_level=1,
    compression=5,
    scale=0.65
))

# comb 21/54, blur=hyperfocal, noise=1 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2,1], depths=[3,0]),
        ForeSpecs(positions=[-1], depths=[2]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='hyperfocal',
    blur_level=6,
    noise_level=1,
    compression=20,
    scale=0.5
))

# comb 22/54, blur=hyperfocal, noise=2 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[-1,2], depths=[2,0]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='hyperfocal',
    blur_level=2,
    noise_level=2,
    compression=0,
    scale=0.8
))

# comb 23/54, blur=hyperfocal, noise=2 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[0], depths=[4]),
        ForeSpecs(positions=[-1], depths=[5]),
    ],
    blur_method='hyperfocal',
    blur_level=4,
    noise_level=2,
    compression=5,
    scale=0.65
))

# comb 24/54, blur=hyperfocal, noise=2 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[2], depths=[0]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='hyperfocal',
    blur_level=7,
    noise_level=2,
    compression=20,
    scale=0.5
))

# comb 25/54, blur=hyperfocal, noise=3 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[-1], depths=[2]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='hyperfocal',
    blur_level=2,
    noise_level=3,
    compression=0,
    scale=0.8
))

# comb 26/54, blur=hyperfocal, noise=3 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[-2,1], depths=[3,0]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='hyperfocal',
    blur_level=6,
    noise_level=3,
    compression=5,
    scale=0.65
))

# comb 27/54, blur=hyperfocal, noise=3 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2], depths=[0]),
    ],
    blur_method='hyperfocal',
    blur_level=9,
    noise_level=3,
    compression=20,
    scale=0.5
))

# comb 28/54, blur=whole, noise=1 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[0], depths=[4]),
    ],
    blur_method='whole',
    blur_level=5,
    noise_level=1,
    compression=0,
    scale=0.8
))

# comb 29/54, blur=whole, noise=1 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[2], depths=[1]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='whole',
    blur_level=7,
    noise_level=1,
    compression=5,
    scale=0.65
))

# comb 30/54, blur=whole, noise=1 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2,1], depths=[3,0]),
        ForeSpecs(positions=[-1], depths=[2]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='whole',
    blur_level=9,
    noise_level=1,
    compression=20,
    scale=0.5
))

# comb 31/54, blur=whole, noise=2 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[2], depths=[3]),
        ForeSpecs(positions=[0], depths=[4]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='whole',
    blur_level=8,
    noise_level=2,
    compression=0,
    scale=0.8
))

# comb 32/54, blur=whole, noise=2 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-1], depths=[5]),
    ],
    blur_method='whole',
    blur_level=5,
    noise_level=2,
    compression=5,
    scale=0.65
))

# comb 33/54, blur=whole, noise=2 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[2]),
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='whole',
    blur_level=2,
    noise_level=2,
    compression=20,
    scale=0.5
))

# comb 34/54, blur=whole, noise=3 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[2,-1], depths=[3,0]),
        ForeSpecs(positions=[-1], depths=[2]),
    ],
    blur_method='whole',
    blur_level=6,
    noise_level=3,
    compression=0,
    scale=0.8
))

# comb 35/54, blur=whole, noise=3 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2,1], depths=[3,0]),
        ForeSpecs(positions=[-1], depths=[2]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    blur_method='whole',
    blur_level=8,
    noise_level=3,
    compression=5,
    scale=0.65
))

# comb 36/54, blur=whole, noise=3 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[2], depths=[3]),
    ],
    blur_method='whole',
    blur_level=3,
    noise_level=3,
    compression=20,
    scale=0.5
))

# comb 37/54, blur=object_motion, noise=1 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[-1], depths=[2]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    motion_method='object_motion',
    motion_level=7,
    motion_angle=45,
    noise_level=1,
    compression=0,
    scale=0.8
))

# comb 38/54, blur=object_motion, noise=1 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[3]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2], depths=[0]),
    ],
    motion_method='object_motion',
    motion_level=7,
    motion_angle=85,
    noise_level=1,
    compression=5,
    scale=0.65
))

# comb 39/54, blur=object_motion, noise=1 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[-1], depths=[5]),
    ],
    motion_method='object_motion',
    motion_level=8,
    motion_angle=60,
    noise_level=1,
    compression=20,
    scale=0.5
))

# comb 40/54, blur=object_motion, noise=2 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[2], depths=[3]),
        ForeSpecs(positions=[0], depths=[4]),
    ],
    motion_method='object_motion',
    motion_level=10,
    motion_angle=90,
    noise_level=2,
    compression=0,
    scale=0.8
))

# comb 41/54, blur=object_motion, noise=2 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[0], depths=[4]),
    ],
    motion_method='object_motion',
    motion_level=9,
    motion_angle=15,
    noise_level=2,
    compression=5,
    scale=0.65
))

# comb 42/54, blur=object_motion, noise=2 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2], depths=[0]),
    ],
    motion_method='object_motion',
    motion_level=10,
    motion_angle=45,
    noise_level=2,
    compression=20,
    scale=0.5
))

# comb 43/54, blur=object_motion, noise=3 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[2], depths=[3]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    motion_method='object_motion',
    motion_level=8,
    motion_angle=90,
    noise_level=3,
    compression=0,
    scale=0.8
))

# comb 44/54, blur=object_motion, noise=3 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[4]),
        ForeSpecs(positions=[-1], depths=[2]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    motion_method='object_motion',
    motion_level=9,
    motion_angle=10,
    noise_level=3,
    compression=5,
    scale=0.65
))

# comb 45/54, blur=object_motion, noise=3 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[-2], depths=[0]),
        ForeSpecs(positions=[2], depths=[3]),
    ],
    motion_method='object_motion',
    motion_level=10,
    motion_angle=90,
    noise_level=3,
    compression=20,
    scale=0.5
))

# comb 46/54, blur=camera_motion, noise=1 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2,1], depths=[3,0]),
    ],
    motion_method='camera_motion',
    motion_level=7,
    motion_angle=45,
    noise_level=1,
    compression=0,
    scale=0.8
))

# comb 47/54, blur=camera_motion, noise=1 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[4]),
        ForeSpecs(positions=[-1,2], depths=[4,0]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    motion_method='camera_motion',
    motion_level=8,
    motion_angle=80,
    noise_level=1,
    compression=5,
    scale=0.65
))

# comb 48/54, blur=camera_motion, noise=1 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[2], depths=[3]),
        ForeSpecs(positions=[0], depths=[4]),
    ],
    motion_method='camera_motion',
    motion_level=10,
    motion_angle=90,
    noise_level=1,
    compression=20,
    scale=0.5
))

# comb 49/54, blur=camera_motion, noise=2 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[0], depths=[4]),
        ForeSpecs(positions=[1,-2], depths=[3,0]),
    ],
    motion_method='camera_motion',
    motion_level=10,
    motion_angle=90,
    noise_level=2,
    compression=0,
    scale=0.8
))

# comb 50/54, blur=camera_motion, noise=2 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[1], depths=[2]),
        ForeSpecs(positions=[-2], depths=[0]),
    ],
    motion_method='camera_motion',
    motion_level=8,
    motion_angle=15,
    noise_level=2,
    compression=5,
    scale=0.65
))

# comb 51/54, blur=camera_motion, noise=2 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[2], depths=[3]),
        ForeSpecs(positions=[0], depths=[4]),
        ForeSpecs(positions=[-2,1], depths=[3,0]),
    ],
    motion_method='camera_motion',
    motion_level=9,
    motion_angle=30,
    noise_level=2,
    compression=20,
    scale=0.5
))

# comb 52/54, blur=camera_motion, noise=3 , compression=1
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[4]),
        ForeSpecs(positions=[-1], depths=[2]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    motion_method='camera_motion',
    motion_level=10,
    motion_angle=85,
    noise_level=3,
    compression=0,
    scale=0.8
))

# comb 53/54, blur=camera_motion, noise=3 , compression=2
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[2], depths=[3]),
        ForeSpecs(positions=[0], depths=[4]),
    ],
    motion_method='camera_motion',
    motion_level=8,
    motion_angle=90,
    noise_level=3,
    compression=5,
    scale=0.65
))

# comb 54/54, blur=camera_motion, noise=3 , compression=3
specs.add(Spec(
    foreground_specs=[
        ForeSpecs(positions=[0], depths=[0]),
        ForeSpecs(positions=[-1], depths=[2]),
        ForeSpecs(positions=[1], depths=[5]),
    ],
    motion_method='camera_motion',
    motion_level=7,
    motion_angle=90,
    noise_level=3,
    compression=20,
    scale=0.5
))

print("Number of images for 21 backgrounds =", specs.n_combinations(21))

i = 0
generate_dataset('../test_set/background/', '../test_set/foreground/', '../test_set/images/', specs=specs, log_file='../testset_inventory_{}.csv'.format(i), log_init_rows=0, verbose=1)
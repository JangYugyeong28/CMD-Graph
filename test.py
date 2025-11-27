# ...existing code...
"""
Cleaned version of test.py:
- Removed large commented blocks and duplicate imports.
- Translated comments to English.
- Kept main functionality: load model, generate shapes, save programs/images, create variations.
- Minor variable name fixes for consistency (rhino_commands_path).
"""

import os
import argparse
import time
import numpy as np
import random
import h5py
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from skimage import measure

from dataset import ShapeNet3D
from model import BlockOuterNet
from criterion import BatchIoU
from misc import (
    decode_multiple_block,
    execute_shape_program_with_trace,
    execute_shape_program_yg,
)
from interpreter import Interpreter
from programs.loop_gen import translate, rotate, end
from convert_command import convert_file
from variation import modify_shape

def set_seed(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_argument():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test the program generator")
    parser.add_argument('--model', type=str, default='./model/ckpts_GA_chair/program_generator_GA_chair.t7',
                        help='path to the trained model checkpoint')
    parser.add_argument('--data', type=str, default='./data/chair_testing.h5',
                        help='path to the test data (.h5)')
    parser.add_argument('--save_path', type=str, default='./output/chair/',
                        help='root folder to save outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--info_interval', type=int, default=10, help='printing interval for batches')
    parser.add_argument('--save_prog', action='store_true', help='save generated programs as text files')
    parser.add_argument('--save_img', action='store_true', help='render reconstructed shapes to images')
    parser.add_argument('--num_render', type=int, default=15, help='maximum number of samples to render')

    opt = parser.parse_args()
    opt.prog_save_path = os.path.join(opt.save_path, 'programs')
    opt.imgs_save_path = os.path.join(opt.save_path, 'images')
    opt.seq_imgs_save_path = os.path.join(opt.save_path, 'sequence_images')
    opt.is_cuda = torch.cuda.is_available()
    return opt

def test_on_shapenet_data(epoch, test_loader, model, opt, gen_shape=False):
    """Run model on the dataloader and optionally decode programs/voxels."""
    model.eval()
    generated_shapes = []
    original_shapes = []
    gen_pgms = []
    gen_params = []

    for idx, data in enumerate(test_loader):
        start = time.time()

        shapes = data
        shapes = Variable(torch.unsqueeze(shapes, 1), requires_grad=False).cuda()
        out = model.decode(shapes)

        if opt.is_cuda:
            torch.cuda.synchronize()
        end = time.time()

        if gen_shape:
            for sample_idx in range(shapes.size(0)):
                generated_shape = decode_multiple_block(
                    out[0][sample_idx:sample_idx+1], out[1][sample_idx:sample_idx+1]
                )
                generated_shapes.append(generated_shape)
                original_shapes.append(data[sample_idx:sample_idx+1].clone().numpy())

                _, save_pgms = torch.max(out[0].data[sample_idx:sample_idx+1], dim=3)
                save_pgms = save_pgms.cpu().numpy()
                save_params = out[1].data[sample_idx:sample_idx+1].cpu().numpy()

                gen_pgms.append(save_pgms)
                gen_params.append(save_params)

                # Create per-sample sequence folder and save step images
                sample_save_path = os.path.join(opt.seq_imgs_save_path, f'sample_{sample_idx}')
                os.makedirs(sample_save_path, exist_ok=True)
                intermediate_shapes = execute_shape_program_with_trace(save_pgms[0], save_params[0], sample_idx, sample_save_path)
                visualize_sequence(intermediate_shapes, sample_save_path)

        if idx % opt.info_interval == 0:
            print(f"Test: epoch {epoch} batch {idx}/{len(test_loader)}, time={end - start:.3f}s")

    if gen_shape:
        generated_shapes = np.concatenate(generated_shapes, axis=0)
        original_shapes = np.concatenate(original_shapes, axis=0)
        gen_pgms = np.concatenate(gen_pgms, axis=0)
        gen_params = np.concatenate(gen_params, axis=0)

    return original_shapes, generated_shapes, gen_pgms, gen_params

def test_on_shapenet_data_for_variation(epoch, shapes, model, opt, gen_shape=False):
    """Run model on a single or batch numpy shapes for generating programs/params."""
    model.eval()
    gen_pgms = []
    gen_params = []

    if shapes.ndim == 3:
        shapes = np.expand_dims(shapes, axis=0)

    if shapes.ndim == 4:
        data = torch.tensor(shapes).float()
    else:
        raise ValueError(f"Expected 4D array (B,D,H,W), got {shapes.shape}")

    data = Variable(torch.unsqueeze(data, 1), requires_grad=False).cuda()  # (B,1,D,H,W)
    print(f"Input shape after unsqueeze: {data.shape}")

    out = model.decode(data)
    if opt.is_cuda:
        torch.cuda.synchronize()

    if gen_shape:
        for sample_idx in range(data.size(0)):
            _, save_pgms = torch.max(out[0].data[sample_idx:sample_idx+1], dim=3)
            save_pgms = save_pgms.cpu().numpy()
            save_params = out[1].data[sample_idx:sample_idx+1].cpu().numpy()
            gen_pgms.append(save_pgms)
            gen_params.append(save_params)

    if gen_shape:
        gen_pgms = np.concatenate(gen_pgms, axis=0)
        gen_params = np.concatenate(gen_params, axis=0)

    return gen_pgms, gen_params

def visualize_sequence(intermediate_shapes, save_path):
    """Save per-step 3D visualizations, skipping identical consecutive steps."""
    for step, shape in enumerate(intermediate_shapes):
        if step > 0 and np.array_equal(intermediate_shapes[step], intermediate_shapes[step - 1]):
            continue
        save_name = os.path.join(save_path, f'step_{step}.png')
        visualize_and_save_3d_no_rotate(shape.squeeze(), save_name)

def rotate_voxel(voxel, axes=(0, 1)):
    """Rotate voxel grid 180 degrees around specified axes."""
    return np.rot90(voxel, k=2, axes=axes)

def rotate_voxel_x(voxel, axes=(0, 2)):
    """Rotate voxel grid 180 degrees around the X-axis."""
    return np.rot90(voxel, k=2, axes=axes)

def generate_random_sequences(pgms, params, num_random_sequences=4):
    """Produce shuffled program sequences (simple random permutation of steps)."""
    random_sequences = []
    for _ in range(num_random_sequences):
        combined = list(zip(pgms, params))
        random.shuffle(combined)
        shuffled_pgms, shuffled_params = zip(*combined)
        random_sequences.append((list(shuffled_pgms), list(shuffled_params)))
    return random_sequences

def visualize_and_save_3d_no_rotate(voxels, file_name):
    """Rotate voxel for better view, extract mesh with marching cubes, and save an image."""
    rotated_voxels = np.rot90(voxels, k=1, axes=(1, 2))
    rotated_voxels = np.rot90(rotated_voxels, k=1, axes=(0, 2))

    voxel_min, voxel_max = rotated_voxels.min(), rotated_voxels.max()
    if voxel_min == voxel_max:
        print(f"No valid surface found in voxel data for {file_name}, skipping visualization.")
        return

    level = 0.5 if voxel_min <= 0.5 <= voxel_max else (voxel_min + voxel_max) / 2.0
    verts, faces, _, _ = measure.marching_cubes(rotated_voxels, level=level)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = zip(*verts)
    ax.plot_trisurf(x, y, faces, z, color='b', lw=1)
    ax.view_init(elev=30, azim=30)
    ax.grid(False)
    ax.axis('off')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_and_save_3d(voxels, file_name):
    """Rotate voxels 180 degrees and save a rendered mesh image."""
    rotated_voxels = rotate_voxel(voxels, axes=(0, 1))
    verts, faces, _, _ = measure.marching_cubes(rotated_voxels, level=0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = zip(*verts)
    ax.plot_trisurf(x, y, faces, z, color='b', lw=1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    plt.savefig(file_name)
    plt.close()

def run():
    """Main entry: load data & model, generate shapes, save programs/images and variations."""
    opt = parse_argument()
    set_seed(42)

    # Create output folders
    for path in [opt.prog_save_path, opt.imgs_save_path, opt.seq_imgs_save_path]:
        os.makedirs(path, exist_ok=True)
    rhino_commands_path = os.path.join(opt.save_path, 'rhino_commands')
    os.makedirs(rhino_commands_path, exist_ok=True)

    print('========= arguments =========')
    for key, val in vars(opt).items():
        print(f"{key:20} {val}")
    print('========= arguments =========')

    # Data loader
    test_set = ShapeNet3D(opt.data)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    # Load model
    ckpt = torch.load(opt.model)
    model = BlockOuterNet(ckpt['opt'])
    model.load_state_dict(ckpt['model'])

    if opt.is_cuda:
        model = model.cuda()
        cudnn.benchmark = True

    # Prepare interpreter (used for converting to DSL)
    interpreter = Interpreter(translate, rotate, end)

    # Generate shapes and programs
    ori_shapes, gen_shapes, pgms, params = test_on_shapenet_data(
        epoch=0, test_loader=test_loader, model=model, opt=opt, gen_shape=True
    )
    print(f"Original shapes: {ori_shapes.shape}, Generated shapes: {gen_shapes.shape}")
    IoU = BatchIoU(ori_shapes, gen_shapes)
    print(f"Calculated IoU: {IoU}")
    print("Mean IoU: {:.3f}".format(IoU.mean()))

    # Save generated programs (DSL) and Rhino conversions
    if opt.save_prog:
        for i in range(min(len(gen_shapes), opt.num_render)):
            program = interpreter.interpret(np.array(pgms[i]), np.array(params[i]))
            dsl_save_file = os.path.join(opt.prog_save_path, f'{i}.txt')
            with open(dsl_save_file, 'w') as out:
                out.write(program)
            rhino_file_path = os.path.join(rhino_commands_path, f'{i}_rhino.txt')
            try:
                convert_file(dsl_save_file, rhino_file_path)
            except Exception as e:
                print(f"Error converting DSL to Rhino for file {dsl_save_file}: {e}")

    # Save reconstructed images and some random permutations
    if opt.save_img:
        data = gen_shapes.transpose((0, 3, 2, 1))
        data = np.flip(data, axis=2)
        for i in range(min(len(gen_shapes), opt.num_render)):
            voxels = data[i]
            save_name = os.path.join(opt.imgs_save_path, f'{i}.png')
            visualize_and_save_3d(voxels, save_name)

            # Generate and save shuffled program sequences
            random_sequences = generate_random_sequences(pgms[i], params[i], num_random_sequences=2)
            for j, (random_pgms, random_params) in enumerate(random_sequences):
                random_program = interpreter.interpret(np.array(random_pgms), np.array(random_params))
                random_dsl_file = os.path.join(opt.prog_save_path, f'random_{i}_{j}.txt')
                with open(random_dsl_file, 'w') as out:
                    out.write(random_program)

                random_rhino_file = os.path.join(rhino_commands_path, f'random_{i}_{j}_rhino.txt')
                random_sequence_save_path = os.path.join(opt.seq_imgs_save_path, f'random_sample_{i}_{j}')
                os.makedirs(random_sequence_save_path, exist_ok=True)
                random_intermediate_shapes = execute_shape_program_with_trace(random_pgms, random_params, i, random_sequence_save_path)

                try:
                    convert_file(random_dsl_file, random_rhino_file)
                except Exception as e:
                    print(f"Error converting DSL to Rhino for file {random_dsl_file}: {e}")

    # Create variations of generated shapes, save images and sequences, and save DSL + Rhino
    modified_shapes = []
    for i, (shape, pgm, param) in enumerate(zip(gen_shapes, pgms, params)):
        try:
            pgm_new, param_new = execute_shape_program_yg(pgm, param)
            modified_variations = modify_shape(shape, pgm_new, param_new, pgm, param, model, opt)

            for j, modified_shape in enumerate(modified_variations):
                modified_shapes.append(modified_shape)
                save_name = os.path.join(opt.imgs_save_path, f'modified_{i}_{j}.png')
                visualize_and_save_3d_no_rotate(modified_shape, save_name)

                gen_pgms_new, gen_params_new = test_on_shapenet_data_for_variation(
                    epoch=0, shapes=modified_shape.squeeze(), model=model, opt=opt, gen_shape=True
                )
                sequence_save_path = os.path.join(opt.seq_imgs_save_path, f'modified_sample_{i}_{j}')
                os.makedirs(sequence_save_path, exist_ok=True)

                intermediate_shapes = execute_shape_program_with_trace(gen_pgms_new[0], gen_params_new[0], i, sequence_save_path)
                visualize_sequence(intermediate_shapes, sequence_save_path)

                if opt.save_prog:
                    modified_program = interpreter.interpret(np.array(gen_pgms_new[0]), np.array(gen_params_new[0]))
                    dsl_save_file = os.path.join(opt.prog_save_path, f'modified_{i}_{j}.txt')
                    with open(dsl_save_file, 'w') as out:
                        out.write(modified_program)
                    rhino_file_path = os.path.join(rhino_commands_path, f'modified_{i}_{j}_rhino.txt')
                    try:
                        convert_file(dsl_save_file, rhino_file_path)
                    except Exception as e:
                        print(f"Error converting DSL to Rhino for file {dsl_save_file}: {e}")

        except Exception as e:
            print(f"Error processing shape {i}: {e}")
            continue

    # Save all modified shapes to an HDF5 file
    save_file = os.path.join(opt.save_path, 'modified_shapes.h5')
    with h5py.File(save_file, 'w') as f:
        f['data'] = np.array(modified_shapes)

if __name__ == '__main__':
    run()
# ...existing code...
